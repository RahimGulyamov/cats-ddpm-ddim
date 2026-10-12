"""Download Linux x86_64 wheels on any online host, including an Apple Silicon Mac.

Resolve platform markers from wheel metadata for Linux, not from the host OS.
This matters for Torch's conditional NVIDIA and Triton dependencies.
"""
import argparse
import email
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version
except ImportError:
    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def target_environment(python_version):
    environment = default_environment()
    environment.update({"sys_platform": "linux", "platform_system": "Linux", "platform_machine": "x86_64",
                        "os_name": "posix", "python_version": python_version,
                        "python_full_version": python_version + ".0", "implementation_name": "cpython",
                        "implementation_version": python_version + ".0", "extra": ""})
    return environment


def active_dependencies(wheel, environment):
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError(f"Expected one METADATA in {wheel.name}")
        metadata = email.message_from_bytes(archive.read(names[0]))
    dependencies = []
    for value in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(value)
        if requirement.marker is None or requirement.marker.evaluate(environment):
            if requirement.url or requirement.extras:
                raise ValueError(f"Dependency needs explicit handling: {requirement}")
            dependencies.append(requirement)
    return canonicalize_name(metadata["Name"]), metadata["Version"], dependencies


def add_requirement(requirements, value):
    requirement = Requirement(value)
    if requirement.url or requirement.extras:
        raise ValueError("Only standard binary distributions without extras are supported")
    name = canonicalize_name(requirement.name)
    requirements.setdefault(name, set()).update(str(s) for s in requirement.specifier)


def requirement_lines(requirements):
    return [name + ",".join(sorted(specs)) for name, specs in sorted(requirements.items())]


def command(python_version, destination, requirements_file, torch_index=None, glibc_minor=28):
    digits = python_version.replace(".", "")
    args = [sys.executable, "-m", "pip", "download", "--index-url", "https://pypi.org/simple",
            "--only-binary=:all:", "--no-deps",
            "--python-version", python_version, "--implementation", "cp", "--abi", "cp" + digits,
            "--dest", str(destination), "--requirement", str(requirements_file)]
    platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(glibc_minor, 4, -1)]
    platforms += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    for value in platforms:
        args.extend(("--platform", value))
    if torch_index:
        args.extend(("--extra-index-url", f"https://download.pytorch.org/whl/{torch_index}"))
    return args


def download(options):
    if options.output.exists() and any(options.output.iterdir()):
        raise ValueError("Wheelhouse already contains files; use a new --output")
    environment = target_environment(options.python_version)
    requirements = {}
    files = options.requirements or [ROOT / "requirements-gpu.txt"]
    if options.include_reference:
        files.append(ROOT / "requirements-reference.txt")
    for path in files:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                add_requirement(requirements, line)
    if options.with_torch:
        add_requirement(requirements, f"torch=={options.torch_version}+{options.cuda_index}")
    index = options.cuda_index if options.with_torch else None
    with tempfile.TemporaryDirectory(prefix="linux-wheels-") as temporary:
        temporary = Path(temporary)
        for iteration in range(12):
            request_file = temporary / "requested.txt"
            request_file.write_text("\n".join(requirement_lines(requirements)) + "\n")
            destination = temporary / f"pass-{iteration}"
            destination.mkdir()
            args = command(options.python_version, destination, request_file, index,
                           int(options.glibc_version.split(".")[1]))
            if options.dry_run:
                print(json.dumps({"target": environment, "initial_requirements": requirement_lines(requirements),
                                  "command": args, "note": "Further passes resolve active Linux Requires-Dist recursively."}, indent=2))
                return
            print(f"Resolve pass {iteration + 1}: {len(requirements)} distributions", flush=True)
            subprocess.run(args, check=True)
            distributions = {}
            missing = False
            for wheel in destination.glob("*.whl"):
                name, version, dependencies = active_dependencies(wheel, environment)
                distributions[name] = {"version": version, "wheel": wheel, "dependencies": dependencies}
                for requirement in dependencies:
                    before = set(requirements.get(canonicalize_name(requirement.name), set()))
                    add_requirement(requirements, str(requirement))
                    if before != requirements[canonicalize_name(requirement.name)]:
                        missing = True
            if set(distributions) != set(requirements):
                missing = True
            for name, distribution in distributions.items():
                required = Requirement(name + ",".join(sorted(requirements[name])))
                if not required.specifier.contains(Version(distribution["version"]), prereleases=True):
                    missing = True
            if missing:
                shutil.rmtree(destination)
                continue
            options.output.mkdir(parents=True, exist_ok=True)
            records = []
            for name, distribution in sorted(distributions.items()):
                wheel = distribution["wheel"]
                shutil.copy2(wheel, options.output / wheel.name)
                records.append({"name": name, "version": distribution["version"], "file": wheel.name,
                                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()})
            manifest = {"target": {"python": options.python_version, "system": "Linux", "machine": "x86_64",
                                   "glibc": options.glibc_version},
                        "with_torch": options.with_torch, "with_reference": options.include_reference,
                        "cuda_index": options.cuda_index if options.with_torch else None, "wheels": records}
            (options.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
            (options.output / "locked.txt").write_text("\n".join(
                f"{r['name']}=={r['version']} --hash=sha256:{r['sha256']}" for r in records) + "\n")
            print(f"Ready: {len(records)} Linux wheels in {options.output}")
            return
        raise RuntimeError("Dependency closure did not converge; no incomplete wheelhouse was published")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-version", choices=("3.10", "3.11", "3.12"), required=True)
    parser.add_argument("--glibc-version", choices=("2.17", "2.28", "2.31", "2.35", "2.39"), default="2.28")
    parser.add_argument("--output", type=Path, default=Path("wheelhouse"))
    parser.add_argument("--with-torch", action="store_true")
    parser.add_argument("--torch-version", default="2.9.0")
    parser.add_argument("--cuda-index", choices=("cu118", "cu121", "cu124", "cu126", "cu128"), default="cu128")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--requirements", type=Path, action="append",
                        help="Download only these requirement files instead of requirements-gpu.txt; repeatable")
    parser.add_argument("--dry-run", action="store_true")
    download(parser.parse_args())
