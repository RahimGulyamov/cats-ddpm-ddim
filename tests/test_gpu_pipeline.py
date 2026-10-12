"""Properties that can be checked on CPU before a four-GPU run."""
import copy
import email
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
from diffusion.advanced import QualityUNet
from diffusion.distributed import BatchStream
from scripts.check_cuda import probe
from scripts.download_wheels import active_dependencies, command, target_environment


class GPUPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)

    def test_rank_streams_partition_global_batch_and_resume(self):
        streams = [BatchStream(144, 8, rank, 4, 2026) for rank in range(4)]
        for _ in range(7):
            batches = [stream.next() for stream in streams]
            self.assertEqual(len(torch.cat(batches).unique()), 32)
        for rank, stream in enumerate(streams):
            resumed = BatchStream(144, 8, rank, 4, 2026, **stream.state_dict())
            for _ in range(12):
                torch.testing.assert_close(resumed.next(), stream.next(), atol=0, rtol=0)

    def test_average_rank_gradients_equals_global_mean_with_accumulation(self):
        global_model = QualityUNet(8, (1, 2), attention_levels=(1,), attention_heads=2,
                                  dropout=0, gradient_checkpointing=False)
        torch.nn.init.normal_(global_model.output[-1].weight, std=0.01)
        local_model = copy.deepcopy(global_model)
        images, target = torch.randn(16, 3, 16, 16), torch.randn(16, 3, 16, 16)
        times = torch.arange(16)
        torch.nn.functional.mse_loss(global_model(images, times), target).backward()
        totals = {name: torch.zeros_like(parameter) for name, parameter in local_model.named_parameters()}
        for rank in range(4):
            local_model.zero_grad(set_to_none=True)
            for micro in range(2):
                start = rank * 4 + micro * 2
                loss = torch.nn.functional.mse_loss(local_model(images[start:start + 2], times[start:start + 2]), target[start:start + 2])
                (loss / 2).backward()
            for name, parameter in local_model.named_parameters():
                totals[name] += parameter.grad / 4
        for name, parameter in global_model.named_parameters():
            torch.testing.assert_close(totals[name], parameter.grad, atol=2e-6, rtol=2e-5)

    def test_activation_checkpointing_preserves_dropout_gradients(self):
        first = QualityUNet(8, (1, 2), attention_levels=(1,), attention_heads=2,
                            dropout=0.1, gradient_checkpointing=True)
        torch.nn.init.normal_(first.output[-1].weight, std=0.01)
        second = copy.deepcopy(first)
        second.gradient_checkpointing = False
        images, target = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
        times = torch.tensor([5, 10])
        state = torch.get_rng_state()
        loss = (first(images, times) - target).square().mean()
        loss.backward()
        torch.set_rng_state(state)
        comparison = (second(images, times) - target).square().mean()
        comparison.backward()
        torch.testing.assert_close(loss, comparison, atol=0, rtol=0)
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertIsNotNone(a.grad)
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)

    def test_linux_cuda_dependencies_are_not_lost_on_mac(self):
        metadata = email.message.Message()
        metadata["Name"] = "torch"
        metadata["Version"] = "2.9.0+cu128"
        metadata["Requires-Dist"] = 'nvidia-cublas-cu12==12.8.4.1; platform_system == "Linux" and platform_machine == "x86_64"'
        metadata["Requires-Dist"] = 'triton==3.5.0; platform_system == "Linux"'
        metadata["Requires-Dist"] = 'optional-test; extra == "test"'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "torch.whl"
            with zipfile.ZipFile(path, "w") as wheel:
                wheel.writestr("torch-2.9.0.dist-info/METADATA", metadata.as_string())
            name, version, dependencies = active_dependencies(path, target_environment("3.11"))
        self.assertEqual((name, version), ("torch", "2.9.0+cu128"))
        self.assertEqual([r.name for r in dependencies], ["nvidia-cublas-cu12", "triton"])

    def test_wheel_command_contains_intermediate_manylinux_tags(self):
        args = command("3.11", Path("wheels"), Path("requirements"), "cu128", 28)
        self.assertIn("manylinux_2_27_x86_64", args)
        self.assertIn("manylinux_2_28_x86_64", args)
        self.assertIn("cp311", args)
        self.assertIn("--no-deps", args)

    def test_runtime_probe_performs_finite_optimizer_update_on_cpu(self):
        result = probe(torch.device("cpu"), precision="fp32")
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
        self.assertIn("AdamW", result["operations"])


if __name__ == "__main__":
    unittest.main()
