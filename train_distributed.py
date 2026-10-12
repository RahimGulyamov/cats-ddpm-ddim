"""Offline, step-based diffusion training on one device or torchrun/DDP."""
import argparse
import contextlib
import copy
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from diffusion.artifacts import grid, hardware, save_checkpoint, write_json
from diffusion.data import load_images
from diffusion.distributed import BatchStream, Context
from diffusion.factory import create_model
from diffusion.journal import Journal
from diffusion.process import Diffusion, synchronize
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


def validate(config):
    options = config["training"]
    keys = ("max_updates", "batch_size_per_gpu", "accumulation_steps", "log_every", "validate_every",
            "checkpoint_every", "snapshot_every", "preview_every", "ddpm_preview_every", "preview_steps", "preview_count")
    for key in keys:
        if not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if options["precision"] not in ("fp32", "bf16", "fp16"):
        raise ValueError("precision must be fp32, bf16 or fp16")
    if not 0 <= options["warmup_updates"] < options["max_updates"]:
        raise ValueError("warmup_updates must be in [0,max_updates)")
    if options["learning_rate"] <= 0 or not 0 <= options["ema_decay"] < 1:
        raise ValueError("Invalid learning rate or EMA decay")
    if options["min_snr_gamma"] is not None and options["min_snr_gamma"] <= 0:
        raise ValueError("min_snr_gamma must be positive or null")
    if not 0 < config["val_fraction"] < 1 or config["cpu_threads"] < 1:
        raise ValueError("Invalid validation split or CPU thread count")
    divisor = 2 ** (len(config["model"]["channel_multipliers"]) - 1)
    if config["image_size"] < divisor or config["image_size"] % divisor:
        raise ValueError("image_size is incompatible with the U-Net")
    timesteps = config["diffusion"]["timesteps"]
    if options["preview_steps"] > timesteps:
        raise ValueError("preview_steps exceeds training timesteps")
    levels = options["validation_levels"]
    if not levels or len(set(levels)) != len(levels) or any(t < 0 or t >= timesteps for t in levels):
        raise ValueError("Invalid validation_levels")


def rng_state(device):
    return {"cpu": torch.get_rng_state(),
            "device": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None}


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["device"], device)


@torch.inference_mode()
def evaluate(model, process, validation, levels, batch_size, seed):
    model.eval()
    generator = torch.Generator().manual_seed(seed + 17)
    noise = torch.randn(validation.shape, generator=generator)
    result = {}
    for timestep in levels:
        noise_sum, clean_sum, unclipped_sum = 0., 0., 0.
        for start in range(0, len(validation), batch_size):
            clean = validation[start:start + batch_size].to(process.device)
            eps = noise[start:start + batch_size].to(process.device)
            times = torch.full((len(clean),), timestep, dtype=torch.long, device=process.device)
            noisy = process.add_noise(clean, times, eps)
            predicted = model(noisy, times).float()
            a = process.alpha_bar[timestep]
            restored = (noisy - (1 - a).sqrt() * predicted) / a.sqrt()
            noise_sum += F.mse_loss(predicted, eps).item() * len(clean)
            clean_sum += F.mse_loss(restored.clamp(-1, 1), clean).item() * len(clean)
            unclipped_sum += F.mse_loss(restored, clean).item() * len(clean)
        result[f"noise_mse_t{timestep}"] = noise_sum / len(validation)
        result[f"x0_mse_clipped_t{timestep}"] = clean_sum / len(validation)
        result[f"x0_mse_unclipped_t{timestep}"] = unclipped_sum / len(validation)
    result["val_noise_mse"] = sum(result[f"noise_mse_t{t}"] for t in levels) / len(levels)
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("Nonfinite validation metrics")
    return result


def run(config, resume=None, stop_after_updates=None):
    validate(config)
    options = config["training"]
    if stop_after_updates is not None and stop_after_updates < 1:
        raise ValueError("--stop-after-updates must be positive")
    torch.set_num_threads(config["cpu_threads"])
    context = Context(config["device"])
    journal = None
    try:
        device, rank, world = context.device, context.rank, context.world_size
        if device.type == "cuda":
            # Seeds are reproducible within a fixed setup; cross-device bitwise identity is not promised.
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            if options["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
                raise ValueError("GPU does not support BF16; choose fp32 or fp16")
        elif options["precision"] != "fp32":
            raise ValueError("Use fp32 for CPU/DDP verification")
        output = Path(config["output_dir"])
        def prepare_output():
            if (output / "last.pt").exists() and resume is None:
                raise ValueError("Run exists; use --resume or --output")
            if (output / "train.csv").exists() and resume is None:
                raise ValueError("Run logs already exist; use a new --output")
            output.mkdir(parents=True, exist_ok=True)
            (output / "previews").mkdir(exist_ok=True)
            (output / "snapshots").mkdir(exist_ok=True)
        context.leader_task(prepare_output)
        training, validation, dataset = load_images(config["data_dir"], config["image_size"],
                                                   config["seed"], config["val_fraction"])
        fingerprints = context.gather(dataset["sha256"])
        if len(set(fingerprints)) != 1:
            raise ValueError("Dataset differs between ranks")
        torch.manual_seed(config["seed"])
        raw_model = create_model(config["model"]).to(device)
        model = raw_model
        if world > 1:
            model = DistributedDataParallel(raw_model,
                        device_ids=[context.local_rank] if device.type == "cuda" else None,
                        broadcast_buffers=False, gradient_as_bucket_view=True)
        ema = copy.deepcopy(raw_model).eval().requires_grad_(False) if rank == 0 else None
        process = Diffusion(**config["diffusion"], device=device)
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=options["learning_rate"], weight_decay=0.01)
        def multiplier(step):
            warmup = options["warmup_updates"]
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = min(1., (step - warmup) / max(1, options["max_updates"] - warmup))
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and options["precision"] == "fp16")
        torch.manual_seed(config["seed"] + rank * 100003 + 31)
        update, skipped, seen_microbatches, best_val, elapsed_before = 0, 0, 0, float("inf"), 0.
        train_history, val_history, stream_state = [], [], {"cycle": 0, "offset": 0}
        if resume:
            state = torch.load(resume, map_location="cpu", weights_only=True, mmap=True)
            if state.get("format_version") != 2:
                raise ValueError("DDP resume requires a format_version=2 training checkpoint")
            for key in ("seed", "image_size", "val_fraction", "model", "diffusion", "training"):
                if config[key] != state["config"][key]:
                    raise ValueError(f"Resume requires identical {key}")
            if world != state["world_size"] or dataset["sha256"] != state["dataset"]["sha256"]:
                raise ValueError("Resume requires the same world size and dataset")
            raw_model.load_state_dict(state["model"])
            if rank == 0:
                ema.load_state_dict(state["ema"])
                train_history, val_history = state["train_history"], state["validation_history"]
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            update, skipped, seen_microbatches = state["updates"], state["skipped_updates"], state["seen_microbatches"]
            best_val, elapsed_before = state["best_val_mse"], state["elapsed_seconds"]
            stream_state = state["stream"]
            restore_rng(state["rng_by_rank"][rank], device)
            del state
        end_update = min(options["max_updates"], stop_after_updates or options["max_updates"])
        if end_update <= update:
            raise ValueError("No updates remaining; --stop-after-updates is a total limit")
        stream = BatchStream(len(training), options["batch_size_per_gpu"], rank, world,
                             config["seed"] + 100, **stream_state)
        global_batch = options["batch_size_per_gpu"] * world * options["accumulation_steps"]
        parameters = sum(p.numel() for p in raw_model.parameters())
        environment = {**hardware(device), "world_size": world, "global_batch_size": global_batch,
                       "batch_size_per_gpu": options["batch_size_per_gpu"], "precision": options["precision"],
                       "cuda": torch.version.cuda, "parameters": parameters,
                       "dataset_train": len(training), "dataset_validation": len(validation)}
        def start_journal():
            nonlocal journal
            journal = Journal(output, options["tensorboard"], train_history, val_history)
            write_json(output / "environment.json", environment)
            write_json(output / "split.json", dataset)
            (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")
            grid(training[:16], title="Train data | 128px GPU experiment").save(output / "dataset.png")
            journal.event("resume" if resume else "start", **environment, start_update=update)
            print(f"{parameters:,} parameters | {world} ranks | global batch {global_batch} | "
                  f"{options['precision']} | updates {update}/{options['max_updates']}", flush=True)
        context.leader_task(start_journal)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        dtype = torch.bfloat16 if options["precision"] == "bf16" else torch.float16
        amp_enabled = device.type == "cuda" and options["precision"] != "fp32"
        while update < end_update:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tick = time.perf_counter()
            statistics = torch.zeros(18, dtype=torch.float64, device=device)
            for micro in range(options["accumulation_steps"]):
                indices = stream.next()
                clean = training[indices].to(device)
                flips = torch.rand(len(clean), device=device) < 0.5
                clean = torch.where(flips[:, None, None, None], clean.flip(-1), clean)
                noise = torch.randn_like(clean)
                times = torch.randint(process.timesteps, (len(clean),), device=device)
                noisy = process.add_noise(clean, times, noise)
                synchronize_gradients = micro == options["accumulation_steps"] - 1
                manager = model.no_sync() if world > 1 and not synchronize_gradients else contextlib.nullcontext()
                with manager:
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
                        predicted = model(noisy, times)
                        per_image_mse = (predicted.float() - noise.float()).square().flatten(1).mean(1)
                        weights = torch.ones_like(per_image_mse)
                        if options["min_snr_gamma"] is not None:
                            a = process.alpha_bar[times]
                            snr = a / (1 - a)
                            weights = torch.minimum(snr, torch.full_like(snr, options["min_snr_gamma"])) / snr
                        optimized_loss = (per_image_mse * weights).mean()
                        loss = optimized_loss / options["accumulation_steps"]
                    if not context.all_finite(loss.detach()):
                        raise RuntimeError("Nonfinite loss on a rank")
                    scaler.scale(loss).backward()
                with torch.no_grad():
                    statistics[0] += per_image_mse.double().sum()
                    statistics[1] += optimized_loss.double() * len(clean)
                    bins = (times * 8 // process.timesteps).clamp(max=7)
                    statistics[2:10].scatter_add_(0, bins, per_image_mse.detach().double())
                    statistics[10:18].scatter_add_(0, bins, torch.ones_like(per_image_mse, dtype=torch.float64))
                seen_microbatches += 1
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.)
            finite_gradient = context.all_finite(gradient_norm)
            if not finite_gradient and not scaler.is_enabled():
                raise RuntimeError("Nonfinite gradient in BF16/FP32")
            if not finite_gradient:
                scaler.update(new_scale=scaler.get_scale() / 2)
                skipped += 1
                if rank == 0:
                    journal.event("fp16_overflow", step=update, skipped_updates=skipped)
                if skipped > 100:
                    raise RuntimeError("Too many FP16 overflows; use BF16")
                continue
            learning_rate = optimizer.param_groups[0]["lr"]
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                skipped += 1
                if rank == 0:
                    journal.event("fp16_overflow", step=update, skipped_updates=skipped)
                if skipped > 100:
                    raise RuntimeError("Too many FP16 overflows; use BF16")
                continue
            scheduler.step()
            update += 1
            if rank == 0:
                decay = min(options["ema_decay"], (1 + update) / (10 + update))
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()), list(raw_model.parameters()), 1 - decay)
            context.sum(statistics)
            memory = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**20,
                                   torch.cuda.max_memory_reserved(device) / 2**20] if device.type == "cuda" else [0., 0.],
                                  device=device)
            if world > 1:
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            synchronize(device)
            seconds = time.perf_counter() - tick
            measured = statistics.cpu().tolist()
            row = {"step": update, "examples_seen": seen_microbatches * options["batch_size_per_gpu"] * world,
                   "epoch_equivalent": seen_microbatches * options["batch_size_per_gpu"] * world / len(training),
                   "train_mse": measured[0] / global_batch, "optimized_loss": measured[1] / global_batch,
                   "learning_rate": learning_rate, "gradient_norm": float(gradient_norm),
                   "update_seconds": seconds, "images_per_second": global_batch / seconds,
                   "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                   "peak_allocated_mb": memory[0].item(), "peak_reserved_mb": memory[1].item(),
                   "skipped_updates": skipped}
            # Only rank 0 writes; propagate disk/writer failures before the next collective.
            context.leader_task(lambda: journal.row("train", row))
            if update == 1 or update % options["log_every"] == 0 or update == end_update:
                def log_update():
                    journal.event("train", **row)
                    journal.scalars("train", row, update)
                    for i in range(8):
                        if measured[10 + i] > 0:
                            journal.scalars("train_noise_bins", {f"bin_{i}": measured[2 + i] / measured[10 + i]}, update)
                    print(f"Update {update}/{options['max_updates']} | noise MSE {row['train_mse']:.5f} | "
                          f"lr {learning_rate:.3g} | grad {row['gradient_norm']:.3f} | "
                          f"{row['images_per_second']:.1f} img/s | peak {row['peak_allocated_mb']:.0f} MiB", flush=True)
                context.leader_task(log_update)
            final = update == end_update
            is_best = False
            if update % options["validate_every"] == 0 or final:
                metrics = context.leader_task(lambda: evaluate(ema, process, validation,
                              options["validation_levels"], options["batch_size_per_gpu"], config["seed"]))
                is_best = metrics["val_noise_mse"] < best_val
                best_val = min(best_val, metrics["val_noise_mse"])
                def log_validation():
                    entry = {"step": update, **metrics}
                    journal.row("validation", entry)
                    journal.event("validation", **entry)
                    journal.scalars("validation", metrics, update)
                    journal.plots()
                    print(f"Validation {update}: noise MSE {metrics['val_noise_mse']:.5f}", flush=True)
                context.leader_task(log_validation)
            if update % options["preview_every"] == 0 or final:
                def preview():
                    size = config["image_size"]
                    shape = (options["preview_count"], 3, size, size)
                    methods = [("ddim", options["preview_steps"])]
                    if update % options["ddpm_preview_every"] == 0 or final:
                        methods.append(("ddpm", process.timesteps))
                    for method, steps in methods:
                        pictures, duration, _ = process.sample(ema, shape, method, steps, config["seed"])
                        canvas = grid(pictures, title=f"Update {update} | EMA | {method.upper()} {steps}")
                        canvas.save(output / "previews" / f"step-{update:07d}-{method}.png")
                        journal.event("preview", step=update, method=method, diffusion_steps=steps,
                                      images=len(pictures), seconds=duration, seed=config["seed"])
                context.leader_task(preview)
            should_save = update % options["checkpoint_every"] == 0 or final or is_best
            if should_save:
                states = context.gather(rng_state(device))
                def checkpoint():
                    elapsed = elapsed_before + time.perf_counter() - started
                    compact = {"format_version": 2, "config": config, "dataset": dataset,
                               "ema": ema.state_dict(), "epoch": stream.cycle,
                               "updates": update, "best_val_mse": best_val, "weights": "ema"}
                    state = {**compact, "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                             "rng_by_rank": states, "stream": stream.state_dict(), "world_size": world,
                             "skipped_updates": skipped, "seen_microbatches": seen_microbatches,
                             "elapsed_seconds": elapsed, "train_history": journal.training,
                             "validation_history": journal.validation}
                    save_checkpoint(output / "last.pt", state)
                    if is_best:
                        save_checkpoint(output / "best.pt", compact)
                    if update % options["snapshot_every"] == 0 or final:
                        save_checkpoint(output / "snapshots" / f"step-{update:07d}.pt", compact)
                    journal.event("checkpoint", step=update, best=is_best)
                context.leader_task(checkpoint)
        def finish():
            summary = {"status": "complete" if update == options["max_updates"] else "paused",
                       "updates": update, "max_updates": options["max_updates"], "parameters": parameters,
                       "best_val_mse": best_val, "last_train_mse": journal.training[-1]["train_mse"],
                       "world_size": world, "global_batch_size": global_batch, "skipped_updates": skipped,
                       "wall_seconds": elapsed_before + time.perf_counter() - started,
                       "environment": environment}
            write_json(output / "summary.json", summary)
            journal.plots()
            journal.event("complete" if summary["status"] == "complete" else "paused", **summary)
        context.leader_task(finish)
    finally:
        if journal:
            journal.close()
        context.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--local-rank", "--local_rank", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.output:
        config["output_dir"] = str(args.output)
    if args.device:
        config["device"] = args.device
    run(config, args.resume, args.stop_after_updates)
