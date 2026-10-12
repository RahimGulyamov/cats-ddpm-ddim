"""Distributed context and a deterministic, resumable stream of rank-local batches."""
import math
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from .artifacts import device_for


class Context:
    def __init__(self, device_name):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = device_for(device_name)
        if self.device.type == "cuda":
            if self.local_rank >= torch.cuda.device_count():
                raise ValueError("LOCAL_RANK exceeds the available GPUs")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        if self.world_size > 1:
            if self.device.type not in ("cpu", "cuda"):
                raise ValueError("Distributed training supports CPU/Gloo and CUDA/NCCL")
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo",
                                    timeout=timedelta(minutes=120))

    def sum(self, value):
        if self.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def all_finite(self, value):
        flag = torch.isfinite(value).all().to(dtype=torch.int32, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def leader_task(self, callback):
        """Propagate callback failures so other ranks do not hang at the next update."""
        result, error = None, None
        if self.rank == 0:
            try:
                result = callback()
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"
        payload = [result, error]
        if self.world_size > 1:
            dist.broadcast_object_list(payload, src=0)
        if payload[1] is not None:
            raise RuntimeError(f"Rank 0 task failed: {payload[1]}")
        return payload[0]

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()


class BatchStream:
    def __init__(self, size, local_batch, rank, world_size, seed, cycle=0, offset=0):
        self.size, self.local_batch = size, local_batch
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.global_microbatch = local_batch * world_size
        self.padded_size = math.ceil(size / self.global_microbatch) * self.global_microbatch
        self.cycle, self.offset = cycle, offset
        self._order, self._cached_cycle = None, None
        if not 0 <= offset < self.padded_size or offset % self.global_microbatch:
            raise ValueError("Invalid saved batch-stream offset")

    def next(self):
        if self._cached_cycle != self.cycle:
            generator = torch.Generator().manual_seed(self.seed + self.cycle)
            order = torch.randperm(self.size, generator=generator)
            self._order = order.repeat(math.ceil(self.padded_size / self.size))[:self.padded_size]
            self._cached_cycle = self.cycle
        start = self.offset + self.rank * self.local_batch
        indices = self._order[start:start + self.local_batch]
        self.offset += self.global_microbatch
        if self.offset == self.padded_size:
            self.cycle += 1
            self.offset = 0
        return indices

    def state_dict(self):
        return {"cycle": self.cycle, "offset": self.offset}
