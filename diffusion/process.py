"""Forward noise and reverse samplers. No scheduler library hides the equations."""
import math
import time

import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Diffusion:
    def __init__(self, timesteps=1000, schedule="linear", device="cpu"):
        if timesteps < 2:
            raise ValueError("timesteps must be at least two")
        self.timesteps = timesteps
        self.device = torch.device(device)
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float64)
        elif schedule == "cosine":
            grid = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
            cumulative = torch.cos((grid + 0.008) / 1.008 * math.pi / 2).square()
            cumulative /= cumulative[0].clone()
            betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
        else:
            raise ValueError("schedule must be linear or cosine")
        self.betas = betas.float().to(self.device)
        self.alphas = (1 - betas).float().to(self.device)
        cumulative = (1 - betas).cumprod(0)
        self.alpha_bar = cumulative.float().to(self.device)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), cumulative[:-1]))
        # Precompute posterior coefficients in float64 to avoid cancellation near t=0.
        self.posterior_x0 = (previous.sqrt() * betas / (1 - cumulative)).float().to(self.device)
        self.posterior_xt = ((1 - betas).sqrt() * (1 - previous) / (1 - cumulative)).float().to(self.device)
        self.posterior_variance = (betas * (1 - previous) / (1 - cumulative)).float().to(self.device)

    def add_noise(self, clean, timesteps, noise):
        a = self.alpha_bar[timesteps][:, None, None, None]
        return a.sqrt() * clean + (1 - a).sqrt() * noise

    def clean_prediction(self, x, epsilon, t):
        a = self.alpha_bar[t]
        return ((x - (1 - a).sqrt() * epsilon) / a.sqrt()).clamp(-1, 1)

    def ddpm_step(self, x, epsilon, t, noise):
        """q(x_{t-1}|x_t,x0_hat): posterior mean and fixed-small variance."""
        clean = self.clean_prediction(x, epsilon, t)
        mean = self.posterior_x0[t] * clean + self.posterior_xt[t] * x
        variance = self.posterior_variance[t]
        return mean + variance.clamp(min=0).sqrt() * noise if t > 0 else mean

    def ddim_step(self, x, epsilon, t, previous_t, eta=0., noise=None):
        """DDIM equation (12); eta=0 removes fresh noise and permits skipped steps."""
        a = self.alpha_bar[t]
        previous = self.alpha_bar[previous_t] if previous_t >= 0 else x.new_tensor(1.)
        clean = self.clean_prediction(x, epsilon, t)
        # Clipping x0 changes the decomposition: keep epsilon consistent with it.
        epsilon = (x - a.sqrt() * clean) / (1 - a).sqrt()
        variance = ((1 - previous) / (1 - a) * (1 - a / previous)).clamp(min=0)
        sigma = eta * variance.sqrt()
        result = previous.sqrt() * clean + (1 - previous - sigma.square()).clamp(min=0).sqrt() * epsilon
        if eta > 0 and previous_t >= 0:
            if noise is None:
                raise ValueError("stochastic DDIM needs a noise tensor")
            result = result + sigma * noise
        return result

    @torch.inference_mode()
    def sample(self, model, shape, method="ddim", steps=100, seed=2026, eta=0., record=False):
        if method not in ("ddpm", "ddim"):
            raise ValueError("method must be ddpm or ddim")
        if not 1 <= steps <= self.timesteps:
            raise ValueError("steps must be in [1, training timesteps]")
        if method == "ddpm" and steps != self.timesteps:
            raise ValueError("This ancestral DDPM uses all training steps; use DDIM for fewer steps")
        if not 0 <= eta <= 1:
            raise ValueError("eta must be in [0, 1]")
        model.eval()
        # CPU generators also work on MPS, and isolate generation from training RNG.
        initial_rng = torch.Generator().manual_seed(seed)
        reverse_rng = torch.Generator().manual_seed(seed + 1)
        x = torch.randn(shape, generator=initial_rng).to(self.device)
        timeline = (list(range(self.timesteps - 1, -1, -1)) if method == "ddpm" else
                    torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist())
        frames = [(self.timesteps, x.cpu())] if record else []
        synchronize(self.device)
        started = time.perf_counter()
        for i, t in enumerate(timeline):
            times = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            epsilon = model(x, times)
            stochastic = (method == "ddpm" and t > 0) or (method == "ddim" and eta > 0 and i < steps - 1)
            noise = torch.randn(shape, generator=reverse_rng).to(self.device) if stochastic else None
            if method == "ddpm":
                x = self.ddpm_step(x, epsilon, t, noise)
            else:
                previous = timeline[i + 1] if i < steps - 1 else -1
                x = self.ddim_step(x, epsilon, t, previous, eta, noise)
            if record and ((i + 1) % max(1, steps // 50) == 0 or i == steps - 1):
                frames.append((t, x.cpu()))
        synchronize(self.device)
        elapsed = time.perf_counter() - started
        return x.cpu(), elapsed, frames
