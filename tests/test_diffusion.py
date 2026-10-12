"""Numerical properties of the diffusion equations, including boundary steps."""
import unittest

import torch
from diffusion.network import NoiseUNet
from diffusion.process import Diffusion


class DiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(8)
        self.process = Diffusion(timesteps=20)
        self.clean = torch.rand(2, 3, 8, 8) * 1.6 - 0.8
        self.noise = torch.randn_like(self.clean)

    def test_forward_distribution_and_oracle_inverse(self):
        t = torch.tensor([0, 19])
        noisy = self.process.add_noise(self.clean, t, self.noise)
        a = self.process.alpha_bar[t][:, None, None, None]
        torch.testing.assert_close((noisy - a.sqrt() * self.clean) / (1 - a).sqrt(), self.noise,
                                   atol=1e-5, rtol=1e-5)
        for index, timestep in enumerate(t.tolist()):
            restored = self.process.clean_prediction(noisy[index], self.noise[index], timestep)
            torch.testing.assert_close(restored, self.clean[index])

    def test_ddim_oracle_preserves_noise_path(self):
        x = self.process.add_noise(self.clean, torch.full((2,), 19), self.noise)
        for t, previous in [(19, 10), (10, 0), (0, -1)]:
            x = self.process.ddim_step(x, self.noise, t, previous)
            expected = self.clean if previous < 0 else self.process.add_noise(
                self.clean, torch.full((2,), previous), self.noise)
            torch.testing.assert_close(x, expected, atol=1e-5, rtol=1e-5)

    def test_ddpm_matches_stochastic_adjacent_ddim(self):
        for t in (1, 10, 19):
            x = self.process.add_noise(self.clean, torch.full((2,), t), self.noise)
            extra = torch.randn_like(x)
            ddpm = self.process.ddpm_step(x, self.noise, t, extra)
            ddim = self.process.ddim_step(x, self.noise, t, t - 1, eta=1., noise=extra)
            torch.testing.assert_close(ddpm, ddim, atol=1e-5, rtol=1e-5)

    def test_final_ddpm_step_has_no_noise(self):
        x = self.process.add_noise(self.clean, torch.zeros(2, dtype=torch.long), self.noise)
        first = self.process.ddpm_step(x, self.noise, 0, torch.randn_like(x))
        second = self.process.ddpm_step(x, self.noise, 0, torch.randn_like(x) * 100)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        torch.testing.assert_close(first, self.clean, atol=2e-4, rtol=2e-4)

    def test_schedules_monotone_and_valid(self):
        for schedule in ("linear", "cosine"):
            d = Diffusion(1000, schedule)
            self.assertTrue(torch.all((d.betas > 0) & (d.betas < 1)))
            self.assertTrue(torch.all(d.alpha_bar[1:] < d.alpha_bar[:-1]))
            self.assertLess(d.alpha_bar[-1].item(), 0.001)

    def test_unet_backward(self):
        model = NoiseUNet(8, (1, 2))
        prediction = model(torch.randn(2, 3, 16, 16), torch.tensor([2, 12]))
        self.assertEqual(prediction.shape, (2, 3, 16, 16))
        prediction.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_seeded_sampling_isolated_and_repeatable(self):
        model = NoiseUNet(8, (1, 2)).eval()
        rng = torch.get_rng_state().clone()
        for method, steps in (("ddim", 5), ("ddpm", 20)):
            first, _, _ = self.process.sample(model, (2, 3, 8, 8), method, steps, 42)
            second, _, _ = self.process.sample(model, (2, 3, 8, 8), method, steps, 42)
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            self.assertTrue(torch.isfinite(first).all())
            self.assertTrue(torch.all(first.abs() <= 1.001))
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)

    def test_invalid_sampling_parameters(self):
        model = NoiseUNet(8, (1, 2))
        for method, steps, eta in (("ddpm", 5, 0), ("ddim", 21, 0), ("ddim", 0, 0),
                                   ("other", 20, 0), ("ddim", 5, 2)):
            with self.assertRaises(ValueError):
                self.process.sample(model, (1, 3, 8, 8), method, steps, eta=eta)


if __name__ == "__main__":
    unittest.main()
