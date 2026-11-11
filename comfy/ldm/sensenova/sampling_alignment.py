"""Noise, CFG and Euler updates for reproducing SenseNova-U1 sampling."""

import torch

from .sampling import resolution_noise_scale


class U1AlignedNoise:
    """Replay SenseNova-U1's BF16 noise stream on the inference device.

    Skip draws for previously generated images so interleave iterations share one stream.
    """

    def __init__(self, seed, device, previous_sizes, noise_scale=1.0):
        self.seed = seed
        self.device = device
        self.previous_sizes = previous_sizes
        self.noise_scale = noise_scale

    def generate_noise(self, input_latent):
        """Replay prior image draws and return FP32 noise with U1's BF16 scaling roundoff."""
        latent = input_latent["samples"]
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        for height, width in self.previous_sizes:
            torch.randn((1, 3, height, width), generator=generator,
                        device=self.device, dtype=torch.bfloat16)
        noise = torch.randn(latent.shape, generator=generator,
                            device=self.device, dtype=torch.bfloat16)
        scale = resolution_noise_scale(*latent.shape[-2:], noise_scale=self.noise_scale)
        # SenseNova-U1 rounds the scaled noise in BF16. The standard sampler
        # applies this scale later, so return an equivalent unscaled FP32 noise.
        return (noise * scale).float() / scale


def bf16_velocity_cfg(args):
    """Apply CFG to velocities in BF16, matching SenseNova-U1 sampling."""

    sigma = args["sigma"].reshape(-1, 1, 1, 1)
    conditional = (args["cond"] / sigma).to(torch.bfloat16)
    if args["cond_scale"] == 1 or args["uncond"] is None:
        guided = conditional
    else:
        unconditional = (args["uncond"] / sigma).to(torch.bfloat16)
        guided = unconditional + args["cond_scale"] * (conditional - unconditional)
    return guided.float() * sigma


def sample_bf16_euler(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Round Euler updates in BF16 while preserving ComfyUI's sampler contract."""

    x = x.to(torch.bfloat16).float()
    for index in range(len(sigmas) - 1):
        sigma = sigmas[index]
        denoised = model(x, sigma.expand(x.shape[0]), **(extra_args or {}))
        velocity = ((x - denoised) / sigma).to(torch.bfloat16)
        if callback is not None:
            callback({"x": x, "i": index, "sigma": sigma,
                      "sigma_hat": sigma, "denoised": denoised})
        x = (x.to(torch.bfloat16) + velocity * (sigmas[index + 1] - sigma)).float()
    return x
