from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

args.cpu = True

from comfy.ldm.sensenova.sampling_alignment import U1AlignedNoise, bf16_velocity_cfg, sample_bf16_euler
from comfy.ldm.sensenova.inference_alignment import separate_guidance
from comfy.ldm.sensenova.sampling import SenseNovaModelSampling, resolution_noise_scale
from comfy.model_patcher import ModelPatcher
from comfy.samplers import sampling_function
from comfy_extras.nodes_sensenova import SenseNovaU1InferenceAlignment


@pytest.mark.parametrize("previous_sizes", [[], [(32, 64)], [(32, 64), (64, 96)]])
def test_aligned_noise_replays_variable_size_stream_without_global_rng_changes(previous_sizes):
    seed = 42
    shape = (1, 3, 96, 64)
    state = torch.random.get_rng_state().clone()
    noise = U1AlignedNoise(seed, torch.device("cpu"), previous_sizes)
    actual = noise.generate_noise({"samples": torch.zeros(shape)})
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(actual, noise.generate_noise({"samples": torch.zeros(shape)}))

    generator = torch.Generator().manual_seed(seed)
    for height, width in previous_sizes:
        torch.randn((1, 3, height, width), generator=generator, dtype=torch.bfloat16)
    scale = resolution_noise_scale(*shape[-2:])
    expected = torch.randn(shape, generator=generator, dtype=torch.bfloat16) * scale
    assert torch.equal((actual * scale).bfloat16(), expected)


@pytest.mark.parametrize("pending_size", [None, (96, 64)])
def test_aligned_noise_history_excludes_external_reference_images(pending_size):
    base = torch.nn.Module()
    base.diffusion_model = torch.nn.Module()
    base.model_config = SimpleNamespace(sampling_settings={})
    base.model_sampling = SenseNovaModelSampling()
    model = ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
    external = torch.zeros(1, 32, 64, 3)
    generated = torch.zeros(1, 64, 32, 3)
    conditioning = [[None, {
        "reference_latents": [external, generated],
        "sensenova_interleave_input_images": 1,
    }]]
    samples = {"samples": torch.zeros(1, 3, *pending_size)} if pending_size else None
    output = SenseNovaU1InferenceAlignment.execute(model, 42, conditioning, samples=samples)
    expected_sizes = [(64, 32)] + ([pending_size] if pending_size else [])
    assert output[1].previous_sizes == expected_sizes
    latent = {"samples": torch.zeros(1, 3, 32, 96)}
    expected = U1AlignedNoise(42, torch.device("cpu"), expected_sizes).generate_noise(latent)
    assert torch.equal(output[1].generate_noise(latent), expected)
    assert conditioning[0][1]["reference_latents"] == [external, generated]
    no_history = SenseNovaU1InferenceAlignment.execute(model, 42, [[None, {"reference_latents": [external]}]])
    assert no_history[1].previous_sizes == []


@pytest.mark.parametrize("cfg_scale", [1.0, 4.0])
def test_aligned_sampler_matches_bf16_flow_updates_through_denoised_guider(cfg_scale):
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.0])
    initial = torch.linspace(-3, 3, 48).reshape(1, 3, 4, 4)

    def predictions(x):
        x = x.bfloat16()
        return x * 0.125 + 0.25, x * 0.0625 - 0.125

    # SenseNova-U1 guides velocities and updates the image in BF16.
    expected = initial.bfloat16()
    for sigma, following in zip(sigmas[:-1], sigmas[1:]):
        conditional, unconditional = predictions(expected)
        vc = (conditional - expected) / sigma
        vu = (unconditional - expected) / sigma
        velocity = vc if cfg_scale == 1 else vu + cfg_scale * (vc - vu)
        expected = expected + (sigma - following) * velocity

    def guider(x, sigma):
        conditional, unconditional = predictions(x)
        sigma = sigma.reshape(-1, 1, 1, 1)
        # Native BaseModel exposes FP32 denoised predictions to CFG.
        vc = (x.bfloat16() - conditional).float() / sigma
        vu = (x.bfloat16() - unconditional).float() / sigma
        return x - bf16_velocity_cfg({
            "cond": x - (x - vc * sigma),
            "uncond": None if cfg_scale == 1 else x - (x - vu * sigma),
            "sigma": sigma.flatten(), "cond_scale": cfg_scale,
        })

    steps = []
    actual = sample_bf16_euler(guider, initial, sigmas, callback=steps.append)
    assert torch.equal(actual, expected.float())
    assert [step["i"] for step in steps] == [0, 1, 2]


@pytest.mark.parametrize("custom_cfg", [False, True])
def test_aligned_cfg_one_uses_native_guider_zero_unconditional_prediction(monkeypatch, custom_cfg):
    x = torch.tensor([1.0, -3.0, 0.25, 100.0]).reshape(1, 1, 1, 4)
    sigma = torch.tensor([0.75])
    velocity = torch.tensor([0.01, 0.3, -0.01, 0.125]).bfloat16()
    conditional = x - velocity.float() * sigma
    conditioning, unconditional = object(), object()
    calls = []

    def predict(model, conds, x, sigma, options):
        calls.append(conds)
        return [conditional]

    monkeypatch.setattr("comfy.samplers.calc_cond_batch", predict)
    options = {"sampler_calc_cond_batch_function": separate_guidance}
    if custom_cfg:
        options["sampler_cfg_function"] = bf16_velocity_cfg
    actual = sampling_function(None, x, sigma, unconditional, conditioning, 1.0, options)
    assert torch.equal(actual, conditional)
    assert calls == [[conditioning]]
