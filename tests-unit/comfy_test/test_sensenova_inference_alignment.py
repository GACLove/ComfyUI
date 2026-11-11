from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from comfy.cli_args import args

args.cpu = True

import comfy.ops
from comfy.model_patcher import ModelPatcher
from comfy.ldm.sensenova import inference_alignment, model as sensenova_model
from comfy.ldm.sensenova.conditioning import prepare_prefix_inputs, preprocess_references
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder
from comfy.ldm.sensenova.model import SenseNovaU15
from comfy.ldm.sensenova.inference_alignment import _llm_rope, decode_text_token_exact_length, prefix_replay_lengths, predict_velocity, rms_norm, vision_forward
from comfy.ldm.sensenova.sampling import SenseNovaModelSampling
from comfy.text_encoders.llama import FixedKV
from comfy_api.internal import make_locked_method_func
from comfy_extras.nodes_sensenova import ModelSamplingSenseNova, SenseNovaU1InferenceAlignment


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA for device-specific RoPE arithmetic"))])
def test_alignment_rope_uses_u1_device_placement(device):
    device = torch.device(device)
    image = torch.randn(1, 3, 64, 96, device=device, dtype=torch.bfloat16)
    device = image.device
    vision = SimpleNamespace(
        patch_embedding=torch.nn.Conv2d(3, 1024, 16, stride=16, device=device, dtype=image.dtype),
        dense_embedding=torch.nn.Conv2d(1024, 8, 2, stride=2, device=device, dtype=image.dtype),
        gelu=torch.nn.GELU(),
    )
    positions = torch.arange(24, device=device)
    checked = set()
    expected_devices = dict.fromkeys(["aten::arange", "aten::pow", "aten::cos", "aten::sin"], torch.device("cpu"))

    class CheckRopeDevice(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func._schema.name in expected_devices:
                assert result.device == expected_devices[func._schema.name], (func, result.device)
                checked.add(func._schema.name)
            return result

    with CheckRopeDevice():
        encoded = vision_forward(vision, image)
    assert checked == expected_devices.keys()

    # LLM inv_freq is initialized with the CPU model, then transferred.
    expected_devices.update({"aten::cos": device, "aten::sin": device})
    checked.clear()
    with CheckRopeDevice():
        cos, sin = _llm_rope(positions, 64, 5000000.0, device, image.dtype)

    assert checked == expected_devices.keys()
    assert encoded.device == cos.device == sin.device == device
    assert encoded.dtype == cos.dtype == sin.dtype == image.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for device-specific scalar promotion")
def test_alignment_velocity_matches_cuda_scalar_division_after_first_step():
    device = torch.device("cuda")
    x = torch.linspace(-30, 30, 100000, device=device).bfloat16().reshape(1, 1, 1, -1)
    predicted = torch.zeros_like(x)
    t = torch.tensor(0.006756722927093506, device=device)
    expected = (x - predicted) / (1 - t).clamp_min(0.02)
    standard = SenseNovaU15.predict_velocity(x, predicted, t.reshape(1)).bfloat16()
    assert not torch.equal(standard, expected)
    assert torch.equal(predict_velocity(x, predicted, t.reshape(1)), expected)


def test_alignment_rms_norm_rounds_before_weight_multiplication():
    module = comfy.ops.manual_cast.RMSNorm(64, eps=1e-6, dtype=torch.bfloat16)
    module.weight = torch.nn.Parameter(torch.linspace(0.3, 2, 64).bfloat16())
    x = torch.linspace(-3, 5, 256).reshape(4, 64).bfloat16()
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + module.eps)
    expected = normalized.bfloat16() * module.weight
    assert torch.equal(rms_norm(module, x), expected)


def test_only_generated_images_use_u1_bf16_normalization():
    external = torch.linspace(0, 1, 48).reshape(1, 4, 4, 3)
    prediction = torch.linspace(-1.3, 1.2, 48).reshape(1, 4, 4, 3).bfloat16()
    generated = prediction.float() * 0.5 + 0.5
    actual = preprocess_references([external, generated], generated_images_start=1, generated_dtype=torch.bfloat16)
    for index, image in enumerate([external, prediction * 0.5 + 0.5]):
        mean = image.new_tensor([0.485, 0.456, 0.406])
        std = image.new_tensor([0.229, 0.224, 0.225])
        expected = ((image - mean) / std).movedim(-1, 1)
        assert torch.equal(actual[index], expected)
    assert actual[0].dtype == torch.float32
    assert actual[1].dtype == torch.bfloat16
    assert generated.dtype == torch.float32


def test_prefix_segments_keep_input_images_in_prefill_and_append_generated_blocks():
    initial = [1, 151670, 151669, 151669, 151671, 2]
    generated = [151670, 151669, 151669, 151671, 42, 43, 151670]
    assert prefix_replay_lengths(initial + generated, len(initial)) == [6, 1, 3, 1, 1, 1]
    assert prefix_replay_lengths(initial, len(initial)) == [6]
    assert prefix_replay_lengths(initial, None) == [6]


def test_alignment_decode_ignores_unused_cache_capacity():
    """Spare KV slots must not change aligned decoding or leak prefetch option overrides."""
    # Exercise the real token loop and attention with two backing capacities.
    # The decoder must see only the committed prefix plus the current token.
    class Layer:
        input_layernorm = None

        def text_modules(self):
            return []

        def forward_decode(self, hidden, rope, cache, options):
            assert cache.key.shape[1] == cache.index + 1
            assert not options["prefetch_dynamic_vbars"]
            cache.key[:, cache.index] = hidden
            cache.value[:, cache.index] = hidden
            return hidden + cache.value.sum(dim=1)

    embed = torch.nn.Embedding(4, 4)
    embed.weight.data.fill_(0.5)
    module = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(
        embed_tokens=embed, layers=[Layer()],
    )))
    options = {"prefetch_dynamic_vbars": True}
    results = []
    for capacity in (5, 128):
        # Nonzero spare slots would visibly corrupt the sum if exposed.
        key = torch.full((1, capacity, 1, 4), 1000.0)
        key[:, :2] = 1
        cache = FixedKV(key, key.clone(), 2, torch.empty(1, dtype=torch.int64), torch.tensor([2], dtype=torch.int32))
        hidden = torch.empty(1, 1, 4)
        for _ in range(3):
            decode_text_token_exact_length(module, torch.tensor([1]), [cache], hidden, None, options)
        assert cache.index == 5 and cache.seqlen.item() == 5
        assert cache.key.shape[1] == capacity
        results.append(hidden.clone())
    assert torch.equal(*results)
    assert options == {"prefetch_dynamic_vbars": True}


def test_inference_alignment_is_scoped_to_patcher_clone_and_restored():
    class Tiny(torch.nn.Module):
        prepare_mrope = staticmethod(SenseNovaU15.prepare_mrope)
        predict_velocity = staticmethod(SenseNovaU15.predict_velocity)
        embed_timesteps = SenseNovaU15.embed_timesteps
        _preprocess_prefix_state = SenseNovaU15._preprocess_prefix_state
        _decode_text_token = SenseNovaU15._decode_text_token

        def __init__(self):
            super().__init__()
            self.norm = comfy.ops.manual_cast.RMSNorm(4, eps=1e-6)

    model = torch.nn.Module()
    model.diffusion_model = Tiny()
    model.model_config = SimpleNamespace(sampling_settings={})
    model.model_sampling = SenseNovaModelSampling()
    model.reference_image_dtype = torch.float32
    model.process_timestep = lambda timestep, **kwargs: timestep * 1000
    original_norm = model.diffusion_model.norm.forward
    original_decode = model.diffusion_model._decode_text_token
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    ordinary = ModelSamplingSenseNova.execute(model=patcher, shift=3)[0]
    assert set(ordinary.object_patches) == {"model_sampling"}
    assert "sampler_calc_cond_batch_function" not in ordinary.model_options
    assert "sampler_cfg_function" not in ordinary.model_options
    output = SenseNovaU1InferenceAlignment.execute(patcher, 42, steps=50, shift=3)
    reference = output[0]
    assert patcher.object_patches == {}
    assert "sampler_calc_cond_batch_function" not in patcher.model_options
    assert model.reference_image_dtype == torch.float32
    assert output[3].shape == (51,)
    assert output[3][0] == 1 and output[3][-1] == 0
    reference.patch_model(load_weights=False)
    assert model.reference_image_dtype == torch.bfloat16
    assert model.diffusion_model.norm.forward != original_norm
    assert model.diffusion_model._decode_text_token != original_decode
    assert torch.equal(model.model_sampling.timestep(torch.tensor([0.75])), torch.tensor([0.25]))
    reference.unpatch_model(unpatch_weights=False)
    assert model.reference_image_dtype == torch.float32
    assert model.diffusion_model.norm.forward == original_norm
    assert model.diffusion_model._decode_text_token == original_decode
    ordinary.patch_model(load_weights=False)
    assert model.reference_image_dtype == torch.float32
    assert model.diffusion_model.norm.forward == original_norm
    ordinary.unpatch_model(unpatch_weights=False)


def test_ordinary_sampling_options_exposes_no_alignment_settings():
    assert {input.id for input in ModelSamplingSenseNova.define_schema().inputs} == {"model", "shift"}


@pytest.mark.parametrize("inputs,valid", [
    ({}, True), ({"shift": 3.0}, True), ({"shift": 0.5}, True),
    ({"shift": 0.0}, False), ({"shift": -1.0}, False),
    ({"shift": float("nan")}, False), ({"shift": float("inf")}, False),
])
def test_alignment_validation_accepts_optional_shift(inputs, valid):
    """V3 validation accepts an omitted shift while rejecting invalid explicit values."""
    node = SenseNovaU1InferenceAlignment
    node.VALIDATE_CLASS()
    validate = make_locked_method_func(node, "validate_inputs", node.PREPARE_CLASS_CLONE({}))
    result = validate(**inputs)
    assert result is True if valid else result == "shift must be finite and greater than zero."


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_alignment_timestep_embeddings_match_native_modules(dtype):
    module = SimpleNamespace(fm_modules={
        "timestep_embedder": TimestepEmbedder(16, frequency_embedding_size=32, max_period=1000,
                                             dtype=dtype, operations=torch.nn),
        "noise_scale_embedder": TimestepEmbedder(16, frequency_embedding_size=32,
                                                dtype=dtype, operations=torch.nn),
    })
    timesteps = torch.tensor([0.0, 0.25, 0.99])
    expected = module.fm_modules["timestep_embedder"](timesteps, dtype)[:, None].expand(-1, 6, -1)
    actual = inference_alignment._time_embedding(module.fm_modules["timestep_embedder"], timesteps, 6, dtype)
    torch.testing.assert_close(actual, expected)
    expected = SenseNovaU15.embed_timesteps(module, timesteps, 64, 96, dtype).expand(-1, 6, -1)
    actual = inference_alignment.embed_timesteps(module, timesteps, 64, 96, dtype)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == dtype


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("replay_start", [None, 3])
def test_alignment_prefix_replay_matches_native_fp32(monkeypatch, with_image, replay_start):
    """Replay must preserve hidden/KV states for text and image histories with optional chunking."""
    for name, value in {"HIDDEN_SIZE": 32, "INTERMEDIATE_SIZE": 64, "HEAD_DIM": 8,
                        "NUM_HEADS": 4, "NUM_KV_HEADS": 2, "NUM_LAYERS": 2}.items():
        monkeypatch.setattr(sensenova_model, name, value)
    monkeypatch.setattr(inference_alignment, "NUM_HEADS", 4)
    module = object.__new__(SenseNovaU15)
    torch.nn.Module.__init__(module)
    module.language_model = sensenova_model.LanguageModel(operations=torch.nn)
    module.vision_model = sensenova_model.VisionModel(operations=torch.nn)
    if with_image:
        ids = torch.tensor([[1, 2, 3, 42, 151670, 151669, 151669, 151671, 43]])
        images = [torch.linspace(0, 1, 32 * 64 * 3).reshape(1, 32, 64, 3)]
    else:
        ids, images = torch.tensor([[1, 2, 3, 42, 43]]), []
    inputs = prepare_prefix_inputs(ids, images, 32, torch.float32, expand_images=False,
                                   generated_start=replay_start)
    expected = module._preprocess_prefix_state(*inputs, compute_dtype=torch.float32)
    actual = inference_alignment.replay_prefix_state(
        module, *inputs, transformer_options={"sensenova_prefix_start": replay_start},
        compute_dtype=torch.float32,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    for actual_cache, expected_cache in zip(actual[1:3], expected[1:3]):
        assert len(actual_cache) == len(expected_cache) == 2
        for actual_layer, expected_layer in zip(actual_cache, expected_cache):
            torch.testing.assert_close(actual_layer, expected_layer, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[3], expected[3])


@pytest.mark.parametrize("masked", [False, True])
def test_alignment_attention_handles_containers_and_grouped_queries(masked):
    query = torch.linspace(-1, 1, 2 * 4 * 3 * 8).reshape(2, 4, 3, 8)
    key = torch.linspace(1, -1, 2 * 2 * 5 * 8).reshape(2, 2, 5, 8)
    value = torch.sin(key * 3)
    mask = torch.zeros(2, 1, 3, 5) if masked else None
    if masked:
        mask[..., -2:] = float("-inf")
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key.repeat_interleave(2, dim=1), value.repeat_interleave(2, dim=1), attn_mask=mask,
    ).transpose(1, 2).reshape(2, 3, 32)
    preferred = ComfyAttention()
    preferred.function = inference_alignment.attention
    actual = optimized_attention(
        AttentionTensorContainer(query), AttentionTensorContainer(key), AttentionTensorContainer(value),
        4, mask=mask, skip_reshape=True, enable_gqa=True, preferred_attention=preferred,
    )
    torch.testing.assert_close(actual, expected)


def test_alignment_attention_rejects_over_budget_before_score_allocation(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "get_free_memory", lambda *args, **kwargs: 1024)
    monkeypatch.setattr(inference_alignment, "attention_basic", lambda *args, **kwargs: pytest.fail("Score allocation exceeded the available budget"))
    query = torch.empty(1, 4, 13, 8, dtype=torch.bfloat16)
    key = torch.empty(1, 2, 17, 8, dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 13, 17, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="memory budget"):
        inference_alignment.attention(query, key, key, 4, mask=mask, skip_reshape=True, enable_gqa=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("precision", [None, torch.float32])
@pytest.mark.parametrize("skip_reshape", [False, True])
@pytest.mark.parametrize("batched_mask", [False, True])
def test_alignment_attention_within_budget_preserves_full_matrix_result(monkeypatch, dtype, precision, skip_reshape, batched_mask):
    monkeypatch.setattr(comfy.model_management, "get_free_memory", lambda *args, **kwargs: 1024**3)
    generator = torch.Generator().manual_seed(42)
    query = torch.randn(2, 4, 13, 8, generator=generator, dtype=dtype)
    key = torch.randn(2, 2, 17, 8, generator=generator, dtype=dtype)
    value = torch.randn(2, 2, 17, 8, generator=generator, dtype=dtype)
    if not skip_reshape:
        query, key, value = [t.transpose(1, 2).flatten(2) for t in (query, key, value)]
    mask = torch.zeros((2, 1, 13, 17) if batched_mask else (13, 17), dtype=dtype)
    mask[..., -2:] = float("-inf")
    options = dict(mask=mask, skip_reshape=skip_reshape, attn_precision=precision, enable_gqa=True)
    expected = inference_alignment.attention_basic(query, key, value, 4, **options)
    actual = inference_alignment.attention(query, key, value, 4, **options)
    assert torch.equal(actual, expected)


def test_alignment_unmasked_attention_uses_selected_backend_without_matrix_budget(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "get_free_memory", lambda *args, **kwargs: pytest.fail("Unmasked attention owns its workspace"))
    expected = torch.empty(1)
    monkeypatch.setattr(inference_alignment, "optimized_attention", lambda *args, **kwargs: expected)
    assert inference_alignment.attention(expected, expected, expected, 1) is expected


def test_alignment_directml_attention_does_not_use_placeholder_memory(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "is_directml_enabled", lambda: True)
    monkeypatch.setattr(comfy.model_management, "get_free_memory", lambda *args, **kwargs: 1024**3)
    expected = torch.empty(1)
    monkeypatch.setattr(inference_alignment, "attention_basic", lambda *args, **kwargs: expected)
    query = torch.empty(1, 32, 2048, 128, dtype=torch.bfloat16, device="meta")
    mask = torch.empty(2048, 2048, dtype=torch.bfloat16, device="meta")
    assert inference_alignment.attention(query, query, query, 32, mask=mask, skip_reshape=True) is expected


@pytest.mark.parametrize("clone_kind", ["shared", "replacement", "dynamic_delegate"])
def test_alignment_methods_follow_cloned_backbone(monkeypatch, clone_kind):
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy.model_patcher.comfy_aimdo.host_buffer, "HostBuffer", lambda *args: None)

    class Tiny(SenseNovaU15):
        def __init__(self, scale):
            torch.nn.Module.__init__(self)
            self.norm = comfy.ops.manual_cast.RMSNorm(4, eps=1e-6, dtype=torch.bfloat16)
            self.norm.weight = torch.nn.Parameter(torch.full((4,), scale, dtype=torch.bfloat16), requires_grad=False)

    def make_patcher(scale, dynamic=False):
        model = torch.nn.Module()
        model.diffusion_model = Tiny(scale)
        model.reference_image_dtype = torch.float32
        model.process_timestep = lambda timestep, **kwargs: timestep * 1000
        cls = comfy.model_patcher.ModelPatcherDynamic if dynamic else ModelPatcher
        return cls(model, torch.device("cpu"), torch.device("cpu"))

    source = make_patcher(1.0, dynamic=clone_kind == "dynamic_delegate")
    target = make_patcher(2.0)
    inference_alignment.apply_u1_inference_alignment(source)
    if clone_kind == "dynamic_delegate":
        source.non_dynamic_delegate_model = target.get_clone_model_override()
        cloned = source.get_non_dynamic_delegate()
    else:
        cloned = source.clone(model_override=target.get_clone_model_override() if clone_kind == "replacement" else None)
    backbone = cloned.model.diffusion_model
    assert cloned.object_patches["diffusion_model._preprocess_prefix_state"].__self__ is backbone
    assert cloned.object_patches["diffusion_model._decode_text_token"].__self__ is backbone
    assert cloned.object_patches["diffusion_model.embed_timesteps"].__self__ is backbone
    assert cloned.object_patches["diffusion_model.norm.forward"].__self__ is backbone.norm
    if clone_kind != "shared":
        assert source.object_patches["diffusion_model.norm.forward"].__self__ is source.model.diffusion_model.norm
    cloned.patch_model(load_weights=False)
    try:
        x = torch.ones(1, 4, dtype=torch.bfloat16)
        expected = torch.nn.functional.rms_norm(x, (4,), None, 1e-6) * backbone.norm.weight
        assert torch.equal(backbone.norm(x), expected)
    finally:
        cloned.unpatch_model(unpatch_weights=False)
