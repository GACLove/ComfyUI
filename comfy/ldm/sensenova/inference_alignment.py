"""Model execution patches for reproducing SenseNova-U1 inference.

Match its arithmetic, device placement and prefix replay order through ModelPatcher.
"""

import math
from types import MethodType

import torch
import torch.nn.functional as F

import comfy.model_prefetch
import comfy.model_management
import comfy.ops
import comfy.samplers
from comfy.ldm.modules.attention import AttentionTensorContainer, attention_basic, get_attn_precision, optimized_attention
from comfy.patcher_extension import CallbacksMP
from comfy.text_encoders.llama import FixedKV

from .conditioning import IMAGE_CONTEXT_ID, IMAGE_END_ID, IMAGE_START_ID
from .model import Attention, ConvDecoder, HEAD_DIM, NUM_HEADS, SenseNovaU15, VisionEmbeddings
from .sampling import SenseNovaModelSampling, resolution_noise_scale


class U1AlignedModelSampling(SenseNovaModelSampling):
    """Use SenseNova-U1's normalized timestep instead of ComfyUI's scaled timestep."""

    def timestep(self, sigma):
        return 1.0 - sigma


def process_timestep(timestep, **kwargs):
    return timestep


def rms_norm(module, x):
    """Normalize before applying learned weights to preserve U1's BF16 rounding."""
    comfy.ops.run_every_op()
    with module.cast_weight(x) as (weight, _):
        normalized = F.rms_norm(x, module.normalized_shape, None, module.eps)
        return normalized if weight is None else normalized * weight


def _vision_rope(value, positions):
    dim = value.shape[-1]
    # SenseNova-U1 precomputes these tables on CPU. GPU pow/cos and fused
    # rotations round differently, even with identical checkpoint tensors.
    frequencies = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    angles = torch.outer(positions.float(), frequencies)
    cosine, sine = angles.cos().to(value.device), angles.sin().to(value.device)
    first, second = value[..., 0::2].float(), value[..., 1::2].float()
    return torch.stack((first * cosine - second * sine, first * sine + second * cosine), dim=-1).flatten(-2)


def vision_forward(module, image):
    batch, channels, height, width = image.shape
    height, width = height // 16, width // 16
    patches = image.view(batch, channels, height, 16, width, 16).permute(0, 2, 4, 1, 3, 5).reshape(-1, channels, 16, 16)
    patches = module.gelu(module.patch_embedding(patches)).view(batch, height * width, 1024)
    positions = torch.arange(height * width)
    patches = torch.cat((
        _vision_rope(patches[..., :512], positions % width),
        _vision_rope(patches[..., 512:], positions // width),
    ), dim=-1).to(image.dtype)
    patches = patches.view(batch, height, width, 1024).permute(0, 3, 1, 2)
    return module.dense_embedding(patches).flatten(2).transpose(1, 2)


def _llm_rope(positions, dim, theta, device, dtype):
    """Build LLM rotary tables with CPU frequencies and device-side trigonometry."""
    # SenseNova-U1 constructs LLM frequencies before moving the model to its
    # inference device. Trigonometry below still runs on the active device.
    frequencies = (1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))).to(device)
    positions = positions.to(device=device, dtype=torch.float32)
    if positions.ndim == 1:
        positions = positions[None]
    angles = (frequencies[None, :, None].expand(positions.shape[0], -1, 1) @ positions[:, None, :]).transpose(1, 2)
    embedding = torch.cat((angles, angles), dim=-1)[:, None]
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


def prepare_mrope(indexes, device, dtype):
    return (
        _llm_rope(indexes[0], HEAD_DIM // 2, 5000000.0, device, dtype),
        _llm_rope(indexes[1], HEAD_DIM // 4, 10000.0, device, dtype),
        _llm_rope(indexes[2], HEAD_DIM // 4, 10000.0, device, dtype),
    )


def _time_embedding(module, timesteps, length, dtype):
    expanded = timesteps[:, None].expand(-1, length).reshape(-1)
    half = module.frequency_embedding_size // 2
    frequencies = torch.exp(-math.log(module.max_period) * torch.arange(half, dtype=torch.float32) / half).to(timesteps.device)
    angles = expanded[:, None].float() * frequencies[None]
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
    return module.mlp(embedding).view(len(timesteps), length, -1)


def embed_timesteps(module, timesteps, height, width, dtype):
    length = (height // 32) * (width // 32)
    embedding = _time_embedding(module.fm_modules["timestep_embedder"], timesteps, length, dtype)
    scale = resolution_noise_scale(height, width) / 16.0
    return embedding + _time_embedding(module.fm_modules["noise_scale_embedder"], torch.full_like(timesteps, scale), length, dtype)


def predict_velocity(x, predicted, timesteps):
    # A CUDA FP32 scalar divisor is converted to BF16 in SenseNova-U1's
    # BF16 tensor division. A batched FP32 divisor instead promotes the result.
    denominator = (1.0 - timesteps).clamp_min(0.02).to(x.dtype).view(-1, 1, 1, 1)
    return (x - predicted) / denominator


def head_forward(module, x):
    return ConvDecoder.forward(module, x.contiguous())


def attention(q, k, v, heads, mask=None, **kwargs):
    """Budget full-matrix masked attention where device free memory is measurable."""
    if mask is not None:
        query = q.peek() if isinstance(q, AttentionTensorContainer) else q
        key = k.peek() if isinstance(k, AttentionTensorContainer) else k
        precision = get_attn_precision(kwargs.get("attn_precision"), query.dtype)
        score_dtype = torch.float32 if precision == torch.float32 else query.dtype
        score_elements = query.shape[0] * heads * query.shape[-2] * key.shape[-2]
        head_dim = query.shape[-1] if kwargs.get("skip_reshape", False) else query.shape[-1] // heads
        qkv_elements = query.numel() + 2 * query.shape[0] * heads * key.shape[-2] * head_dim
        # Reserve score/softmax temporaries, mask expansion, casts, and expanded QKV.
        required = (4 * score_elements + qkv_elements) * comfy.model_management.dtype_size(score_dtype)
        # DirectML reports a fixed placeholder rather than measured device memory.
        budget = (math.inf if comfy.model_management.is_directml_enabled() and query.device.type not in ("cpu", "mps")
                  else comfy.model_management.get_free_memory(query.device) // 2)
        del query, key
        if required > budget:
            raise RuntimeError(
                f"SenseNova-U1 masked attention exceeds its memory budget: "
                f"{required / 1024**2:.0f} MiB required, {budget / 1024**2:.0f} MiB available. "
                "Reduce the prefix length or use standard ComfyUI sampling."
            )
    function = attention_basic if mask is not None else optimized_attention
    return function(q, k, v, heads, mask=mask, **kwargs)


def decode_text_token_exact_length(module, token, caches, hidden_states, rope, transformer_options=None):
    """Decode with the active KV length to match SenseNova-U1 attention reductions."""

    # Masked spare KV slots can still change BF16 attention reductions. Match
    # the repository's actual sequence length while retaining the shared buffers.
    views = [FixedKV(cache.key[:, :cache.index + 1], cache.value[:, :cache.index + 1],
                     cache.index, cache.position, cache.seqlen) for cache in caches]
    options = {} if transformer_options is None else transformer_options.copy()
    # Changing lengths cannot reuse a fixed-shape decode CUDA graph. The normal
    # cast/offload path remains active; diffusion keeps its prefetch settings.
    options["prefetch_dynamic_vbars"] = False
    SenseNovaU15._decode_text_token(module, token, views, hidden_states, rope, options)
    for cache, view in zip(caches, views):
        cache.index = view.index


def prefix_replay_lengths(ids, start):
    """Keep the initial prompt together, then replay generated text and image blocks."""

    if start is None:
        return [len(ids)]
    lengths = [start] if start else []
    cursor = start
    while cursor < len(ids):
        end = cursor + 1
        if ids[cursor] == IMAGE_CONTEXT_ID and cursor > 0 and ids[cursor - 1] == IMAGE_START_ID:
            while end < len(ids) and ids[end] == IMAGE_CONTEXT_ID:
                end += 1
            if end < len(ids) and ids[end] == IMAGE_END_ID:
                end += 1
        lengths.append(end - cursor)
        cursor = end
    return lengths


def replay_prefix_state(module, text_input_ids, reference_images=None, prefix_indexes=None, prefix_mask=None,
                        transformer_options=None, text_suffix=None, compute_dtype=None, interrupt=None):
    """Rebuild prefix hidden states and KV caches in SenseNova-U1 generation order."""

    prefix, indexes, mask, prefix_time = module._prepare_prefix(
        text_input_ids, reference_images, prefix_indexes, prefix_mask, text_suffix, compute_dtype
    )
    options = {} if transformer_options is None else transformer_options.copy()
    ids = text_input_ids if text_suffix is None else torch.cat((text_input_ids, text_suffix), dim=1)
    segments = prefix_replay_lengths(ids[0].tolist(), options.get("sensenova_prefix_start"))
    rope = module.prepare_mrope(indexes, prefix.device, prefix.dtype)
    keys, values = [], []
    prefetch = comfy.model_prefetch.make_prefetch_queue(
        [layer.text_modules() for layer in module.language_model.model.layers], prefix.device, options
    )
    with comfy.model_prefetch.malloc_graph_scope(prefix.device):
        for layer_index, layer in enumerate(module.language_model.model.layers):
            if interrupt is not None:
                interrupt()
            comfy.model_prefetch.prefetch_queue_pop(prefetch, prefix.device, layer, prefix.dtype, malloc_scope="prefix")
            options["block_index"] = layer_index
            offset, key, value, outputs = 0, None, None, []
            for length in segments:
                stop = offset + length
                hidden = prefix[:, offset:stop]
                segment_rope = tuple(tuple(freq[:, :, offset:stop] for freq in pair) for pair in rope)
                query, next_key, next_value = layer.self_attn._project(layer.input_layernorm(hidden), segment_rope, False)
                key = next_key if key is None else torch.cat((key, next_key), dim=2)
                value = next_value if value is None else torch.cat((value, next_value), dim=2)
                attended = optimized_attention(
                    AttentionTensorContainer(query), AttentionTensorContainer(key), AttentionTensorContainer(value),
                    NUM_HEADS, mask=mask[..., offset:stop, :stop], skip_reshape=True, enable_gqa=True,
                    preferred_attention=layer.self_attn.comfy_attention, transformer_options=options,
                )
                hidden = hidden + layer.self_attn.o_proj(attended)
                outputs.append(hidden + layer.mlp(layer.post_attention_layernorm(hidden)))
                offset = stop
            with comfy.model_prefetch.pause_malloc_graph():
                keys.append(key.clone())
                values.append(value.clone())
            prefix.copy_(torch.cat(outputs, dim=1))
            del outputs, key, value
        comfy.model_prefetch.prefetch_queue_pop(prefetch, prefix.device, None, malloc_scope="prefix")
    return prefix, keys, values, prefix_time


def separate_guidance(args):
    """Evaluate CFG branches separately, using zero predictions for omitted conditioning."""
    return [
        torch.zeros_like(args["input"]) if conditioning is None else comfy.samplers.calc_cond_batch(
            args["model"], [conditioning], args["input"], args["sigma"], args["model_options"]
        )[0]
        for conditioning in args["conds"]
    ]


def _rebind_u1_inference_alignment(source, cloned):
    """Bind aligned methods to a replacement backbone when a patcher clone owns one."""
    if source.model is not cloned.model:
        apply_u1_inference_alignment(cloned)


def apply_u1_inference_alignment(patcher):
    """Install SenseNova-U1 arithmetic and prefix/decode execution on a model patcher."""

    patcher.remove_callbacks_with_key(CallbacksMP.ON_CLONE, "sensenova_u1_alignment")
    patcher.add_callback_with_key(CallbacksMP.ON_CLONE, "sensenova_u1_alignment", _rebind_u1_inference_alignment)
    model = patcher.model.diffusion_model
    patcher.add_object_patch("reference_image_dtype", torch.bfloat16)
    patcher.add_object_patch("process_timestep", process_timestep)
    patcher.add_object_patch("diffusion_model.prepare_mrope", prepare_mrope)
    patcher.add_object_patch("diffusion_model.embed_timesteps", MethodType(embed_timesteps, model))
    patcher.add_object_patch("diffusion_model.predict_velocity", predict_velocity)
    patcher.add_object_patch("diffusion_model._preprocess_prefix_state", MethodType(replay_prefix_state, model))
    patcher.add_object_patch("diffusion_model._decode_text_token", MethodType(decode_text_token_exact_length, model))
    patcher.model_options["sampler_calc_cond_batch_function"] = separate_guidance
    for name, module in model.named_modules():
        path = "diffusion_model." + name
        if isinstance(module, torch.nn.RMSNorm):
            patcher.add_object_patch(path + ".forward", MethodType(rms_norm, module))
        elif isinstance(module, VisionEmbeddings):
            patcher.add_object_patch(path + ".forward", MethodType(vision_forward, module))
        elif isinstance(module, ConvDecoder):
            patcher.add_object_patch(path + ".forward", MethodType(head_forward, module))
        elif isinstance(module, Attention):
            patcher.add_object_patch(path + ".comfy_attention.function", attention)
