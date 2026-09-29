# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from nemo_automodel.components.distributed.parallelizer import get_model_layer_groups
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4 import fsdp as dsv4_fsdp
from nemo_automodel.components.models.deepseek_v4 import layers as dsv4_layers
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.fsdp import _is_deepseek_v4_module, _iter_dsv4_fp32_modules
from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4Attention,
    DeepseekV4RotaryEmbedding,
    build_causal_padding_mask,
)
from nemo_automodel.components.models.deepseek_v4.model import (
    DeepseekV4ForCausalLM,
    DeepseekV4VisionGate,
    apply_deepseek_v4_image_visibility,
)
from nemo_automodel.components.models.deepseek_v4.optimized_kernels import (
    build_dsv4_sparse_topk_indices,
    dsv4_sparse_attention,
    is_dsv4_kernel_available,
)
from nemo_automodel.components.models.deepseek_v4.processing import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    THINKING_END_TOKEN,
    USER_TOKEN,
    DeepseekV4VisionProcessor,
    build_image_block,
    preprocess_image,
)
from nemo_automodel.components.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter
from nemo_automodel.components.models.deepseek_v4.vision import (
    DeepseekV4VisionAligner,
    DeepseekV4VisionBlock,
    DeepseekV4VisionRMSNorm,
    DeepseekV4VisionTransformer,
)
from nemo_automodel.components.moe.config import MoEConfig

# Over the default 5s budget on purpose: TileLang compilation takes up to about 50s on a cold worker.
# Reduce cold compiler startup before lowering this further.
pytestmark = pytest.mark.timeout(120)


def _can_run_tilelang_sparse_attention() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (8, 9)
        and is_dsv4_kernel_available("sparse_attn")
    )


def _sparse_attention_torch_module_oracle(
    q: torch.Tensor,
    kv: torch.Tensor,
    sinks: torch.Tensor,
    topk_indices: torch.Tensor,
    sm_scale: float,
    *,
    backend: str,
) -> torch.Tensor:
    """Run the Torch sparse-attention oracle behind the module's TileLang dispatch.

    Args:
        q: Query tensor of shape [batch, sequence, heads, head_dim].
        kv: Key/value tensor of shape [batch, key_sequence, head_dim].
        sinks: Attention-sink tensor of shape [heads].
        topk_indices: Sparse key indices of shape [batch, sequence, selected_keys].
        sm_scale: Attention score scaling factor.
        backend: Module-selected backend, ignored by this CPU oracle.

    Returns:
        Attention output tensor of shape [batch, sequence, heads, head_dim].
    """
    del backend
    return dsv4_sparse_attention(q, kv, sinks, topk_indices, sm_scale, backend="sparse_torch")


def _vision_config(**overrides) -> DeepseekV4Config:
    defaults = dict(
        vocab_size=256,
        hidden_size=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        num_hash_layers=1,
        num_nextn_predict_layers=0,
        dtype="float32",
        vision_n_layers=2,
        vision_dim=32,
        vision_n_heads=4,
        vision_inter_dim=20,
        vision_patch_size=2,
        vision_downsample_ratio=3,
        vision_max_n_token=384,
        vision_min_pixels=147456,
        vision_max_wh_ratio=8,
    )
    defaults.update(overrides)
    return DeepseekV4Config(**defaults)


@pytest.mark.parametrize(
    ("height", "width", "expected_grid"),
    [
        (32, 32, (28, 28, 10, 10)),  # minimum-pixel upscale
        (384, 384, (28, 28, 10, 10)),  # square
        (377, 511, (27, 37, 9, 13)),  # non-divisible aligner height/width
        (511, 377, (37, 27, 13, 9)),  # portrait
        (100, 800, (10, 78, 4, 26)),  # exact 8:1 boundary
        (100, 801, (10, 78, 4, 26)),  # ratio clamp path
        (60, 1000, (10, 78, 4, 26)),  # strongly panoramic
        (1000, 60, (112, 7, 38, 3)),  # strongly vertical
        (768, 1536, (36, 72, 12, 24)),  # large source, visual-budget resize
    ],
)
def test_preprocess_image_size_matrix_matches_reference_grids(height, width, expected_grid):
    config = _vision_config(vision_patch_size=14)
    image = Image.new("RGB", (width, height), color=(17, 31, 63))

    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = preprocess_image(image, config)

    assert (n_vit_h, n_vit_w, n_llm_h, n_llm_w) == expected_grid
    assert patches.shape == (n_vit_h * n_vit_w, 3, 14, 14)
    assert patches.dtype == torch.bfloat16
    types, permutation = build_image_block(n_llm_h, n_llm_w, start_pos=5)
    assert types.numel() <= config.vision_max_n_token
    assert permutation.numel() == n_llm_h * n_llm_w


@pytest.mark.parametrize("start_pos", [0, 1, 2, 3, 7, 19])
@pytest.mark.parametrize("grid", [(1, 1), (2, 3), (3, 4), (4, 5), (7, 2)])
def test_image_block_has_valid_n_layout(start_pos, grid):
    n_llm_h, n_llm_w = grid
    types, permutation = build_image_block(n_llm_h, n_llm_w, start_pos)

    assert int((types == IMAGE_START).sum()) == 1
    assert int((types == IMAGE_END).sum()) == 1
    assert int((types == IMAGE).sum()) == n_llm_h * n_llm_w
    assert int((types == IMAGE_NEW_LINE).sum()) == n_llm_h
    assert sorted(permutation.tolist()) == list(range(n_llm_h * n_llm_w))
    assert (start_pos + int((types == IMAGE_PAD).cumprod(0).sum())) % 4 == 3


@pytest.mark.parametrize(("n_h", "n_w"), [(3, 3), (4, 5), (5, 7), (6, 9), (8, 4)])
def test_vision_and_aligner_support_divisible_and_ragged_grids(n_h, n_w):
    torch.manual_seed(17)
    config = _vision_config()
    vision = DeepseekV4VisionTransformer(config)
    aligner = DeepseekV4VisionAligner(config)
    patches = torch.randn(n_h * n_w, 3, 2, 2, requires_grad=True)

    encoded = vision(patches, n_h, n_w)
    aligned = aligner(encoded, n_h, n_w)
    aligned.square().mean().backward()

    assert encoded.shape == (n_h * n_w, config.vision_dim)
    assert aligned.shape == (
        math.ceil(n_h / config.vision_downsample_ratio) * math.ceil(n_w / config.vision_downsample_ratio),
        config.hidden_size,
    )
    assert patches.grad is not None
    assert torch.isfinite(patches.grad).all()


def test_image_visibility_is_bidirectional_only_inside_matching_span():
    types = torch.tensor(
        [[-1, IMAGE_PAD, IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END, -1, IMAGE_START, IMAGE, IMAGE_END, -1]]
    )
    sequence_length = types.shape[1]
    causal = torch.full((1, 1, sequence_length, sequence_length), -torch.inf)
    causal = torch.triu(causal, diagonal=1)

    visible = apply_deepseek_v4_image_visibility(causal, types)

    assert torch.equal(visible[0, 0, 2:6, 2:6], torch.zeros(4, 4))
    assert torch.equal(visible[0, 0, 7:10, 7:10], torch.zeros(3, 3))
    assert torch.isneginf(visible[0, 0, 2, 7])
    assert torch.isneginf(visible[0, 0, 6, 7])
    assert visible[0, 0, 10, 9] == 0


@pytest.mark.parametrize(
    "types",
    [
        [-1, IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END, -1, -1],
        [-1, IMAGE_START, IMAGE, IMAGE_END, -1, IMAGE_START, IMAGE, IMAGE, IMAGE_END, -1],
        [-1] * 130 + [IMAGE_START] + [IMAGE] * 180 + [IMAGE_END] + [-1] * 20,
    ],
)
def test_sparse_visual_window_indices_match_released_reference(types):
    token_types = torch.tensor([types], dtype=torch.long)
    seq_len = token_types.shape[1]
    window_size = 128
    max_image_tokens = 384

    actual = build_dsv4_sparse_topk_indices(
        batch_size=1,
        seq_len=seq_len,
        key_len=seq_len,
        window_size=window_size,
        device=token_types.device,
        vision_token_types=token_types,
        max_image_tokens=max_image_tokens,
    )

    # Direct transcription of inference/model.py's get_image_visible and
    # get_window_topk_idxs_visible, kept local so this remains a stable oracle.
    idx = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    is_start = token_types == IMAGE_START
    is_end = token_types == IMAGE_END
    valid = (is_start.cumsum(1) > is_end.cumsum(1)) | is_end
    starts = torch.where(is_start, idx, 0).cummax(1).values
    left = ((idx - starts) * valid).clamp(max=max_image_tokens - 1)
    ends = torch.where(is_end, idx, seq_len).flip(1).cummin(1).values.flip(1)
    right = ((ends - idx) * valid).clamp(max=max_image_tokens)
    width = min(seq_len, window_size + max_image_tokens)
    left_add = (left - (window_size - 1)).clamp(min=0)
    reference_starts = (idx - (window_size - 1) - left_add).clamp(min=0)
    expected = reference_starts.unsqueeze(-1) + torch.arange(width)
    expected = torch.where(expected > (idx + right).unsqueeze(-1), -1, expected)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("device", "dtype", "uses_tilelang"),
    [
        pytest.param("cpu", torch.float32, False, id="torch_sparse_oracle"),
        pytest.param(
            "cuda",
            torch.bfloat16,
            True,
            id="tilelang",
            marks=pytest.mark.skipif(
                not _can_run_tilelang_sparse_attention(),
                reason="TileLang sparse attention requires CUDA compute capability 8.9 or newer",
            ),
        ),
    ],
)
def test_visual_sparse_attention_module_matches_dense_forward_backward(monkeypatch, device, dtype, uses_tilelang):
    torch.manual_seed(123)
    if uses_tilelang:
        hidden_size, num_heads, head_dim, seq_len, window_size = 128, 16, 512, 192, 128
        image_start, image_end, max_image_tokens = 16, 176, 384
    else:
        hidden_size, num_heads, head_dim, seq_len, window_size = 32, 2, 16, 48, 16
        image_start, image_end, max_image_tokens = 10, 36, 32

    config = _vision_config(
        hidden_size=hidden_size,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        head_dim=head_dim,
        qk_rope_head_dim=min(64, head_dim // 2),
        q_lora_rank=hidden_size,
        o_lora_rank=hidden_size,
        o_groups=1,
        n_routed_experts=2,
        n_shared_experts=0,
        num_experts_per_tok=1,
        max_position_embeddings=seq_len,
        compress_ratios=[0],
        sliding_window=window_size,
        attention_dropout=0.0,
        num_hash_layers=0,
        hc_mult=1,
        vision_max_n_token=max_image_tokens,
        dtype=dtype,
    )
    backend_kwargs = {
        "linear": "torch",
        "rms_norm": "torch",
        "rope_fusion": False,
        "enable_hf_state_dict_adapter": False,
        "dispatcher": "torch",
        "experts": "torch",
    }
    reference = DeepseekV4Attention(config, layer_idx=0, backend=BackendConfig(attn="sdpa", **backend_kwargs))
    actual = DeepseekV4Attention(config, layer_idx=0, backend=BackendConfig(attn="tilelang", **backend_kwargs))
    reference.init_weights(torch.device("cpu"))
    actual.load_state_dict(reference.state_dict())
    reference.to(device=device, dtype=dtype).train()
    actual.to(device=device, dtype=dtype).train()
    reference.sinks_param.to(dtype=torch.float32)
    actual.sinks_param.to(dtype=torch.float32)

    hidden_states = torch.randn(1, seq_len, hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    rotary = DeepseekV4RotaryEmbedding(
        rope_theta=float(config.rope_theta),
        head_dim=head_dim,
        partial_rotary_factor=float(config.qk_rope_head_dim) / float(head_dim),
        device=torch.device(device),
    )
    position_embeddings = rotary(hidden_states, position_ids)
    attention_mask = build_causal_padding_mask(
        None,
        seq_len=seq_len,
        dtype=dtype,
        device=torch.device(device),
        batch_size=1,
        sliding_window=window_size,
    )
    vision_token_types = torch.full((1, seq_len), -1, dtype=torch.long, device=device)
    vision_token_types[:, image_start] = IMAGE_START
    vision_token_types[:, image_start + 1 : image_end] = IMAGE
    vision_token_types[:, image_end] = IMAGE_END
    attention_mask = apply_deepseek_v4_image_visibility(attention_mask, vision_token_types)
    assert image_end - image_start >= window_size
    assert attention_mask[0, 0, image_start, image_end] == 0

    if not uses_tilelang:
        monkeypatch.setattr(dsv4_layers, "dsv4_sparse_attention", _sparse_attention_torch_module_oracle)

    grad_output = torch.randn(1, seq_len, hidden_size, device=device, dtype=dtype)
    reference_input = hidden_states.detach().clone().requires_grad_(True)
    reference_output, _ = reference(
        reference_input,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        vision_token_types=vision_token_types,
    )
    reference_output.backward(grad_output)

    actual_input = hidden_states.detach().clone().requires_grad_(True)
    actual_output, _ = actual(
        actual_input,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        vision_token_types=vision_token_types,
    )
    actual_output.backward(grad_output)

    forward_rtol, forward_atol = (2e-2, 2e-2) if uses_tilelang else (1e-5, 1e-6)
    grad_rtol, grad_atol = (5e-2, 5e-2) if uses_tilelang else (1e-5, 1e-6)
    torch.testing.assert_close(actual_output, reference_output, rtol=forward_rtol, atol=forward_atol)
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=grad_rtol, atol=grad_atol)
    for (actual_name, actual_param), (reference_name, reference_param) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert actual_name == reference_name
        assert actual_param.grad is not None, actual_name
        assert reference_param.grad is not None, reference_name
        if uses_tilelang:
            gradient_cosine = torch.nn.functional.cosine_similarity(
                actual_param.grad.float().flatten(), reference_param.grad.float().flatten(), dim=0
            )
            assert gradient_cosine > 0.999, f"{actual_name} gradient cosine too low: {gradient_cosine:.6f}"
            torch.testing.assert_close(
                actual_param.grad,
                reference_param.grad,
                rtol=grad_rtol,
                atol=0.15,
                msg=lambda msg: f"{actual_name} gradient mismatch:\n{msg}",
            )
        else:
            torch.testing.assert_close(
                actual_param.grad,
                reference_param.grad,
                rtol=grad_rtol,
                atol=grad_atol,
                msg=lambda msg: f"{actual_name} gradient mismatch:\n{msg}",
            )


@pytest.mark.parametrize("provide_metadata", [False, True])
def test_text_only_vision_forward_keeps_sparse_attention_fast_path(provide_metadata):
    config = _vision_config(
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=2,
        n_shared_experts=0,
        max_position_embeddings=32,
        hc_mult=2,
        compress_ratios=[0],
        sliding_window=4,
    )
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
        dispatcher="torch",
        experts="torch_mm",
    )
    model = DeepseekV4ForCausalLM(config, backend=backend).eval()
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    input_ids = torch.arange(8).unsqueeze(0)
    kwargs = {"attention_mask": torch.ones_like(input_ids)}
    if provide_metadata:
        kwargs["vision_token_types"] = torch.full_like(input_ids, -1)

    layer = model.model.layers["0"]
    with patch.object(layer, "forward", wraps=layer.forward) as layer_forward, torch.inference_mode():
        output = model(input_ids, **kwargs)

    assert output.logits.shape[:2] == input_ids.shape
    assert layer_forward.call_args.kwargs["vision_token_types"] is None


def _moe_config() -> MoEConfig:
    return MoEConfig(
        dim=16,
        inter_dim=8,
        moe_inter_dim=8,
        n_routed_experts=8,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=1e-3,
        score_func="sqrtsoftplus",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        dtype=torch.float32,
    )


def test_vision_gate_uses_hash_for_text_and_visual_bias_for_image_tokens():
    gate = DeepseekV4VisionGate(
        _vision_config(),
        _moe_config(),
        gate_precision=None,
        hash_routing=True,
    )
    with torch.no_grad():
        gate.weight.zero_()
        gate.tid2eid[3] = torch.tensor([1, 2])
        gate.bias_vl.copy_(torch.arange(8, dtype=torch.float32))
    input_ids = torch.tensor([[3, 256 + IMAGE]])
    types = torch.tensor([[-1, IMAGE]])
    gate.set_routing_context(input_ids, types)

    _, indices, _ = gate(torch.zeros(2, 16), torch.ones(2, dtype=torch.bool), None)

    assert indices[0].tolist() == [1, 2]
    assert indices[1].tolist() == [7, 6]


def test_vision_gate_switches_between_text_and_visual_score_bias():
    gate = DeepseekV4VisionGate(
        _vision_config(),
        _moe_config(),
        gate_precision=None,
        hash_routing=False,
    )
    with torch.no_grad():
        gate.weight.zero_()
        gate.e_score_correction_bias.copy_(torch.tensor([0, 0, 0, 0, 4, 3, 0, 0], dtype=torch.float32))
        gate.bias_vl.copy_(torch.tensor([0, 0, 0, 0, 0, 0, 3, 4], dtype=torch.float32))
    gate.set_routing_context(torch.tensor([[3, 256 + IMAGE]]), torch.tensor([[-1, IMAGE]]))

    _, indices, _ = gate(torch.zeros(2, 16), torch.ones(2, dtype=torch.bool), None)

    assert indices[0].tolist() == [4, 5]
    assert indices[1].tolist() == [7, 6]


def test_processor_renders_released_vision_prompt_format():
    processor = object.__new__(DeepseekV4VisionProcessor)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (2, 2))},
                {"type": "text", "text": "What is shown?"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "A chest radiograph."}]},
    ]

    prompt, images = processor._render_conversation(conversation)

    assert prompt == (
        BOS_TOKEN
        + USER_TOKEN
        + IMAGE_PLACEHOLDER
        + "\n\nWhat is shown?"
        + ASSISTANT_TOKEN
        + THINKING_END_TOKEN
        + "A chest radiograph."
        + EOS_TOKEN
    )
    assert len(images) == 1


def test_state_dict_adapter_maps_native_vision_keys_bidirectionally():
    config = _vision_config()
    adapter = DeepSeekV4StateDictAdapter(config, _moe_config(), BackendConfig(), dtype=torch.float32)
    native = {
        "vision.patch_embed.proj.weight": torch.randn(32, 12),
        "aligner.w1.bias": torch.randn(16),
        "image_start": torch.randn(16),
        "layers.0.ffn.gate.bias_vl": torch.randn(8),
        "layers.0.ffn.gate.bias": torch.randn(8),
    }

    internal = adapter.from_hf(dict(native))

    assert set(internal) == {
        "model.vision.patch_embed.proj.weight",
        "model.aligner.w1.bias",
        "model.image_start",
        "model.layers.0.mlp.gate.bias_vl",
        "model.layers.0.mlp.gate.e_score_correction_bias",
    }
    restored = adapter.to_hf(internal)
    assert set(restored) == set(native)


def test_parallelizer_discovers_language_and_vision_blocks():
    config = _vision_config(
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_shared_experts=1,
        num_nextn_predict_layers=1,
        max_position_embeddings=128,
        hc_mult=4,
        compress_ratios=[0],
        sliding_window=16,
    )
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
        dispatcher="torch",
        experts="torch_mm",
    )
    model = DeepseekV4ForCausalLM(config, backend=backend)

    groups = get_model_layer_groups(model)

    assert len(groups["language"]) == 1
    assert len(groups["vision"]) == 2
    assert isinstance(model.mtp.layers[0].mlp.gate, DeepseekV4VisionGate)
    assert "mtp.layers.0.mlp.gate.bias_vl" in model.state_dict()


@pytest.mark.parametrize("image_microbatches", [(True,), (False, True, False, True)])
def test_causal_lm_consumes_staged_pp_media_chunk(image_microbatches: tuple[bool, ...]) -> None:
    config = _vision_config(
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_shared_experts=1,
        max_position_embeddings=128,
        hc_mult=4,
        compress_ratios=[0],
        sliding_window=16,
    )
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
        dispatcher="torch",
        experts="torch_mm",
    )
    torch.manual_seed(42)
    model = DeepseekV4ForCausalLM(config, backend=backend).eval()
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    types, _ = build_image_block(1, 1, start_pos=0)
    vision_token_types = torch.cat([types, torch.tensor([-1])]).unsqueeze(0)
    input_ids = torch.cat([config.vocab_size + types, torch.tensor([3])]).unsqueeze(0)
    pixel_chunks = [
        torch.randn(9 if has_image else 0, 3, config.vision_patch_size, config.vision_patch_size)
        for has_image in image_microbatches
    ]
    grid_chunks = [
        torch.tensor([[3, 3]]) if has_image else torch.empty((0, 2), dtype=torch.long)
        for has_image in image_microbatches
    ]
    input_chunks = [input_ids if has_image else torch.full_like(input_ids, 3) for has_image in image_microbatches]
    type_chunks = [
        vision_token_types if has_image else torch.full_like(vision_token_types, -1) for has_image in image_microbatches
    ]

    with torch.inference_mode():
        expected = [
            model(
                ids,
                attention_mask=torch.ones_like(ids),
                vision_token_types=token_types,
                pixel_values=pixels if has_image else None,
                image_grid_hws=grid if has_image else None,
            ).logits
            for has_image, ids, token_types, pixels, grid in zip(
                image_microbatches, input_chunks, type_chunks, pixel_chunks, grid_chunks
            )
        ]

    model._vlm_pixel_values_chunks = pixel_chunks
    model._vlm_image_grid_hws_chunks = grid_chunks
    model._vlm_chunk_idx = 0

    with torch.inference_mode():
        for chunk_idx, (ids, token_types, expected_logits) in enumerate(zip(input_chunks, type_chunks, expected)):
            output = model(
                ids,
                attention_mask=torch.ones_like(ids),
                vision_token_types=token_types,
            )
            torch.testing.assert_close(output.logits, expected_logits, rtol=0, atol=0)
            assert model._vlm_chunk_idx == chunk_idx + 1


def test_vision_blocks_expose_fp32_norm_islands_to_dsv4_fsdp():
    block = DeepseekV4VisionBlock(_vision_config())

    assert _is_deepseek_v4_module(block)
    fp32_modules = list(_iter_dsv4_fp32_modules(block))
    assert fp32_modules == [block.norm1, block.norm2]


@pytest.mark.parametrize("whole_tower", [False, True])
def test_vision_norm_fsdp_policy_preserves_bf16_activation_dtype(
    monkeypatch: pytest.MonkeyPatch, whole_tower: bool
) -> None:
    config = _vision_config(torch_dtype="bfloat16")
    block = DeepseekV4VisionTransformer(config) if whole_tower else DeepseekV4VisionBlock(config)
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))
        return module

    monkeypatch.setattr(dsv4_fsdp, "fully_shard", fake_fully_shard)
    input_policy = torch.distributed.fsdp.MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )

    dsv4_fsdp.fully_shard_deepseek_v4(
        block,
        mesh=object(),
        mp_policy=input_policy,
        offload_policy=object(),
    )

    norms = [module for module in block.modules() if isinstance(module, DeepseekV4VisionRMSNorm)]
    norm_policies = [kwargs["mp_policy"] for module, kwargs in calls if module in norms]
    assert len(norm_policies) == len(norms)
    if whole_tower:
        assert any(module is block.norm for module, _ in calls)
    assert all(policy.param_dtype == torch.float32 for policy in norm_policies)
    assert all(policy.reduce_dtype == torch.float32 for policy in norm_policies)
    assert all(policy.output_dtype is None for policy in norm_policies)
    assert all(policy.cast_forward_inputs is False for policy in norm_policies)
