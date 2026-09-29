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

"""Vision arithmetic, processing, image placement and modality routing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoProcessor, PreTrainedTokenizerFast

from nemo_automodel._transformers.registry import resolve_custom_config_cls
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.models.deepseek_v41.processing import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    TEXT,
    DeepseekV41Processor,
    image_inputs_from_batch,
)
from nemo_automodel.components.models.deepseek_v41.vision import DeepseekV41VisionAligner, DeepseekV41VisionTransformer


def _tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "one": 1, "two": 2, "tail": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="<｜begin▁of▁sentence｜>",
        eos_token="<｜end▁of▁sentence｜>",
        pad_token="<｜end▁of▁sentence｜>",
        additional_special_tokens=[IMAGE_PLACEHOLDER, "<｜User｜>", "<｜Assistant｜>", "<｜System｜>", "</think>"],
    )


def _config() -> DeepseekV41Config:
    return DeepseekV41Config(
        text_config=DeepseekV41TextConfig(hidden_size=8, engram_layer_ids=[], dtype="float32"),
        vision_config=DeepseekV41VisionConfig(
            num_hidden_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            intermediate_size=12,
            patch_size=2,
            downsample_ratio=2,
            min_pixels=4,
            max_image_tokens=32,
        ),
        image_token_id=_tokenizer().convert_tokens_to_ids(IMAGE_PLACEHOLDER),
        dtype="float32",
    )


def _reference_vision(
    patches: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    config: DeepseekV41Config,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Evaluate released vision equations independently of the reused V4 modules.

    Args:
        patches: Tensor of shape [height * width, 3, patch_size, patch_size].
        parameters: Vision/aligner tensors keyed with vision./aligner. prefixes.
            Linear weights use [out_features, in_features], biases/norm weights
            [features], and each block has ordinary unfused QKV ordering.
        config: Nested model configuration.
        height: Number of patch rows.
        width: Number of patch columns.

    Returns:
        Aligned features of shape [ceil(height / ratio) * ceil(width / ratio), hidden].
    """
    vision = config.vision_config

    def linear(hidden: torch.Tensor, key: str) -> torch.Tensor:
        """Apply the reference linear parameters.

        Args:
            hidden: Tensor of shape [..., in_features], arbitrary leading axes.
            key: Linear parameter prefix within the reference mapping.

        Returns:
            Tensor of shape [..., out_features] retaining the input leading axes.
        """
        return F.linear(hidden, parameters[key + ".weight"], parameters.get(key + ".bias"))

    def normalize(hidden: torch.Tensor, key: str) -> torch.Tensor:
        """Apply fp32 RMS normalization to patch states.

        Args:
            hidden: Tensor of shape [patches, vision_hidden].
            key: Reference norm parameter prefix.

        Returns:
            Tensor of shape [patches, vision_hidden].
        """
        normalized = hidden.float() * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-6)
        return (normalized * parameters[key + ".weight"]).to(hidden.dtype)

    hidden = linear(patches.flatten(1), "vision.patch_embed.proj")
    head_dim = vision.hidden_size // vision.num_attention_heads
    rotary_dim = head_dim // 2
    frequency = 1 / (vision.rope_theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    coordinates = torch.cartesian_prod(torch.arange(height), torch.arange(width))
    phases = (coordinates[:, :, None] * frequency).flatten(1).unsqueeze(1)
    cosine, sine = phases.cos(), phases.sin()
    for index in range(vision.num_hidden_layers):
        prefix = f"vision.blocks.{index}"
        qkv = linear(normalize(hidden, prefix + ".norm1"), prefix + ".attn.wqkv")
        query, key, value = [
            tensor.view(height * width, vision.num_attention_heads, head_dim) for tensor in qkv.chunk(3, -1)
        ]
        q_left, q_right = query.chunk(2, -1)
        k_left, k_right = key.chunk(2, -1)
        query = torch.cat((q_left * cosine - q_right * sine, q_right * cosine + q_left * sine), -1)
        key = torch.cat((k_left * cosine - k_right * sine, k_right * cosine + k_left * sine), -1)
        attended = F.scaled_dot_product_attention(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1))
        hidden = hidden + linear(attended.transpose(0, 1).reshape(height * width, -1), prefix + ".attn.wo")
        gate, up = linear(normalize(hidden, prefix + ".norm2"), prefix + ".mlp.w1").chunk(2, -1)
        hidden = hidden + linear(F.silu(gate) * up, prefix + ".mlp.w2")
    hidden = normalize(hidden, "vision.norm")
    ratio = vision.downsample_ratio
    grid = hidden.view(height, width, vision.hidden_size).permute(2, 0, 1)
    grid = F.pad(grid, (0, -width % ratio, 0, -height % ratio))
    # Explicit cells establish the channel/row/column order of the spatial merger.
    cells = []
    for row in range(0, grid.shape[1], ratio):
        for column in range(0, grid.shape[2], ratio):
            cells.append(grid[:, row : row + ratio, column : column + ratio].reshape(-1))
    return linear(F.gelu(linear(torch.stack(cells), "aligner.w1")), "aligner.w2")


@pytest.mark.parametrize("grid", [(2, 4), (3, 5)])
def test_vision_forward_and_all_parameter_gradients_match_reference(grid: tuple[int, int]) -> None:
    torch.manual_seed(11)
    config = _config()
    vision, aligner = DeepseekV41VisionTransformer(config), DeepseekV41VisionAligner(config)
    height, width = grid
    patches = torch.randn(height * width, 3, 2, 2, requires_grad=True)
    reference_patches = patches.detach().clone().requires_grad_()
    model_parameters = {
        **{f"vision.{name}": parameter for name, parameter in vision.named_parameters()},
        **{f"aligner.{name}": parameter for name, parameter in aligner.named_parameters()},
    }
    parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in model_parameters.items()}
    actual = aligner(vision(patches, height, width), height, width)
    expected = _reference_vision(reference_patches, parameters, config, height=height, width=width)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(patches.grad, reference_patches.grad, atol=1e-6, rtol=1e-5)
    for name, parameter in model_parameters.items():
        torch.testing.assert_close(parameter.grad, parameters[name].grad, atol=2e-6, rtol=2e-5)


def test_processor_image_ids_types_and_ordered_patch_placement() -> None:
    config = _config()
    processor = DeepseekV41Processor(_tokenizer(), config)
    red, blue = Image.new("RGB", (4, 6), "red"), Image.new("RGB", (7, 5), "blue")
    batch = processor(
        text=[f"one {IMAGE_PLACEHOLDER} tail", f"two {IMAGE_PLACEHOLDER} {IMAGE_PLACEHOLDER}"],
        images=[[red], [blue, red]],
        padding=True,
        return_tensors="pt",
    )
    types = batch["vision_token_types"]
    assert torch.equal(
        batch["input_ids"][types >= 0], torch.full_like(batch["input_ids"][types >= 0], config.image_token_id)
    )
    assert set(types.unique().tolist()) == {TEXT, IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END}
    records = image_inputs_from_batch(batch["pixel_values"], batch["image_grid_hws"], types, downsample_ratio=2)
    assert [record.batch_index for record in records] == [0, 1, 1]
    assert [(record.n_vit_h, record.n_vit_w) for record in records] == [(3, 2), (3, 4), (3, 2)]
    assert records[0].types.tolist() == [IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE, IMAGE_NEW_LINE, IMAGE_END]
    assert records[1].types.tolist() == [
        IMAGE_START,
        IMAGE,
        IMAGE,
        IMAGE_NEW_LINE,
        IMAGE,
        IMAGE,
        IMAGE_NEW_LINE,
        IMAGE_END,
    ]
    # Solid-color first image remains exact after pixel normalization and patchification.
    torch.testing.assert_close(records[0].patches[:, 0], torch.ones_like(records[0].patches[:, 0]))
    torch.testing.assert_close(records[0].patches[:, 1:], -torch.ones_like(records[0].patches[:, 1:]))


def test_processor_truncation_removes_whole_images_and_rejects_partial_spans() -> None:
    processor = DeepseekV41Processor(_tokenizer(), _config())
    image = Image.new("RGB", (4, 6), "red")
    text = f"one {IMAGE_PLACEHOLDER} two {IMAGE_PLACEHOLDER} tail"
    with pytest.raises(ValueError, match="truncates.*image span"):
        processor(text, [image, image], truncation=True, max_length=3)
    batch = processor(text, [image, image], truncation=True, max_length=8, return_tensors="pt")
    assert batch["image_grid_hws"].shape == (1, 2)
    records = image_inputs_from_batch(
        batch["pixel_values"], batch["image_grid_hws"], batch["vision_token_types"], downsample_ratio=2
    )
    assert len(records) == 1
    assert batch["input_ids"].shape == (1, 8)


def test_standard_chat_matches_official_v41_system_and_user_transitions() -> None:
    processor = DeepseekV41Processor(_tokenizer(), _config())
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "answer"},
        {"role": "system", "content": "new rules"},
    ]
    expected = (
        "<｜begin▁of▁sentence｜><｜System｜>rules<｜User｜>one\n\ntwo"
        "<｜Assistant｜></think>answer<｜end▁of▁sentence｜>"
        "<｜System｜>new rules<｜Assistant｜></think>"
    )
    assert processor.apply_chat_template(messages) == expected
    assert processor.apply_chat_template(messages, add_generation_prompt=False) == expected.removesuffix(
        "<｜Assistant｜></think>"
    )
    with pytest.raises(ValueError, match="metadata"):
        processor.apply_chat_template([{"role": "assistant", "content": "", "tool_calls": []}])


def test_chat_image_blocks_expand_in_content_order() -> None:
    processor = DeepseekV41Processor(_tokenizer(), _config())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "one"},
                {"type": "image", "image": Image.new("RGB", (4, 6), "red")},
                {"type": "text", "text": "two"},
            ],
        }
    ]
    expected = f"<｜begin▁of▁sentence｜><｜User｜>one\n\n{IMAGE_PLACEHOLDER}\n\ntwo<｜Assistant｜></think>"
    assert processor.apply_chat_template(messages) == expected
    batch = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
    records = image_inputs_from_batch(
        batch["pixel_values"], batch["image_grid_hws"], batch["vision_token_types"], downsample_ratio=2
    )
    assert len(records) == 1 and records[0].start == 3


def test_processor_save_reload_preserves_image_settings_and_outputs(tmp_path: Path) -> None:
    assert resolve_custom_config_cls("deepseek_v41") is DeepseekV41Config
    processor = DeepseekV41Processor(_tokenizer(), _config())
    image = Image.new("RGB", (7, 5), "red")
    expected = processor(IMAGE_PLACEHOLDER, image, return_tensors="pt")
    processor.save_pretrained(tmp_path)
    restored = AutoProcessor.from_pretrained(tmp_path, local_files_only=True, trust_remote_code=False)
    assert isinstance(restored, DeepseekV41Processor)
    assert restored.config.vision_config.patch_size == 2
    assert restored.config.vision_config.downsample_ratio == 2
    actual = restored(IMAGE_PLACEHOLDER, image, return_tensors="pt")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_full_model_text_and_image_forward_backward_reaches_vision_and_delimiters() -> None:
    torch.manual_seed(39)
    tokenizer = _tokenizer()
    text_config = DeepseekV41TextConfig(
        vocab_size=32,
        hidden_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=4,
        o_lora_rank=4,
        o_groups=1,
        hc_mult=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        compress_ratios=[0],
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        candidate_source_layer_id=-1,
        engram_layer_ids=[],
        num_nextn_predict_layers=0,
        dspark_block_size=0,
        dtype="float32",
    )
    config = DeepseekV41Config(
        text_config=text_config,
        vision_config=_config().vision_config,
        image_token_id=tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER),
        dtype="float32",
    )
    model = DeepseekV41ForCausalLM(
        config,
        backend=BackendConfig(
            attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch_mm", dispatcher="torch"
        ),
    )
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    processor = DeepseekV41Processor(tokenizer, config)
    batch = processor(f"one {IMAGE_PLACEHOLDER} two tail", Image.new("RGB", (4, 6), "red"), return_tensors="pt")
    batch["pixel_values"] = batch["pixel_values"].float().requires_grad_()
    labels = batch["input_ids"].clone()
    labels[batch["vision_token_types"] >= 0] = -100
    output = model(**batch, labels=labels)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert torch.isfinite(batch["pixel_values"].grad).all()
    assert batch["pixel_values"].grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        if name.startswith(("model.vision.", "model.aligner.", "model.image_")):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    for name in ("image_start", "image_end", "image_newline"):
        assert model.get_parameter(f"model.{name}").grad.abs().sum() > 0
    # Image placeholders were completely overwritten with visual states, so
    # their ordinary text embedding row has no route to the loss.
    assert model.model.embed_tokens.weight.grad[config.image_token_id].count_nonzero() == 0


@pytest.mark.parametrize("corruption", ["newline", "start", "patch_count", "grid_count"])
def test_image_input_parser_rejects_inconsistent_spans(corruption: str) -> None:
    processor = DeepseekV41Processor(_tokenizer(), _config())
    batch = processor(IMAGE_PLACEHOLDER, Image.new("RGB", (4, 6)), return_tensors="pt")
    if corruption == "newline":
        batch["vision_token_types"][0, 2] = IMAGE
    elif corruption == "start":
        batch["vision_token_types"][0, 0] = TEXT
    elif corruption == "patch_count":
        batch["pixel_values"] = batch["pixel_values"][:-1]
    else:
        batch["image_grid_hws"] = batch["image_grid_hws"][:0]
    with pytest.raises(ValueError):
        image_inputs_from_batch(
            batch["pixel_values"], batch["image_grid_hws"], batch["vision_token_types"], downsample_ratio=2
        )


def test_image_masks_reach_engram_and_modality_routing() -> None:
    tokenizer = _tokenizer()
    config = DeepseekV41Config(
        text_config=DeepseekV41TextConfig(
            vocab_size=32,
            hidden_size=8,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            head_dim=32,
            qk_rope_head_dim=4,
            q_lora_rank=4,
            o_lora_rank=4,
            o_groups=1,
            hc_mult=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            compress_ratios=[0],
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            candidate_source_layer_id=-1,
            engram_layer_ids=[0],
            engram_num_embeddings=[101],
            engram_vocab_size=11,
            engram_max_ngram_size=2,
            engram_n_heads=1,
            engram_head_dim=32,
            engram_compressed_vocab_size=len(tokenizer),
            dtype="float32",
        ),
        vision_config=_config().vision_config,
        image_token_id=tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER),
        dtype="float32",
    )
    model = DeepseekV41ForCausalLM(
        config,
        tokenizer=tokenizer,
        backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch_fp32", experts="torch", dispatcher="torch"),
    )
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    layer = model.model.layers["0"]
    with torch.no_grad():
        layer.mlp.gate.weight.zero_()
        layer.mlp.gate.e_score_correction_bias.copy_(torch.tensor([10.0, 9.0, 0.0, 0.0]))
        layer.mlp.gate.bias_vl.copy_(torch.tensor([0.0, 0.0, 10.0, 9.0]))
    batch = DeepseekV41Processor(tokenizer, config)(
        f"one {IMAGE_PLACEHOLDER} two", Image.new("RGB", (4, 6), "blue"), return_tensors="pt"
    )
    observed = {}

    def record_hash_mask(_module, args, kwargs):
        """Capture the bool [batch, sequence] input mask without changing it."""
        observed["hash_mask"] = kwargs["token_mask"].detach().clone()

    def record_engram(_module, args, kwargs, output):
        """Compare [batch, sequence, streams, hidden] before and after memory writes."""
        observed["engram_mask"] = kwargs["token_mask"].detach().clone()
        image_mask = batch["vision_token_types"] >= 0
        torch.testing.assert_close(output[image_mask], args[0][image_mask], rtol=0, atol=0)

    def record_routes(_module, _args, output):
        """Capture selected expert IDs [tokens, topk] without replacing gate outputs."""
        observed["routes"] = output[1].detach().clone()

    handles = [
        model.model.engram_hash.register_forward_pre_hook(record_hash_mask, with_kwargs=True),
        layer.engram.register_forward_hook(record_engram, with_kwargs=True),
        layer.mlp.gate.register_forward_hook(record_routes),
    ]
    try:
        result = model(**batch)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.isfinite(result.logits).all()
    text_mask = batch["vision_token_types"] < 0
    torch.testing.assert_close(observed["hash_mask"], text_mask, rtol=0, atol=0)
    torch.testing.assert_close(observed["engram_mask"], text_mask, rtol=0, atol=0)
    routes = observed["routes"].sort(-1).values
    torch.testing.assert_close(routes[text_mask.flatten()], torch.tensor([0, 1]).expand(int(text_mask.sum()), 2))
    torch.testing.assert_close(routes[~text_mask.flatten()], torch.tensor([2, 3]).expand(int((~text_mask).sum()), 2))


@pytest.mark.parametrize("vision_block", [False, True])
def test_shared_vision_fsdp_keeps_all_norms_fp32(monkeypatch: pytest.MonkeyPatch, vision_block: bool) -> None:
    from nemo_automodel.components.models.deepseek_v4 import fsdp as v4_fsdp
    from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM
    from nemo_automodel.components.models.deepseek_v4.vision import DeepseekV4VisionRMSNorm
    from nemo_automodel.components.models.deepseek_v41.fsdp import fully_shard_deepseek_v41

    config = _config()
    config.dtype = "bfloat16"
    tower = DeepseekV41VisionTransformer(config)
    module = tower.blocks[0] if vision_block else tower
    norms = [child for child in module.modules() if isinstance(child, DeepseekV4VisionRMSNorm)]
    calls = []
    monkeypatch.setattr(v4_fsdp, "fully_shard", lambda child, **kwargs: calls.append((child, kwargs)))
    policy = torch.distributed.fsdp.MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.bfloat16
    )

    assert DeepseekV4ForCausalLM._nemo_fully_shard is v4_fsdp.fully_shard_deepseek_v4
    assert DeepseekV41ForCausalLM._nemo_fully_shard is fully_shard_deepseek_v41
    fully_shard_deepseek_v41(module, mesh=object(), mp_policy=policy, reshard_after_forward=True)

    assert [child for child, _ in calls] == [*norms, module]
    for _, kwargs in calls[:-1]:
        norm_policy = kwargs["mp_policy"]
        assert norm_policy.param_dtype == norm_policy.reduce_dtype == torch.float32
        assert norm_policy.output_dtype is None
        assert norm_policy.cast_forward_inputs is False
    assert calls[-1][1]["mp_policy"] is policy


def test_v41_nonvision_fsdp_preserves_requested_compute_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_automodel.components.models.deepseek_v41 import fsdp as v41_fsdp

    # FP32 master weights outside vision must still honor BF16 mixed precision.
    module = torch.nn.Linear(8, 8, dtype=torch.float32)
    policy = torch.distributed.fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    calls = []
    monkeypatch.setattr(v41_fsdp, "fully_shard", lambda child, **kwargs: calls.append((child, kwargs)))
    v41_fsdp.fully_shard_deepseek_v41(module, mesh=object(), mp_policy=policy)
    assert len(calls) == 1
    assert calls[0][0] is module
    assert calls[0][1]["mp_policy"] is policy
