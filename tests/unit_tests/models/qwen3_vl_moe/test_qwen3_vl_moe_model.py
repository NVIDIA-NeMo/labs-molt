# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeModelOutputWithPast,
)

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_vl_moe.model import (
    Fp32SafeQwen3VLMoeTextRotaryEmbedding,
    Fp32SafeQwen3VLMoeVisionRotaryEmbedding,
    ModelClass,
    Qwen3VLMoeBlock,
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeModel,
    Qwen3VLMoeTextModelBackend,
)
from nemo_automodel.components.moe.config import MoEConfig

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def device():
    return torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture
def text_config():
    return Qwen3VLMoeTextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        router_aux_loss_coef=0.01,
        norm_topk_prob=False,
        pad_token_id=0,
        rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 1.0},
    )


@pytest.fixture
def moe_config(text_config):
    return MoEConfig(
        dim=text_config.hidden_size,
        inter_dim=text_config.intermediate_size,
        moe_inter_dim=text_config.moe_intermediate_size,
        n_routed_experts=text_config.num_experts,
        n_shared_experts=0,
        n_activated_experts=text_config.num_experts_per_tok,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=text_config.router_aux_loss_coef,
        norm_topk_prob=text_config.norm_topk_prob,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        softmax_before_topk=True,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def vl_config(text_config):
    vision_cfg = dict(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=32,
        num_position_embeddings=8,
        deepstack_visual_indexes=[0, 1],
    )
    return Qwen3VLMoeConfig(text_config=text_config.to_dict(), vision_config=vision_cfg)


@_requires_cuda
class TestFp32SafeRotaryEmbeddings:
    def test_text_rotary_inv_freq_remains_fp32(self, text_config):
        rotary = Fp32SafeQwen3VLMoeTextRotaryEmbedding(config=text_config)
        originals = {
            name: buf.detach().clone()
            for name, buf in rotary.named_buffers(recurse=False)
            if name in ("inv_freq", "original_inv_freq")
        }

        rotary = rotary.to(torch.float16)

        assert originals
        for name, original in originals.items():
            buf = getattr(rotary, name)
            assert buf.dtype == torch.float32
            torch.testing.assert_close(buf.float(), original.float())

    def test_vision_rotary_inv_freq_remains_fp32(self):
        rotary = Fp32SafeQwen3VLMoeVisionRotaryEmbedding(dim=16)
        originals = {
            name: buf.detach().clone()
            for name, buf in rotary.named_buffers(recurse=False)
            if name in ("inv_freq", "original_inv_freq")
        }

        rotary = rotary.to(torch.float16)

        assert originals
        for name, original in originals.items():
            buf = getattr(rotary, name)
            assert buf.dtype == torch.float32
            torch.testing.assert_close(buf.float(), original.float())


@_requires_cuda
class TestQwen3VLMoeBlock:
    """Tests for Qwen3VLMoeBlock position_embeddings to freqs_cis conversion."""

    def test_forward_converts_position_embeddings_to_freqs_cis(self, text_config, backend_config, moe_config, device):
        """Test that position_embeddings (cos, sin) are converted to freqs_cis format."""
        block = Qwen3VLMoeBlock(0, text_config, moe_config, backend_config).to(device)

        batch, seq_len = 2, 4
        hidden_size = text_config.hidden_size
        head_dim = text_config.head_dim

        x = torch.randn(batch, seq_len, hidden_size, device=device)
        # position_embeddings: (cos, sin) each with shape [..., head_dim * 2]
        cos = torch.randn(batch, seq_len, head_dim * 2, device=device)
        sin = torch.randn(batch, seq_len, head_dim * 2, device=device)
        position_embeddings = (cos, sin)

        # Mock parent forward to capture freqs_cis
        captured_kwargs = {}

        def mock_forward(self, x, freqs_cis, **kwargs):
            captured_kwargs["freqs_cis"] = freqs_cis
            return x

        with patch.object(block.__class__.__bases__[0], "forward", mock_forward):
            block.forward(x=x, position_embeddings=position_embeddings)

        # Verify freqs_cis was constructed from position_embeddings
        assert "freqs_cis" in captured_kwargs
        freqs_cis = captured_kwargs["freqs_cis"]
        # freqs_cis should be cat of cos[:head_dim] and sin[:head_dim]
        expected_freqs_cis = torch.cat((cos[..., :head_dim], sin[..., :head_dim]), dim=-1)
        torch.testing.assert_close(freqs_cis, expected_freqs_cis)

    def test_forward_uses_freqs_cis_directly_when_provided(self, text_config, backend_config, moe_config, device):
        """Test that freqs_cis is used directly when provided (position_embeddings ignored)."""
        block = Qwen3VLMoeBlock(0, text_config, moe_config, backend_config).to(device)

        batch, seq_len = 2, 4
        hidden_size = text_config.hidden_size
        head_dim = text_config.head_dim

        x = torch.randn(batch, seq_len, hidden_size, device=device)
        freqs_cis = torch.randn(batch, seq_len, head_dim * 2, device=device)

        captured_kwargs = {}

        def mock_forward(self, x, freqs_cis, **kwargs):
            captured_kwargs["freqs_cis"] = freqs_cis
            return x

        with patch.object(block.__class__.__bases__[0], "forward", mock_forward):
            block.forward(x=x, freqs_cis=freqs_cis)

        # Verify the same freqs_cis was passed through
        torch.testing.assert_close(captured_kwargs["freqs_cis"], freqs_cis)

    def test_forward_raises_when_no_freqs_cis_or_position_embeddings(
        self, text_config, backend_config, moe_config, device
    ):
        """Test that ValueError is raised when neither freqs_cis nor position_embeddings provided."""
        block = Qwen3VLMoeBlock(0, text_config, moe_config, backend_config).to(device)

        x = torch.randn(2, 4, text_config.hidden_size, device=device)

        with pytest.raises(ValueError, match="requires freqs_cis or position_embeddings"):
            block.forward(x=x)


@_requires_cuda
class TestQwen3VLMoeTextModelBackendLayersDict:
    """Tests for layers being nn.ModuleDict instead of nn.ModuleList."""

    def test_layers_is_module_dict(self, text_config, backend_config, moe_config):
        """Test that layers is an nn.ModuleDict with string keys."""
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)

        assert isinstance(model.layers, nn.ModuleDict)
        assert all(isinstance(key, str) for key in model.layers.keys())
        assert list(model.layers.keys()) == [str(i) for i in range(text_config.num_hidden_layers)]

    def test_layers_are_qwen3vlmoe_blocks(self, text_config, backend_config, moe_config):
        """Test that each layer is a Qwen3VLMoeBlock instance."""
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)

        for layer in model.layers.values():
            assert isinstance(layer, Qwen3VLMoeBlock)


@_requires_cuda
class TestQwen3VLMoeTextModelBackend:
    def test_initialization_sets_expected_components(self, text_config, backend_config, moe_config):
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)

        assert model.config is text_config
        assert model.backend is backend_config
        assert model.embed_tokens.num_embeddings == text_config.vocab_size
        assert len(model.layers) == text_config.num_hidden_layers
        assert isinstance(model.rotary_emb, Fp32SafeQwen3VLMoeTextRotaryEmbedding)

    def test_forward_skips_norm_when_none(self, text_config, backend_config, moe_config, device):
        """Test that forward() skips norm layer when it is None (PP support)."""
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config).to(device)
        model.norm = None  # Simulate PP stage without norm

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq_len), device=device)

        cos = torch.zeros(3, batch, seq_len, text_config.head_dim * 2, device=device)
        sin = torch.ones_like(cos)

        for layer in model.layers.values():
            layer.forward = MagicMock(side_effect=lambda x, **_: x)

        with patch.object(model.rotary_emb, "forward", return_value=(cos, sin)):
            # Should not raise when norm is None
            output = model(input_ids=input_ids)

        assert isinstance(output, Qwen3VLMoeModelOutputWithPast)
        assert output.last_hidden_state.shape == (batch, seq_len, text_config.hidden_size)

    def test_forward_runs_layers_and_returns_output(self, text_config, backend_config, moe_config, device):
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config).to(device)
        batch, seq_len = 2, 3
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq_len), device=device)

        cos = torch.zeros(3, batch, seq_len, text_config.head_dim * 2, device=device)
        sin = torch.ones_like(cos)

        for layer in model.layers.values():
            layer.forward = MagicMock(side_effect=lambda x, **_: x + 1)

        with patch.object(model.rotary_emb, "forward", return_value=(cos, sin)):
            output = model(input_ids=input_ids)

        assert isinstance(output, Qwen3VLMoeModelOutputWithPast)
        assert output.last_hidden_state.shape == (batch, seq_len, text_config.hidden_size)
        assert output.past_key_values is None
        assert all(layer.forward.call_count == 1 for layer in model.layers.values())
        freqs_shape = model.layers["0"].forward.call_args.kwargs["freqs_cis"].shape
        assert freqs_shape == (3, batch, seq_len, text_config.head_dim * 2)

    def test_forward_applies_deepstack_visual_embeds(self, text_config, backend_config, moe_config, device):
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config).to(device)
        batch, seq_len = 1, 2
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq_len), device=device)

        cos = torch.zeros(3, batch, seq_len, text_config.head_dim * 2, device=device)
        sin = torch.ones_like(cos)

        for layer in model.layers.values():
            layer.forward = MagicMock(side_effect=lambda x, **_: x)

        deepstack_visual_embeds = [
            torch.randn(1, text_config.hidden_size, device=device) for _ in range(len(model.layers))
        ]
        visual_pos_masks = torch.tensor([[True, False]], device=device)

        with (
            patch.object(model.rotary_emb, "forward", return_value=(cos, sin)),
            patch.object(model, "_deepstack_process", side_effect=lambda hs, *_: hs) as mock_deepstack,
        ):
            model(
                input_ids=input_ids,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )

        assert mock_deepstack.call_count == len(model.layers)

    def test_deepstack_process_adds_visual_embeds(self, text_config, backend_config, moe_config, device):
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config).to(device)

        hidden_states = torch.zeros(1, 3, text_config.hidden_size, device=device)
        visual_pos_masks = torch.tensor([[False, True, False]], device=device)
        visual_embeds = torch.full((1, text_config.hidden_size), 2.0, device=device)

        out = model._deepstack_process(hidden_states.clone(), visual_pos_masks, visual_embeds)

        torch.testing.assert_close(out[visual_pos_masks], visual_embeds)
        torch.testing.assert_close(out[visual_pos_masks.logical_not()], hidden_states[visual_pos_masks.logical_not()])

    def test_init_weights_invokes_layer_init(self, text_config, backend_config, moe_config):
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)

        for layer in model.layers.values():
            layer.init_weights = MagicMock()

        original = model.embed_tokens.weight.clone()

        with patch.object(model.norm, "reset_parameters") as mock_norm:
            buffer_ctx = torch.cuda.device(torch.cuda.current_device())
            model.init_weights(buffer_device=buffer_ctx)

        mock_norm.assert_called_once()
        assert not torch.equal(model.embed_tokens.weight, original)
        for layer in model.layers.values():
            layer.init_weights.assert_called_once()

    def test_get_set_input_embeddings(self, text_config, backend_config, moe_config):
        model = Qwen3VLMoeTextModelBackend(text_config, backend=backend_config, moe_config=moe_config)
        new_embed = nn.Embedding(text_config.vocab_size, text_config.hidden_size)

        model.set_input_embeddings(new_embed)

        assert model.get_input_embeddings() is new_embed


@_requires_cuda
class TestQwen3VLMoeForConditionalGeneration:
    def test_initialization_configures_backend_components(self, vl_config, backend_config, moe_config):
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)

        assert model.backend is backend_config
        assert isinstance(model.model, Qwen3VLMoeModel)
        assert isinstance(model.model.language_model, Qwen3VLMoeTextModelBackend)
        assert model.model.moe_config is model.model.language_model.moe_config

        vision_model = getattr(model.model, "visual")
        assert isinstance(vision_model.rotary_pos_emb, Fp32SafeQwen3VLMoeVisionRotaryEmbedding)
        assert vision_model.rotary_pos_emb.inv_freq.dtype == torch.float32

    def test_pad_token_id_defaults_to_negative_one_when_missing(self, vl_config, backend_config, moe_config):
        """Test that pad_token_id defaults to -1 when text_config.pad_token_id is 0 (falsy)."""
        # Test that the model correctly handles the getattr fallback
        # When pad_token_id is 0 (falsy but valid), it should use 0, not -1
        vl_config.text_config.pad_token_id = 0
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)
        # 0 is a valid pad_token_id, should not fallback to -1
        assert model.pad_token_id == 0

    def test_pad_token_id_uses_config_value_when_present(self, vl_config, backend_config, moe_config):
        """Test that pad_token_id uses the config value when present."""
        vl_config.text_config.pad_token_id = 42

        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)

        assert model.pad_token_id == 42

    def test_pad_token_id_defaults_to_negative_one_when_none(self, vl_config, backend_config, moe_config):
        """Test that pad_token_id defaults to -1 when text_config.pad_token_id is None."""
        vl_config.text_config.pad_token_id = None

        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)

        assert model.pad_token_id == -1

    def test_forward_returns_logits_from_lm_head(self, vl_config, backend_config, moe_config, device):
        """Test that forward() returns logits from lm_head when present."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        # Get model dtype for consistent tensor creation
        model_dtype = next(model.parameters()).dtype

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)

        with patch.object(model.model, "forward") as mock_model_forward:
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(
                batch, seq_len, vl_config.text_config.hidden_size, device=device, dtype=model_dtype
            )
            mock_model_forward.return_value = mock_output

            out = model.forward(input_ids=input_ids)

        # Should return logits with vocab_size dimension
        logits = out.logits
        assert logits.shape == (batch, seq_len, vl_config.text_config.vocab_size)
        mock_model_forward.assert_called_once()

    def test_forward_returns_hidden_states_when_no_lm_head(self, vl_config, backend_config, moe_config, device):
        """Test that forward() returns hidden states when lm_head is None."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        model.lm_head = None  # Simulate PP stage without lm_head

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)

        with patch.object(model.model, "forward") as mock_model_forward:
            mock_output = MagicMock()
            hidden_states = torch.randn(batch, seq_len, vl_config.text_config.hidden_size, device=device)
            mock_output.last_hidden_state = hidden_states
            mock_model_forward.return_value = mock_output

            result = model.forward(input_ids=input_ids)

        # Should return hidden states directly
        torch.testing.assert_close(result, hidden_states)

    def test_forward_retrieves_pixel_values_from_stored_chunks(self, vl_config, backend_config, moe_config, device):
        """Test that forward() retrieves pixel_values from stored chunks for PP VLM support."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        # Get model dtype for consistent tensor creation
        model_dtype = next(model.parameters()).dtype

        batch, seq_len = 1, 4
        # Create input_ids with media tokens (151655 for image)
        input_ids = torch.tensor([[151655, 1, 2, 3]], device=device)

        # Store pixel_values chunks for PP VLM support
        pixel_values_chunk = torch.randn(1, 3, 4, 4, device=device, dtype=model_dtype)
        image_grid_hws_chunk = torch.tensor([[2, 2]], device=device)  # [N, 2] format
        model._vlm_pixel_values_chunks = [pixel_values_chunk]
        model._vlm_image_grid_hws_chunks = [image_grid_hws_chunk]
        model._vlm_chunk_idx = 0

        captured_kwargs = {}

        with patch.object(model.model, "forward") as mock_model_forward:
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(
                batch, seq_len, vl_config.text_config.hidden_size, device=device, dtype=model_dtype
            )
            mock_model_forward.return_value = mock_output

            def capture_kwargs(*args, **kwargs):
                captured_kwargs.update(kwargs)
                return mock_output

            mock_model_forward.side_effect = capture_kwargs
            model.forward(input_ids=input_ids)

        # pixel_values should be retrieved from chunks
        assert "pixel_values" in captured_kwargs
        torch.testing.assert_close(captured_kwargs["pixel_values"], pixel_values_chunk)
        # image_grid_thw should be converted from [N, 2] to [N, 3] format
        assert "image_grid_thw" in captured_kwargs
        assert captured_kwargs["image_grid_thw"].shape == (1, 3)
        # Chunk index should be incremented
        assert model._vlm_chunk_idx == 1

    def test_forward_handles_thd_format(self, vl_config, backend_config, moe_config, device):
        """Test that forward() correctly handles thd format by calling squeeze_input_for_thd."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        model_dtype = next(model.parameters()).dtype

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        attention_mask = torch.ones(batch, seq_len, device=device)
        padding_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)

        squeezed_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)
        squeezed_position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        squeezed_padding_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
        squeezed_kwargs = {"foo": "bar"}

        # Mock the model.model.forward to avoid internal tensor operations
        mock_hidden = torch.randn(batch, seq_len, vl_config.text_config.hidden_size, device=device, dtype=model_dtype)

        with (
            patch(
                "nemo_automodel.components.models.qwen3_vl_moe.model.squeeze_input_for_thd",
                return_value=(squeezed_ids, squeezed_position_ids, squeezed_padding_mask, squeezed_kwargs),
            ) as mock_squeeze,
            patch.object(model.model, "forward") as mock_model_forward,
        ):
            mock_output = MagicMock()
            mock_output.last_hidden_state = mock_hidden
            mock_model_forward.return_value = mock_output

            result = model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                qkv_format="thd",
            )

            # Result should be logits from lm_head
            assert result.logits.shape == (batch, seq_len, vl_config.text_config.vocab_size)

            # Verify squeeze_input_for_thd was called with correct args
            squeeze_args = mock_squeeze.call_args[0]
            assert squeeze_args[0] is input_ids
            assert squeeze_args[1] is position_ids
            assert squeeze_args[2] is padding_mask
            assert squeeze_args[3]["qkv_format"] == "thd"

            # Verify model.model.forward was called
            mock_model_forward.assert_called_once()

    def test_initialize_weights_invokes_language_model(self, vl_config, backend_config, moe_config):
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)

        with (
            patch.object(model.model.language_model, "init_weights") as mock_init,
            patch("torch.nn.init.trunc_normal_") as mock_trunc,
        ):
            buffer_ctx = torch.cuda.device(torch.cuda.current_device())
            model.initialize_weights(buffer_device=buffer_ctx, dtype=torch.float32)

        mock_init.assert_called_once()
        mock_trunc.assert_called_once()
        assert model.lm_head.weight.dtype == torch.float32

    def test_state_dict_adapter_created_when_enabled(self, vl_config, backend_config, moe_config):
        backend_config.enable_hf_state_dict_adapter = True
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)

        assert hasattr(model, "state_dict_adapter")


@_requires_cuda
class TestQwen3VLMoeModel:
    def test_property_accessors_delegate_to_language_model(self, vl_config, backend_config, moe_config):
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config)
        core = model.model

        assert isinstance(core, Qwen3VLMoeModel)
        assert core.layers is core.language_model.layers
        assert core.embed_tokens is core.language_model.embed_tokens
        assert core.norm is core.language_model.norm

    def test_forward_uses_embed_tokens_when_inputs_embeds_not_provided(
        self, vl_config, backend_config, moe_config, device
    ):
        """Test that forward() uses embed_tokens to create inputs_embeds from input_ids."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        core = model.model

        batch, seq_len = 2, 3
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)

        with patch.object(core.language_model, "forward") as mock_lang_forward:
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(
                batch, seq_len, vl_config.text_config.hidden_size, device=device
            )
            mock_lang_forward.return_value = mock_output

            core.forward(input_ids=input_ids)

        # language_model.forward should be called with inputs_embeds derived from input_ids
        call_kwargs = mock_lang_forward.call_args.kwargs
        assert call_kwargs["input_ids"] is None
        assert call_kwargs["inputs_embeds"] is not None
        assert call_kwargs["inputs_embeds"].shape == (batch, seq_len, vl_config.text_config.hidden_size)

    def test_forward_accepts_float_input_ids_as_inputs_embeds(self, vl_config, backend_config, moe_config, device):
        """Test that forward() treats float tensor input_ids as inputs_embeds (PP support)."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        core = model.model

        batch, seq_len = 2, 3
        # Pass float tensor as input_ids (this is actually inputs_embeds from previous PP stage)
        float_input = torch.randn(
            batch, seq_len, vl_config.text_config.hidden_size, device=device, dtype=torch.bfloat16
        )

        with (
            patch.object(core, "get_input_embeddings", return_value=None),
            patch.object(core.language_model, "forward") as mock_lang_forward,
        ):
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(
                batch, seq_len, vl_config.text_config.hidden_size, device=device
            )
            mock_lang_forward.return_value = mock_output

            core.forward(input_ids=float_input)

            # language_model.forward should be called with the float tensor as inputs_embeds
            call_kwargs = mock_lang_forward.call_args.kwargs
            assert call_kwargs["input_ids"] is None
            torch.testing.assert_close(call_kwargs["inputs_embeds"], float_input)

    def test_forward_raises_when_no_embeds_and_no_embed_tokens(self, vl_config, backend_config, moe_config, device):
        """Test that forward() raises ValueError when no inputs_embeds and no embed_tokens."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        core = model.model

        batch, seq_len = 2, 3
        # Pass integer input_ids without embed_tokens
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)

        with patch.object(core, "get_input_embeddings", return_value=None):
            with pytest.raises(ValueError, match="inputs_embeds must be provided"):
                core.forward(input_ids=input_ids)


@_requires_cuda
class TestQwen3VLMoeModelInlineVision:
    """Tests for the inline vision processing path (replaces super().forward(input_ids=None))."""

    def test_inline_vision_passes_input_ids_to_get_placeholder_mask(
        self, vl_config, backend_config, moe_config, device
    ):
        """Test that inline vision path passes input_ids (not None) to get_placeholder_mask."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        core = model.model

        batch, seq_len = 1, 4
        hidden_size = vl_config.text_config.hidden_size
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)
        inputs_embeds = torch.randn(batch, seq_len, hidden_size, device=device)
        pixel_values = torch.randn(1, 3, 4, 4, device=device)
        image_grid_thw = torch.tensor([[1, 2, 2]], device=device)

        captured_input_ids = {}

        mock_image_output = MagicMock()
        mock_image_output.pooler_output = [torch.randn(2, hidden_size, device=device)]
        mock_image_output.deepstack_features = [torch.randn(2, hidden_size, device=device)]

        def capture_get_placeholder_mask(input_ids_arg, inputs_embeds=None, image_features=None, video_features=None):
            captured_input_ids["input_ids"] = input_ids_arg
            mask = torch.zeros(batch, seq_len, hidden_size, dtype=torch.bool, device=device)
            return mask, mask

        mock_lang_output = MagicMock()
        mock_lang_output.last_hidden_state = torch.randn(batch, seq_len, hidden_size, device=device)

        with (
            patch.object(core, "get_image_features", return_value=mock_image_output),
            patch.object(core, "get_placeholder_mask", side_effect=capture_get_placeholder_mask),
            patch.object(
                core,
                "get_rope_index",
                return_value=(torch.zeros(3, batch, seq_len, device=device), torch.zeros(batch, device=device)),
            ),
            patch.object(core.language_model, "forward", return_value=mock_lang_output),
        ):
            core.forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        # input_ids should be passed (not None) for integer comparison
        assert captured_input_ids["input_ids"] is not None
        assert torch.equal(captured_input_ids["input_ids"], input_ids)

    def test_inline_vision_calls_get_rope_index_with_keyword_args(self, vl_config, backend_config, moe_config, device):
        """Test that get_rope_index is called with keyword args (not positional)."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        core = model.model

        batch, seq_len = 1, 4
        hidden_size = vl_config.text_config.hidden_size
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)
        inputs_embeds = torch.randn(batch, seq_len, hidden_size, device=device)
        pixel_values = torch.randn(1, 3, 4, 4, device=device)
        image_grid_thw = torch.tensor([[1, 2, 2]], device=device)
        mm_token_type_ids = torch.zeros(batch, seq_len, dtype=torch.int, device=device)

        mock_image_output = MagicMock()
        mock_image_output.pooler_output = [torch.randn(2, hidden_size, device=device)]
        mock_image_output.deepstack_features = [torch.randn(2, hidden_size, device=device)]

        mock_mask = torch.zeros(batch, seq_len, hidden_size, dtype=torch.bool, device=device)
        mock_lang_output = MagicMock()
        mock_lang_output.last_hidden_state = torch.randn(batch, seq_len, hidden_size, device=device)

        with (
            patch.object(core, "get_image_features", return_value=mock_image_output),
            patch.object(core, "get_placeholder_mask", return_value=(mock_mask, mock_mask)),
            patch.object(
                core,
                "get_rope_index",
                return_value=(torch.zeros(3, batch, seq_len, device=device), torch.zeros(batch, device=device)),
            ) as mock_rope,
            patch.object(core.language_model, "forward", return_value=mock_lang_output),
        ):
            core.forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )

        # get_rope_index should be called with keyword args
        call_kwargs = mock_rope.call_args.kwargs
        assert "mm_token_type_ids" in call_kwargs
        assert "image_grid_thw" in call_kwargs
        assert "video_grid_thw" in call_kwargs
        torch.testing.assert_close(call_kwargs["mm_token_type_ids"], mm_token_type_ids)

    def test_inline_vision_returns_qwen3vlmoe_output(self, vl_config, backend_config, moe_config, device):
        """Test that inline vision path returns Qwen3VLMoeModelOutputWithPast."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        core = model.model

        batch, seq_len = 1, 4
        hidden_size = vl_config.text_config.hidden_size
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)
        inputs_embeds = torch.randn(batch, seq_len, hidden_size, device=device)
        pixel_values = torch.randn(1, 3, 4, 4, device=device)
        image_grid_thw = torch.tensor([[1, 2, 2]], device=device)
        position_ids = torch.zeros(3, batch, seq_len, dtype=torch.long, device=device)

        mock_image_output = MagicMock()
        mock_image_output.pooler_output = [torch.randn(2, hidden_size, device=device)]
        mock_image_output.deepstack_features = [torch.randn(2, hidden_size, device=device)]

        mock_mask = torch.zeros(batch, seq_len, hidden_size, dtype=torch.bool, device=device)
        mock_lang_output = MagicMock()
        mock_lang_output.last_hidden_state = torch.randn(batch, seq_len, hidden_size, device=device)

        with (
            patch.object(core, "get_image_features", return_value=mock_image_output),
            patch.object(core, "get_placeholder_mask", return_value=(mock_mask, mock_mask)),
            patch.object(core.language_model, "forward", return_value=mock_lang_output),
        ):
            result = core.forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
            )

        assert isinstance(result, Qwen3VLMoeModelOutputWithPast)
        assert result.last_hidden_state is not None

    def test_inline_vision_skipped_when_no_pixel_values(self, vl_config, backend_config, moe_config, device):
        """Test that text-only path is used when no pixel_values."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        core = model.model

        batch, seq_len = 1, 4
        input_ids = torch.randint(0, vl_config.text_config.vocab_size, (batch, seq_len), device=device)

        with (
            patch.object(core, "get_image_features") as mock_img,
            patch.object(core.language_model, "forward") as mock_lang,
        ):
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(
                batch, seq_len, vl_config.text_config.hidden_size, device=device
            )
            mock_lang.return_value = mock_output

            core.forward(input_ids=input_ids)

        # get_image_features should NOT be called
        mock_img.assert_not_called()
        # language_model should be called directly
        mock_lang.assert_called_once()


@_requires_cuda
class TestQwen3VLMoeForConditionalGenerationPPGuard:
    """Tests for the PP attention mask size guard."""

    def test_pp_attention_mask_dropped_on_size_mismatch(self, vl_config, backend_config, moe_config, device):
        """Test that mismatched attention_mask is dropped for PP non-Stage-0 ranks."""
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        model_dtype = next(model.parameters()).dtype

        batch, seq_len_embeds, seq_len_mask = 1, 4, 8
        inputs_embeds = torch.randn(
            batch, seq_len_embeds, vl_config.text_config.hidden_size, device=device, dtype=model_dtype
        )
        attention_mask = torch.ones(batch, seq_len_mask, device=device)

        captured_kwargs = {}
        with patch.object(model.model, "forward") as mock_forward:
            mock_output = MagicMock()
            mock_output.last_hidden_state = inputs_embeds
            mock_forward.return_value = mock_output

            def capture(*, attention_mask=None, **kw):
                captured_kwargs["attention_mask"] = attention_mask
                return mock_output

            mock_forward.side_effect = lambda *a, **kw: capture(**kw)
            model.forward(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        # Mismatched mask should be dropped (set to None)
        assert captured_kwargs["attention_mask"] is None

    def test_video_token_id_is_151656(self, vl_config, backend_config, moe_config, device):
        """Test that video token ID 151656 (not 151652) is used for media token detection.

        After the PP video chunking refactor, image and video chunks live on
        separate attributes (``_vlm_pixel_values_chunks`` /
        ``_vlm_image_grid_hws_chunks`` for images,
        ``_vlm_pixel_values_videos_chunks`` / ``_vlm_video_grid_thw_chunks``
        for videos). A video token in ``input_ids`` must therefore consume a
        video chunk, not an image chunk.
        """
        model = Qwen3VLMoeForConditionalGeneration(vl_config, backend=backend_config, moe_config=moe_config).to(device)
        model_dtype = next(model.parameters()).dtype

        # input_ids with video token 151656
        input_ids = torch.tensor([[1, 151656, 2, 3]], device=device)

        pixel_values_videos_chunk = torch.randn(8, 3, 2, 2, device=device, dtype=model_dtype)
        video_grid_thw_chunk = torch.tensor([[1, 2, 2]], device=device)
        model._vlm_pixel_values_videos_chunks = [pixel_values_videos_chunk]
        model._vlm_video_grid_thw_chunks = [video_grid_thw_chunk]
        model._vlm_chunk_idx = 0

        with patch.object(model.model, "forward") as mock_forward:
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(
                1, 4, vl_config.text_config.hidden_size, device=device, dtype=model_dtype
            )
            mock_forward.return_value = mock_output

            model.forward(input_ids=input_ids)

        # Chunk should have been consumed (idx incremented)
        assert model._vlm_chunk_idx == 1


@_requires_cuda
class TestQwen3VLMoeFromPretrainedAndModelClass:
    def test_from_pretrained_classmethod(self, vl_config):
        # Use the tiny `vl_config` fixture instead of the default ``Qwen3VLMoeConfig()``,
        # which describes the full model (24 layers, 60 experts, 151K vocab) and takes
        # ~100s to materialize on GPU. The classmethod's delegation behaviour is
        # independent of model size.
        # `vl_config` already carries a valid tiny rope/pad setup (it is the same
        # config every other test in this module builds and runs forward with), so no
        # extra rope_parameters/pad_token_id massaging is required here.
        cfg = vl_config

        with (
            patch(
                "transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe.Qwen3VLMoeConfig.from_pretrained",
                return_value=cfg,
            ) as mock_from_pretrained,
            patch.object(
                Qwen3VLMoeForConditionalGeneration, "from_config", wraps=Qwen3VLMoeForConditionalGeneration.from_config
            ) as mock_from_config,
        ):
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained("qwen3/vl-moe")

        assert isinstance(model, Qwen3VLMoeForConditionalGeneration)
        mock_from_pretrained.assert_called_once_with("qwen3/vl-moe")
        assert mock_from_config.call_args[0][0] is cfg

    def test_modelclass_export_exists(self):
        assert ModelClass is Qwen3VLMoeForConditionalGeneration


class TestQwen3VLMoeModelInlineVisionCpu:
    """CPU-runnable coverage for the inline vision path.

    Invokes `Qwen3VLMoeModel.forward` as an unbound method with a MagicMock `self`
    so we exercise the new branches without instantiating the full CUDA model.
    """

    @staticmethod
    def _mock_self(spatial_merge_size: int = 2, dtype: torch.dtype = torch.float32):
        ms = MagicMock()
        ms.visual = MagicMock()
        ms.visual.dtype = dtype
        ms.visual.spatial_merge_size = spatial_merge_size
        ms.rope_deltas = None
        ms.get_input_embeddings.return_value = None
        return ms

    @staticmethod
    def _zero_mask(batch: int, seq_len: int, hidden: int):
        return torch.zeros(batch, seq_len, hidden, dtype=torch.bool)

    def test_images_only_path_passes_input_ids_and_returns_qwen3vlmoe_output(self):
        ms = self._mock_self()
        batch, seq_len, hidden = 1, 4, 8
        input_ids = torch.tensor([[10, 20, 30, 40]])
        inputs_embeds = torch.randn(batch, seq_len, hidden)
        pixel_values = torch.randn(4, 3, 2, 2)
        image_grid_thw = torch.tensor([[1, 2, 2]])
        position_ids = torch.zeros(3, batch, seq_len, dtype=torch.long)

        mock_image_output = MagicMock()
        mock_image_output.pooler_output = [torch.randn(2, hidden)]
        mock_image_output.deepstack_features = [torch.randn(2, hidden)]
        ms.get_image_features.return_value = mock_image_output

        zero_mask = self._zero_mask(batch, seq_len, hidden)
        captured = {}

        def capture_placeholder(ids, inputs_embeds=None, image_features=None, video_features=None):
            captured["input_ids"] = ids
            return zero_mask, zero_mask

        ms.get_placeholder_mask.side_effect = capture_placeholder

        mock_lang_output = MagicMock()
        mock_lang_output.last_hidden_state = torch.randn(batch, seq_len, hidden)
        ms.language_model.return_value = mock_lang_output

        result = Qwen3VLMoeModel.forward(
            ms,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
        )

        assert captured["input_ids"] is not None
        assert torch.equal(captured["input_ids"], input_ids)
        ms.get_image_features.assert_called_once()
        assert isinstance(result, Qwen3VLMoeModelOutputWithPast)
        assert result.last_hidden_state is not None

    def test_videos_only_path_computes_rope_when_position_ids_missing(self):
        ms = self._mock_self()
        batch, seq_len, hidden = 1, 4, 8
        input_ids = torch.tensor([[10, 20, 30, 40]])
        inputs_embeds = torch.randn(batch, seq_len, hidden)
        pixel_values_videos = torch.randn(4, 3, 2, 2)
        video_grid_thw = torch.tensor([[1, 2, 2]])
        mm_token_type_ids = torch.zeros(batch, seq_len, dtype=torch.int)

        mock_video_output = MagicMock()
        mock_video_output.pooler_output = [torch.randn(2, hidden)]
        mock_video_output.deepstack_features = [torch.randn(2, hidden)]
        ms.get_video_features.return_value = mock_video_output

        zero_mask = self._zero_mask(batch, seq_len, hidden)
        ms.get_placeholder_mask.return_value = (zero_mask, zero_mask)

        rope_pos = torch.zeros(3, batch, seq_len, dtype=torch.long)
        rope_deltas = torch.zeros(batch, dtype=torch.long)
        ms.get_rope_index.return_value = (rope_pos, rope_deltas)

        mock_lang_output = MagicMock()
        mock_lang_output.last_hidden_state = torch.randn(batch, seq_len, hidden)
        ms.language_model.return_value = mock_lang_output

        result = Qwen3VLMoeModel.forward(
            ms,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

        ms.get_video_features.assert_called_once()
        ms.get_rope_index.assert_called_once()
        call_kwargs = ms.get_rope_index.call_args.kwargs
        assert "mm_token_type_ids" in call_kwargs
        assert "video_grid_thw" in call_kwargs
        torch.testing.assert_close(call_kwargs["mm_token_type_ids"], mm_token_type_ids)
        assert isinstance(result, Qwen3VLMoeModelOutputWithPast)

    def test_images_and_videos_merged_visual_call(self):
        ms = self._mock_self()
        batch, seq_len, hidden = 1, 4, 8
        input_ids = torch.tensor([[10, 20, 30, 40]])
        inputs_embeds = torch.randn(batch, seq_len, hidden)
        pixel_values = torch.randn(4, 3, 2, 2)
        pixel_values_videos = torch.randn(4, 3, 2, 2)
        image_grid_thw = torch.tensor([[1, 2, 2]])
        video_grid_thw = torch.tensor([[1, 2, 2]])
        position_ids = torch.zeros(3, batch, seq_len, dtype=torch.long)

        merged_output = MagicMock()
        merged_output.pooler_output = torch.randn(2, hidden)
        merged_output.deepstack_features = [torch.randn(2, hidden)]
        ms.visual.return_value = merged_output

        image_mask = torch.zeros(batch, seq_len, hidden, dtype=torch.bool)
        image_mask[0, 0, :] = True
        video_mask = torch.zeros(batch, seq_len, hidden, dtype=torch.bool)
        video_mask[0, 1, :] = True
        ms.get_placeholder_mask.side_effect = [
            (image_mask, image_mask),
            (video_mask, video_mask),
        ]

        mock_lang_output = MagicMock()
        mock_lang_output.last_hidden_state = torch.randn(batch, seq_len, hidden)
        ms.language_model.return_value = mock_lang_output

        result = Qwen3VLMoeModel.forward(
            ms,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
        )

        ms.visual.assert_called_once()
        visual_kwargs = ms.visual.call_args.kwargs
        assert "grid_thw" in visual_kwargs
        assert visual_kwargs["grid_thw"].shape[0] == 2
        ms.get_image_features.assert_not_called()
        ms.get_video_features.assert_not_called()
        lang_kwargs = ms.language_model.call_args.kwargs
        assert lang_kwargs["visual_pos_masks"] is not None
        assert lang_kwargs["deepstack_visual_embeds"] is not None
        assert len(lang_kwargs["deepstack_visual_embeds"]) == 1
        assert isinstance(result, Qwen3VLMoeModelOutputWithPast)

    def test_text_only_path_skips_inline_vision(self):
        ms = self._mock_self()
        batch, seq_len, hidden = 1, 4, 8
        inputs_embeds = torch.randn(batch, seq_len, hidden)

        mock_lang_output = MagicMock()
        mock_lang_output.last_hidden_state = torch.randn(batch, seq_len, hidden)
        ms.language_model.return_value = mock_lang_output

        Qwen3VLMoeModel.forward(ms, input_ids=None, inputs_embeds=inputs_embeds)

        ms.get_image_features.assert_not_called()
        ms.get_video_features.assert_not_called()
        ms.visual.assert_not_called()
        ms.language_model.assert_called_once()

    def test_dict_attention_mask_unwrapped_for_rope(self):
        ms = self._mock_self()
        batch, seq_len, hidden = 1, 4, 8
        input_ids = torch.tensor([[10, 20, 30, 40]])
        inputs_embeds = torch.randn(batch, seq_len, hidden)
        pixel_values = torch.randn(4, 3, 2, 2)
        image_grid_thw = torch.tensor([[1, 2, 2]])
        full_mask = torch.ones(batch, seq_len, dtype=torch.long)
        attention_mask = {"full_attention": full_mask}

        mock_image_output = MagicMock()
        mock_image_output.pooler_output = [torch.randn(2, hidden)]
        mock_image_output.deepstack_features = [torch.randn(2, hidden)]
        ms.get_image_features.return_value = mock_image_output

        zero_mask = self._zero_mask(batch, seq_len, hidden)
        ms.get_placeholder_mask.return_value = (zero_mask, zero_mask)
        ms.get_rope_index.return_value = (
            torch.zeros(3, batch, seq_len, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
        )

        mock_lang_output = MagicMock()
        mock_lang_output.last_hidden_state = torch.randn(batch, seq_len, hidden)
        ms.language_model.return_value = mock_lang_output

        Qwen3VLMoeModel.forward(
            ms,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        rope_kwargs = ms.get_rope_index.call_args.kwargs
        torch.testing.assert_close(rope_kwargs["attention_mask"], full_mask)


class TestQwen3VLMoeForConditionalGenerationPpGuardCpu:
    """CPU-runnable coverage for the PP mask guard and chunked pixel_values dispatch."""

    @staticmethod
    def _mock_self():
        ms = MagicMock()
        ms._vlm_pixel_values_chunks = None
        ms._vlm_image_grid_hws_chunks = None
        ms._vlm_chunk_idx = 0
        return ms

    def test_pp_attention_mask_dropped_on_seq_len_mismatch(self):
        ms = self._mock_self()
        hidden = 8
        inputs_embeds = torch.randn(1, 4, hidden)
        attention_mask = torch.ones(1, 8, dtype=torch.long)

        captured = {}

        def capture(**kw):
            captured["attention_mask"] = kw.get("attention_mask")
            captured["padding_mask"] = kw.get("padding_mask")
            out = MagicMock()
            out.last_hidden_state = kw["inputs_embeds"]
            return out

        ms.model.side_effect = lambda **kw: capture(**kw)

        Qwen3VLMoeForConditionalGeneration.forward(
            ms,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            padding_mask=torch.ones(1, 8, dtype=torch.long),
        )

        assert captured["attention_mask"] is None
        assert captured["padding_mask"] is None

    def test_pp_attention_mask_preserved_on_matching_seq_len(self):
        ms = self._mock_self()
        hidden = 8
        inputs_embeds = torch.randn(1, 4, hidden)
        attention_mask = torch.ones(1, 4, dtype=torch.long)

        captured = {}

        def capture(**kw):
            captured["attention_mask"] = kw.get("attention_mask")
            out = MagicMock()
            out.last_hidden_state = kw["inputs_embeds"]
            return out

        ms.model.side_effect = lambda **kw: capture(**kw)

        Qwen3VLMoeForConditionalGeneration.forward(
            ms,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        assert captured["attention_mask"] is not None
        torch.testing.assert_close(captured["attention_mask"], attention_mask)

    def test_chunked_pixel_values_consumed_for_image_token(self):
        """Image chunks are consumed when input_ids contains <|image_pad|> (151655)."""
        ms = self._mock_self()
        pixel_chunk = torch.randn(4, 3, 2, 2)
        grid_hws_chunk = torch.tensor([[2, 2]])
        ms._vlm_pixel_values_chunks = [pixel_chunk]
        ms._vlm_image_grid_hws_chunks = [grid_hws_chunk]
        ms._vlm_chunk_idx = 0

        input_ids = torch.tensor([[1, 151655, 2, 3]])

        captured = {}

        def capture(**kw):
            captured["pixel_values"] = kw.get("pixel_values")
            captured["image_grid_thw"] = kw.get("image_grid_thw")
            out = MagicMock()
            out.last_hidden_state = torch.randn(1, 4, 8)
            return out

        ms.model.side_effect = lambda **kw: capture(**kw)

        Qwen3VLMoeForConditionalGeneration.forward(ms, input_ids=input_ids)

        assert ms._vlm_chunk_idx == 1
        assert captured["pixel_values"] is pixel_chunk
        thw = captured["image_grid_thw"]
        assert thw is not None and thw.shape == (1, 3)
        assert torch.equal(thw[:, 0], torch.ones(1, dtype=thw.dtype))
        assert torch.equal(thw[:, 1:], grid_hws_chunk)

    def test_chunked_pixel_values_videos_consumed_for_video_token(self):
        """Video chunks are consumed when input_ids contains <|video_pad|> (151656).

        Image chunks must not be consumed for video-only input — image and video
        streams now route through separate chunk arrays after the PP video
        chunking refactor.
        """
        ms = self._mock_self()
        video_chunk = torch.randn(8, 3, 2, 2)
        video_grid_chunk = torch.tensor([[1, 2, 2]])
        ms._vlm_pixel_values_videos_chunks = [video_chunk]
        ms._vlm_video_grid_thw_chunks = [video_grid_chunk]
        ms._vlm_pixel_values_chunks = None
        ms._vlm_image_grid_hws_chunks = None
        ms._vlm_chunk_idx = 0

        input_ids = torch.tensor([[1, 151656, 2, 3]])

        captured = {}

        def capture(**kw):
            captured["pixel_values_videos"] = kw.get("pixel_values_videos")
            captured["video_grid_thw"] = kw.get("video_grid_thw")
            captured["pixel_values"] = kw.get("pixel_values")
            out = MagicMock()
            out.last_hidden_state = torch.randn(1, 4, 8)
            return out

        ms.model.side_effect = lambda **kw: capture(**kw)

        Qwen3VLMoeForConditionalGeneration.forward(ms, input_ids=input_ids)

        assert ms._vlm_chunk_idx == 1
        assert captured["pixel_values_videos"] is video_chunk
        assert captured["video_grid_thw"] is video_grid_chunk
        assert captured["pixel_values"] is None  # image stream untouched

    def test_chunked_pixel_values_skipped_without_media_tokens(self):
        ms = self._mock_self()
        pixel_chunk = torch.randn(4, 3, 2, 2)
        grid_hws_chunk = torch.tensor([[2, 2]])
        ms._vlm_pixel_values_chunks = [pixel_chunk]
        ms._vlm_image_grid_hws_chunks = [grid_hws_chunk]
        ms._vlm_chunk_idx = 0

        input_ids = torch.tensor([[1, 2, 3, 4]])

        captured = {}

        def capture(**kw):
            captured["pixel_values"] = kw.get("pixel_values")
            out = MagicMock()
            out.last_hidden_state = torch.randn(1, 4, 8)
            return out

        ms.model.side_effect = lambda **kw: capture(**kw)

        Qwen3VLMoeForConditionalGeneration.forward(ms, input_ids=input_ids)

        assert ms._vlm_chunk_idx == 0
        assert captured["pixel_values"] is None

    def test_chunked_pixel_values_passthrough_when_already_3d_grid(self):
        ms = self._mock_self()
        pixel_chunk = torch.randn(4, 3, 2, 2)
        grid_thw_chunk = torch.tensor([[1, 2, 2]])
        ms._vlm_pixel_values_chunks = [pixel_chunk]
        ms._vlm_image_grid_hws_chunks = [grid_thw_chunk]
        ms._vlm_chunk_idx = 0

        input_ids = torch.tensor([[1, 151655, 2, 3]])

        captured = {}

        def capture(**kw):
            captured["image_grid_thw"] = kw.get("image_grid_thw")
            out = MagicMock()
            out.last_hidden_state = torch.randn(1, 4, 8)
            return out

        ms.model.side_effect = lambda **kw: capture(**kw)

        Qwen3VLMoeForConditionalGeneration.forward(ms, input_ids=input_ids)

        torch.testing.assert_close(captured["image_grid_thw"], grid_thw_chunk)
