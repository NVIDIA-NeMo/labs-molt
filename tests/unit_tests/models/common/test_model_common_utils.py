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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from nemo_automodel.components.models.common.utils import (
    BackendConfig,
    TEFp8Config,
    compute_lm_head_logits,
    generation_config_from_model_config,
    get_is_first_microbatch,
    get_is_optim_step,
    get_rope_config,
    initialize_linear_module,
    initialize_rms_norm_module,
    load_pretrained_generation_config,
    restore_pretrained_generation_config,
    set_is_first_microbatch,
    set_is_optim_step,
)


class TestQuackBackend:
    def test_quack_linear_preserves_linear_contract(self):
        class FakeQuackLinear(nn.Linear):
            pass

        with patch(
            "nemo_automodel.components.models.common.utils.safe_import_from",
            return_value=(True, FakeQuackLinear),
        ):
            module = initialize_linear_module("quack", 8, 16, bias=True, device="cpu", dtype=torch.float32)

        assert isinstance(module, FakeQuackLinear)
        assert module.weight.shape == (16, 8)
        assert module.bias.shape == (16,)

    def test_quack_rms_norm_preserves_rms_norm_contract(self):
        class FakeQuackRMSNorm(nn.RMSNorm):
            pass

        with patch(
            "nemo_automodel.components.models.common.utils.safe_import_from",
            return_value=(True, FakeQuackRMSNorm),
        ):
            module = initialize_rms_norm_module("quack", 8, eps=1e-6, device="cpu", dtype=torch.float32)

        assert isinstance(module, FakeQuackRMSNorm)
        assert module.weight.shape == (8,)
        assert module.eps == 1e-6

    @pytest.mark.parametrize(
        ("initializer", "args", "match"),
        [
            (initialize_linear_module, ("quack", 8, 16), "linear='quack'"),
            (initialize_rms_norm_module, ("quack", 8), "rms_norm='quack'"),
        ],
    )
    def test_quack_backend_reports_missing_optional_dependency(self, initializer, args, match):
        with (
            patch(
                "nemo_automodel.components.models.common.utils.safe_import_from",
                return_value=(False, object()),
            ),
            pytest.raises(ImportError, match=match),
        ):
            initializer(*args)


class TestIsOptimStep:
    def teardown_method(self):
        set_is_optim_step(False)

    def test_default_is_false(self):
        set_is_optim_step(False)
        assert get_is_optim_step() is False

    def test_set_true(self):
        set_is_optim_step(True)
        assert get_is_optim_step() is True

    def test_set_false(self):
        set_is_optim_step(True)
        set_is_optim_step(False)
        assert get_is_optim_step() is False


class TestIsFirstMicrobatch:
    def teardown_method(self):
        set_is_first_microbatch(None)

    def test_default_is_none(self):
        set_is_first_microbatch(None)
        assert get_is_first_microbatch() is None

    def test_set_true(self):
        set_is_first_microbatch(True)
        assert get_is_first_microbatch() is True

    def test_set_false(self):
        set_is_first_microbatch(False)
        assert get_is_first_microbatch() is False

    def test_set_none(self):
        set_is_first_microbatch(True)
        set_is_first_microbatch(None)
        assert get_is_first_microbatch() is None

    def test_grad_accumulation_lifecycle(self):
        """Simulate the typical GA lifecycle: True -> False -> True -> False."""
        # Start of optimizer step: first microbatch
        set_is_first_microbatch(True)
        assert get_is_first_microbatch() is True

        # After first microbatch
        set_is_first_microbatch(False)
        assert get_is_first_microbatch() is False

        # Next optimizer step: first microbatch again
        set_is_first_microbatch(True)
        assert get_is_first_microbatch() is True

        # After first microbatch
        set_is_first_microbatch(False)
        assert get_is_first_microbatch() is False


class TestTEFp8Config:
    def test_default_recipe(self):
        cfg = TEFp8Config()
        assert cfg.recipe == "current"

    def test_block_recipe(self):
        cfg = TEFp8Config(recipe="block")
        assert cfg.recipe == "block"

    def test_passthrough_recipe_object(self):
        """Non-string recipe objects are passed through directly."""
        sentinel = object()
        cfg = TEFp8Config(recipe=sentinel)
        assert cfg.build_recipe() is sentinel

    def test_maybe_te_autocast_without_te(self):
        """Without TE installed, maybe_te_autocast returns nullcontext."""
        cfg = TEFp8Config()
        with patch("nemo_automodel.components.models.common.utils.HAVE_TE", False):
            ctx = cfg.maybe_te_autocast()
            assert isinstance(ctx, nullcontext)

    def test_build_recipe_without_te(self):
        """Without TE installed, build_recipe returns None."""
        cfg = TEFp8Config()
        with patch("nemo_automodel.components.models.common.utils.HAVE_TE", False):
            assert cfg.build_recipe() is None


class TestBackendConfigTeFp8:
    def test_te_fp8_default_is_none(self):
        """BackendConfig.te_fp8 defaults to None."""
        cfg = BackendConfig()
        assert cfg.te_fp8 is None

    def test_te_fp8_dict_normalized(self):
        """A dict te_fp8 is converted to TEFp8Config."""
        cfg = BackendConfig(te_fp8={"recipe": "block"}, experts="te", dispatcher="deepep")
        assert isinstance(cfg.te_fp8, TEFp8Config)
        assert cfg.te_fp8.recipe == "block"

    def test_te_fp8_requires_te_backend(self):
        """te_fp8 requires linear='te' or experts='te'."""
        with pytest.raises(ValueError, match="te_fp8 requires at least one TE backend"):
            BackendConfig(te_fp8=TEFp8Config(), linear="torch", experts="torch")


class TestGetRopeConfig:
    def test_extracts_rope_theta(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"},
        )
        rope_theta, _, _ = get_rope_config(config)
        assert rope_theta == 1_000_000

    def test_returns_rope_parameters_as_rope_scaling(self):
        params = {"rope_theta": 10_000, "rope_type": "default"}
        config = SimpleNamespace(rope_parameters=params)
        _, rope_parameters, _ = get_rope_config(config)
        assert rope_parameters is params

    def test_partial_rotary_factor(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 10_000, "partial_rotary_factor": 0.5},
        )
        _, _, partial_rotary_factor = get_rope_config(config)
        assert partial_rotary_factor == 0.5

    def test_partial_rotary_factor_defaults_to_one(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 10_000},
        )
        _, _, partial_rotary_factor = get_rope_config(config)
        assert partial_rotary_factor == 1.0

    def test_yarn_scaling_parameters(self):
        config = SimpleNamespace(
            rope_parameters={
                "rope_theta": 1_000_000,
                "rope_type": "yarn",
                "factor": 4.0,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "original_max_position_embeddings": 8192,
            },
        )
        rope_theta, rope_parameters, _ = get_rope_config(config)
        assert rope_theta == 1_000_000
        assert rope_parameters["factor"] == 4.0
        assert rope_parameters["beta_slow"] == 1.0
        assert rope_parameters["beta_fast"] == 32.0
        assert rope_parameters["original_max_position_embeddings"] == 8192


class TestComputeLmHeadLogits:
    """Cover every branch of the shared lm_head projection helper."""

    HIDDEN = 8
    VOCAB = 16

    def _lm_head(self):
        torch.manual_seed(0)
        return nn.Linear(self.HIDDEN, self.VOCAB, bias=False)

    def test_none_lm_head_returns_hidden_states(self):
        """A non-final PP stage (lm_head is None) passes hidden states through."""
        hidden = torch.randn(2, 5, self.HIDDEN)
        out = compute_lm_head_logits(None, hidden, logits_to_keep=0)
        assert out.logits is hidden

    def test_keep_zero_projects_all_positions_3d(self):
        """logits_to_keep == 0 projects every position without slicing (BSHD)."""
        lm_head = self._lm_head()
        hidden = torch.randn(2, 5, self.HIDDEN)
        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=0)
        assert out.logits.shape == (2, 5, self.VOCAB)
        torch.testing.assert_close(out.logits, lm_head(hidden))

    def test_keep_zero_projects_all_positions_2d(self):
        """logits_to_keep == 0 with a 2D [T, H] (THD/packed) hidden state."""
        lm_head = self._lm_head()
        hidden = torch.randn(7, self.HIDDEN)
        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=0)
        assert out.logits.shape == (7, self.VOCAB)
        torch.testing.assert_close(out.logits, lm_head(hidden))

    def test_keep_int_slices_last_n_3d(self):
        """A positive int keeps only the last N positions (BSHD)."""
        lm_head = self._lm_head()
        hidden = torch.randn(2, 5, self.HIDDEN)
        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=2)
        assert out.logits.shape == (2, 2, self.VOCAB)
        torch.testing.assert_close(out.logits, lm_head(hidden[:, -2:, :]))

    def test_keep_int_slices_last_n_2d(self):
        """A positive int keeps only the last N positions (THD/packed)."""
        lm_head = self._lm_head()
        hidden = torch.randn(7, self.HIDDEN)
        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=3)
        assert out.logits.shape == (3, self.VOCAB)
        torch.testing.assert_close(out.logits, lm_head(hidden[-3:, :]))

    def test_keep_tensor_indices_3d(self):
        """A tensor of indices selects those positions (BSHD)."""
        lm_head = self._lm_head()
        hidden = torch.randn(2, 5, self.HIDDEN)
        idx = torch.tensor([0, 2, 4])
        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=idx)
        assert out.logits.shape == (2, 3, self.VOCAB)
        torch.testing.assert_close(out.logits, lm_head(hidden[:, idx, :]))

    def test_keep_tensor_indices_2d(self):
        """A tensor of indices selects those positions (THD/packed)."""
        lm_head = self._lm_head()
        hidden = torch.randn(7, self.HIDDEN)
        idx = torch.tensor([1, 3, 5])
        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=idx)
        assert out.logits.shape == (3, self.VOCAB)
        torch.testing.assert_close(out.logits, lm_head(hidden[idx, :]))

    def test_is_thd_restores_batch_dim_on_2d_logits(self):
        """THD/packed input -> 2D [T, V] logits get a leading batch dim restored."""
        lm_head = self._lm_head()
        hidden = torch.randn(7, self.HIDDEN)
        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=0, is_thd=True)
        assert out.logits.shape == (1, 7, self.VOCAB)
        torch.testing.assert_close(out.logits, lm_head(hidden).unsqueeze(0))

    def test_is_thd_with_logits_to_keep(self):
        """is_thd composes with slicing: last-N 2D logits become [1, N, V]."""
        lm_head = self._lm_head()
        hidden = torch.randn(7, self.HIDDEN)
        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=2, is_thd=True)
        assert out.logits.shape == (1, 2, self.VOCAB)
        torch.testing.assert_close(out.logits, lm_head(hidden[-2:, :]).unsqueeze(0))

    def test_is_thd_noop_when_logits_already_3d(self):
        """A 3D logits result (BSHD path) is left untouched, never made 4D."""
        lm_head = self._lm_head()
        hidden = torch.randn(2, 5, self.HIDDEN)
        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=0, is_thd=True)
        assert out.logits.shape == (2, 5, self.VOCAB)

    def test_is_thd_none_lm_head_unsqueezes_passthrough(self):
        """With lm_head=None, the 2D passthrough hidden states also get the batch dim."""
        hidden = torch.randn(7, self.HIDDEN)
        out = compute_lm_head_logits(None, hidden, logits_to_keep=0, is_thd=True)
        assert out.logits.shape == (1, 7, self.HIDDEN)
        torch.testing.assert_close(out.logits, hidden.unsqueeze(0))

    def test_fp32_lm_head_projects_in_fp32_and_restores_dtype(self):
        """fp32_lm_head upcasts to fp32 for the matmul, then casts logits back to input dtype."""
        lm_head = self._lm_head()  # nn.Linear with an fp32 weight
        hidden = torch.randn(2, 5, self.HIDDEN, dtype=torch.bfloat16)
        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=0, fp32_lm_head=True)
        assert out.logits.dtype == torch.bfloat16
        torch.testing.assert_close(out.logits, lm_head(hidden.float()).to(torch.bfloat16))

    def test_fp32_lm_head_with_slicing(self):
        """fp32_lm_head composes with logits_to_keep slicing (2D THD input)."""
        lm_head = self._lm_head()
        hidden = torch.randn(7, self.HIDDEN, dtype=torch.bfloat16)
        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=3, fp32_lm_head=True)
        assert out.logits.shape == (3, self.VOCAB)
        assert out.logits.dtype == torch.bfloat16
        torch.testing.assert_close(out.logits, lm_head(hidden[-3:, :].float()).to(torch.bfloat16))

    def test_fp32_lm_head_ignored_when_lm_head_none(self):
        """fp32_lm_head is a no-op on the lm_head=None passthrough."""
        hidden = torch.randn(7, self.HIDDEN, dtype=torch.bfloat16)
        out = compute_lm_head_logits(None, hidden, fp32_lm_head=True)
        assert out.logits is hidden

    def test_lm_head_dtype_casts_after_slicing(self):
        lm_head = self._lm_head().to(torch.bfloat16)
        hidden = torch.randn(2, 5, self.HIDDEN, dtype=torch.float32)

        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=2)

        expected = lm_head(hidden[:, -2:, :].to(torch.bfloat16))
        assert out.logits.shape == (2, 2, self.VOCAB)
        assert out.logits.dtype == torch.bfloat16
        torch.testing.assert_close(out.logits, expected)

    def test_fsdp_lm_head_uses_policy_compute_dtype(self):
        hidden_size = self.HIDDEN
        vocab_size = self.VOCAB

        class FakeFSDPHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size, dtype=torch.float32))

            def _get_fsdp_state(self):
                return SimpleNamespace(_mp_policy=SimpleNamespace(param_dtype=torch.bfloat16))

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                """Project hidden states of shape [batch, sequence, hidden]."""
                assert hidden_states.dtype == torch.bfloat16
                return hidden_states @ self.weight.to(hidden_states.dtype).T

        lm_head = FakeFSDPHead()
        hidden = torch.randn(2, 5, self.HIDDEN, dtype=torch.float32)

        with patch("nemo_automodel.components.models.common.utils.FSDPModule", FakeFSDPHead):
            out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=2)

        assert lm_head.weight.dtype == torch.float32
        assert out.logits.shape == (2, 2, self.VOCAB)
        assert out.logits.dtype == torch.bfloat16

    def test_output_hidden_states_attaches_hidden(self):
        """output_hidden_states attaches the final hidden states; default leaves them None."""
        lm_head = self._lm_head()
        hidden = torch.randn(2, 5, self.HIDDEN)
        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=0)
        assert out.hidden_states is None
        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=0, output_hidden_states=True)
        assert out.hidden_states is hidden

    def test_output_hidden_states_thd_restores_batch_dim(self):
        """With is_thd, a 2D hidden state gets a leading batch dim in the hidden_states field."""
        lm_head = self._lm_head()
        hidden = torch.randn(7, self.HIDDEN)
        out = compute_lm_head_logits(lm_head, hidden, logits_to_keep=0, is_thd=True, output_hidden_states=True)
        assert out.hidden_states.shape == (1, 7, self.HIDDEN)
        torch.testing.assert_close(out.hidden_states, hidden.unsqueeze(0))


class TestGenerationConfigHelpers:
    def test_from_model_config_takes_stop_tokens_from_a_pretrained_config(self):
        from transformers import PretrainedConfig

        config = PretrainedConfig(bos_token_id=3, eos_token_id=[2, 11], pad_token_id=0)

        generation_config = generation_config_from_model_config(config)

        assert generation_config.bos_token_id == 3
        assert generation_config.eos_token_id == [2, 11]
        assert generation_config.pad_token_id == 0

    def test_from_model_config_falls_back_to_defaults_for_plain_objects(self):
        generation_config = generation_config_from_model_config(SimpleNamespace(eos_token_id=2))

        assert generation_config.eos_token_id is None

    def test_load_pretrained_returns_none_without_any_config_file(self, tmp_path):
        assert load_pretrained_generation_config(tmp_path) is None

    def test_load_pretrained_falls_back_to_the_generation_fields_of_config_json(self, tmp_path):
        """Legacy checkpoints keep do_sample/temperature in config.json only; HF reads them from there."""
        import json

        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "llama", "eos_token_id": 2, "do_sample": True, "temperature": 0.5})
        )

        generation_config = load_pretrained_generation_config(tmp_path)

        assert generation_config.eos_token_id == 2
        assert generation_config.do_sample is True
        assert generation_config.temperature == 0.5

    def test_restore_replaces_a_real_generation_config_only(self, tmp_path):
        from transformers import GenerationConfig

        GenerationConfig(eos_token_id=[2, 11]).save_pretrained(tmp_path)
        can_generate = nn.Module()
        can_generate.generation_config = GenerationConfig()
        cannot_generate = nn.Module()

        restore_pretrained_generation_config(can_generate, tmp_path)
        restore_pretrained_generation_config(cannot_generate, tmp_path)

        assert can_generate.generation_config.eos_token_id == [2, 11]
        assert not hasattr(cannot_generate, "generation_config")

    def test_load_pretrained_fallback_keeps_an_override_that_equals_a_default(self, tmp_path):
        """``eos_token_id=None`` is a deliberate "do not stop"; it equals the default, so
        the diff-based merge cannot see it and the caller has to name it."""
        import json

        from transformers import PretrainedConfig

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "eos_token_id": 0}))

        generation_config = load_pretrained_generation_config(
            tmp_path, config=PretrainedConfig(eos_token_id=None), config_overrides={"eos_token_id"}
        )

        assert generation_config.eos_token_id is None

    def test_load_pretrained_reads_the_subfolder_of_a_local_directory(self, tmp_path):
        """transformers resolves an absolute local path to the parent's generation_config.json
        even with ``subfolder`` set; the helper joins the directory itself so the child wins."""
        from transformers import GenerationConfig

        GenerationConfig(eos_token_id=7).save_pretrained(tmp_path)
        GenerationConfig(eos_token_id=[2, 11]).save_pretrained(tmp_path / "child")

        assert load_pretrained_generation_config(tmp_path, subfolder="child").eos_token_id == [2, 11]

    def test_load_pretrained_forwards_the_hub_loading_options(self, tmp_path):
        from transformers import GenerationConfig

        GenerationConfig(eos_token_id=3).save_pretrained(tmp_path)

        with patch.object(GenerationConfig, "from_pretrained", wraps=GenerationConfig.from_pretrained) as spy:
            load_pretrained_generation_config(tmp_path, revision="abc123", token="secret", local_files_only=True)

        _, forwarded = spy.call_args
        assert forwarded["revision"] == "abc123"
        assert forwarded["token"] == "secret"
        assert forwarded["local_files_only"] is True

    @pytest.mark.parametrize("disk_eos", [2, None], ids=["conflicting", "omitted"])
    def test_load_pretrained_fallback_keeps_explicit_config_overrides(self, tmp_path, disk_eos):
        """An eos override passed at load time already sits on the in-memory config; the
        config.json fallback must neither replace it with the file's value nor with None."""
        import json

        from transformers import PretrainedConfig

        raw = {"model_type": "llama", "do_sample": True}
        if disk_eos is not None:
            raw["eos_token_id"] = disk_eos
        (tmp_path / "config.json").write_text(json.dumps(raw))

        generation_config = load_pretrained_generation_config(tmp_path, config=PretrainedConfig(eos_token_id=9))

        assert generation_config.eos_token_id == 9
        assert generation_config.do_sample is True

    def test_load_pretrained_reads_the_checkpoint_generation_file(self, tmp_path):
        from transformers import GenerationConfig

        GenerationConfig(eos_token_id=[2, 11], do_sample=True, temperature=0.7).save_pretrained(tmp_path)

        generation_config = load_pretrained_generation_config(tmp_path)

        assert generation_config.eos_token_id == [2, 11]
        assert generation_config.do_sample is True
        assert generation_config.temperature == 0.7
