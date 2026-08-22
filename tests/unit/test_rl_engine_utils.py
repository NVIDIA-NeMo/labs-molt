# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
import torch.nn as nn

from nemo_automodel.components.datasets.datum import LossInputLayout
from nemo_automodel.engine import Engine, LossFnOutputBatch, PerTokenOutput, collate_prebatched

from molt.models.critic import _ValueHead, _install_value_head
from molt.trainer.workers.engine_utils import prepare_rl_engine_datum


def _experience():
    return SimpleNamespace(
        sequences=torch.tensor([[10, 11, 12, 13, 0], [20, 21, 22, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
        action_mask=torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
        values=torch.tensor([[0.0, 0.1, 0.2, 0.0], [0.3, 0.4, 0.0, 0.0]]),
        returns=torch.tensor([[0.0, 1.1, 1.2, 0.0], [1.3, 1.4, 0.0, 0.0]]),
        mm_train_inputs=[],
    )


def _wrapper(**kwargs):
    defaults = {"model": nn.Linear(1, 1), "is_vlm": False, "_packing_style": "automodel"}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_prepare_padded_rl_datum_uses_shifted_prediction_axis():
    experience = _experience()
    prepared = prepare_rl_engine_datum(
        experience,
        _wrapper(),
        loss_fields={"old_values": experience.values, "returns": experience.returns},
        packing_samples=False,
    )

    assert prepared.dense_shape == (2, 4)
    assert prepared.packed_indices is None
    assert torch.equal(prepared.datum.model_inputs["input_ids"], experience.sequences[:, :-1])
    assert torch.equal(
        prepared.datum.model_inputs["attention_mask"],
        torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
    )
    assert torch.equal(
        prepared.datum.model_inputs["position_ids"],
        torch.tensor([[0, 1, 2, 1], [0, 1, 1, 1]]),
    )
    assert torch.equal(prepared.datum.loss_fn_inputs["target_tokens"], experience.sequences[:, 1:])
    assert set(prepared.datum.loss_fn_input_layouts.values()) == {LossInputLayout.PER_TOKEN}


def test_prepare_packed_rl_datum_uses_automodel_thd_and_restores_dense_output():
    experience = _experience()
    prepared = prepare_rl_engine_datum(
        experience,
        _wrapper(),
        loss_fields={"old_values": experience.values, "returns": experience.returns},
        packing_samples=True,
    )

    model_inputs = prepared.datum.model_inputs
    assert model_inputs["qkv_format"] == "thd"
    assert torch.equal(model_inputs["input_ids"], torch.tensor([[10, 11, 12, 20, 21]]))
    assert torch.equal(model_inputs["seq_lens"], torch.tensor([[3, 2]]))
    assert torch.equal(prepared.datum.loss_fn_inputs["weights"], torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]]))

    restored = prepared.restore_token_output(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))
    assert torch.equal(restored, torch.tensor([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 0.0, 0.0]]))


def test_prepare_vlm_rl_datum_keeps_media_and_builds_token_types():
    experience = _experience()
    experience.sequences[0, 1] = 99
    experience.mm_train_inputs = [
        {"pixel_values": torch.ones(1, 3, 2, 2), "image_grid_thw": torch.tensor([[1, 2, 2]])},
        None,
    ]
    prepared = prepare_rl_engine_datum(
        experience,
        _wrapper(is_vlm=True, _image_token_id=99, _video_token_id=None),
        loss_fields={"old_values": experience.values, "returns": experience.returns},
        packing_samples=False,
    )

    assert "position_ids" not in prepared.datum.model_inputs
    assert prepared.datum.model_inputs["pixel_values"].shape == (1, 3, 2, 2)
    assert prepared.datum.model_inputs["mm_token_type_ids"][0, 1] == 1


class _HeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(initializer_range=0.01, tie_word_embeddings=True)
        self.config.text_config = SimpleNamespace(tie_word_embeddings=True)
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, head):
        self.lm_head = head


def test_install_value_head_replaces_task_head_before_fsdp():
    model = _HeadModel()

    assert _install_value_head(model) is None

    assert isinstance(model.lm_head, _ValueHead)
    assert model.lm_head.proj.weight.shape == (1, 4)
    assert not model.config.tie_word_embeddings
    assert not model.config.text_config.tie_word_embeddings


class _ScalarValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, input_ids):
        return input_ids.float().unsqueeze(-1) * self.scale


def test_prebatched_rl_datum_runs_one_engine_backward_window():
    experience = _experience()
    model = _ScalarValueModel()
    prepared = prepare_rl_engine_datum(
        experience,
        _wrapper(model=model),
        loss_fields={"old_values": experience.values, "returns": experience.returns},
        packing_samples=False,
    )
    engine = Engine(model, device="cpu", microbatch_size=1, collate_fn=collate_prebatched)

    def loss_fn(output, loss_inputs):
        values = output.squeeze(-1)
        weights = loss_inputs["weights"]
        loss_matrix = 0.5 * (values - loss_inputs["returns"]).pow(2)
        return (loss_matrix * weights).sum(), LossFnOutputBatch(
            per_token={"action_values": PerTokenOutput(values * weights)}
        )

    result = engine.forward_backward([prepared.datum], loss_fn)
    expected_values = experience.sequences[:, :-1].float() * 0.1
    expected_sum = (0.5 * (expected_values - experience.returns).pow(2) * experience.action_mask).sum()

    assert torch.allclose(result.loss_sum, expected_sum.double())
    assert torch.allclose(result.loss, expected_sum.double() / experience.action_mask.sum())
    assert torch.equal(result.loss_fn_outputs[0]["action_values"], expected_values * experience.action_mask)
    assert model.scale.grad is not None
