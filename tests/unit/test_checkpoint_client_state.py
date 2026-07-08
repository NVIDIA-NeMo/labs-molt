# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-state (de)serialization for resumable checkpoints.

``CheckpointManager`` used to hand-roll a JSON type-tagging codec to squeeze
tensors / tuples / non-str-key dicts — notably a ``StatefulDataLoader.state_dict()``
— into ``extra_state.json``. That is now a plain ``torch.save`` to
``extra_state.pt`` (the same mechanism AutoModel uses for its own dataloader/RNG
state via ``save_on_dp_ranks``). These tests pin the ``torch.save`` round-trip and
the scalar-coercing metric sidecar.

They mirror the exact extra_state read/write lines in ``save_ckpt`` / ``load_ckpt``
(the model/optimizer halves still go through AutoModel's ``Checkpointer`` + DCP
and need a live distributed model, so they are out of scope here).
"""

import json
import os

import torch

from molt.trainer.fsdp.checkpoint import CheckpointManager


class _FakeStrategy:
    def print(self, *msg):  # _read_ckpt_metric logs warnings through this
        pass


def _cm() -> CheckpointManager:
    return CheckpointManager(_FakeStrategy())


def _assert_equal(a, b, path="client_state"):
    """Deep compare allowing tensor value-equality."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), f"{path}: tensor vs {type(b)}"
        assert a.dtype == b.dtype, f"{path}: dtype {a.dtype} != {b.dtype}"
        assert torch.equal(a, b), f"{path}: tensor values differ"
        return
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a.keys()) == set(b.keys()), f"{path}: dict keys differ"
        for k in a:
            _assert_equal(a[k], b[k], f"{path}[{k!r}]")
        return
    if isinstance(a, (list, tuple)):
        assert type(a) is type(b), f"{path}: {type(a)} != {type(b)}"
        assert len(a) == len(b), f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_equal(x, y, f"{path}[{i}]")
        return
    assert a == b, f"{path}: {a!r} != {b!r}"


def _realistic_client_state():
    """Shape mirrors rl_trainer/sft_trainer: scalars + a StatefulDataLoader state
    dict (which carries tensors and non-str / tuple keys via RNG generator state).
    """
    return {
        "episode": 3,
        "global_step": 42,
        "consumed_samples": 42 * 256,
        "total_consumed_prompts": 10752,
        "best_eval_metric_key": "eval/accuracy",
        "best_eval_metric_value": 0.7881,
        "data_loader_state_dict": {
            "_index_sampler_state": {"samples_yielded": 10752},
            "_sampler_iter_state": {
                "generator": torch.randint(0, 255, (16,), dtype=torch.uint8),
                "perm": torch.arange(7, dtype=torch.int64),
            },
            "shard_dtype": torch.float32,
            "epoch_seed_pairs": [(0, 1234), (1, 5678)],
            "by_int_key": {0: "a", 1: "b"},
        },
        "rollout_generator_state_dict": {
            "pending_prompts": [("prompt", "label", ["image.png"])],
        },
    }


def test_torch_save_round_trip(tmp_path):
    """New format: save_ckpt writes extra_state.pt; load_ckpt reads it back exact."""
    cm = _cm()
    cs = _realistic_client_state()

    # ---- exactly what save_ckpt does for client state ----
    extra = {"client_state": dict(cs)}
    pt_path = os.path.join(tmp_path, "extra_state.pt")
    cm._atomic_write_torch(pt_path, extra)

    assert os.path.isfile(pt_path)
    assert not os.path.exists(f"{pt_path}.tmp.{os.getpid()}")  # atomic rename cleaned up

    # ---- exactly what load_ckpt does ----
    loaded = torch.load(pt_path, weights_only=False)
    states = loaded.get("client_state", {}) or {}

    _assert_equal(cs, states)


def test_load_missing_extra_state_is_empty(tmp_path):
    """No extra_state.pt (e.g. very first run) -> empty client state, no crash."""
    extra_pt = os.path.join(tmp_path, "extra_state.pt")
    states = {}
    if os.path.isfile(extra_pt):
        states = torch.load(extra_pt, weights_only=False).get("client_state", {}) or {}
    assert states == {}


def test_metric_round_trip(tmp_path):
    """metric.json: scalar coercion (python/torch/numpy) + read back as float."""
    cm = _cm()
    for raw, expected in [
        (0.788, 0.788),
        (torch.tensor(0.5), 0.5),
        (None, None),
    ]:
        cm._write_ckpt_metric(str(tmp_path), raw, metric_key="eval/accuracy")
        assert cm._read_ckpt_metric(str(tmp_path)) == expected

    # numpy scalar (advantage/eval metrics are often numpy) coerces too
    import numpy as np

    cm._write_ckpt_metric(str(tmp_path), np.float32(0.25), metric_key="k")
    assert abs(cm._read_ckpt_metric(str(tmp_path)) - 0.25) < 1e-6

    # the written file is plain JSON (no type-tagging)
    with open(cm._get_ckpt_metric_path(str(tmp_path))) as f:
        payload = json.load(f)
    assert abs(payload["metric_value"] - 0.25) < 1e-6
