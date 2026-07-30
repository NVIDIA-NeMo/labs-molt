# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LoRA config derivation — the PeftConfig handed to AutoModel's from_pretrained.

LoRA itself lives in AutoModel; molt only maps CLI flags onto PeftConfig and refuses the
one parallelism combination AutoModel rejects after the weight load (custom MoE + TP>1).
These cover that mapping plus the CLI surface, so a flag rename can't silently no-op.
"""

import argparse

import pytest

from molt.cli.common_args import add_lora_args
from molt.models.base import _lora_peft_config
from molt.utils.config import hierarchize


def _parse(*argv):
    parser = argparse.ArgumentParser()
    add_lora_args(parser, prefix="model.")
    return hierarchize(parser.parse_args(list(argv))).model.lora


def test_lora_off_by_default_returns_no_peft_config():
    lora = _parse()
    assert lora.rank == 0
    # rank 0 must stay a full-parameter run: a PeftConfig here would freeze the base.
    assert _lora_peft_config(lora.rank, lora.alpha, lora.dropout, lora.target_modules, is_moe=False, tp_size=1) is None


def test_cli_flags_map_onto_peft_config():
    lora = _parse("--model.lora.rank", "16", "--model.lora.alpha", "64", "--model.lora.dropout", "0.05")
    cfg = _lora_peft_config(lora.rank, lora.alpha, lora.dropout, lora.target_modules, is_moe=False, tp_size=1)
    # `rank` is molt's name for what AutoModel calls `dim`; alpha/rank sets the update scale.
    assert (cfg.dim, cfg.alpha, cfg.dropout) == (16, 64, 0.05)


def test_default_target_modules_are_the_dense_projections():
    cfg = _lora_peft_config(8, 32, 0.0, None, is_moe=False, tp_size=1)
    # Patterns are anchored fullmatches, so '*_proj' adapts dense attention/MLP linears but
    # not custom-MoE grouped experts, which are named '*_projs'.
    assert cfg.target_modules == ["*_proj"]


def test_explicit_target_modules_are_forwarded():
    lora = _parse("--model.lora.rank", "8", "--model.lora.target_modules", "*", "*_projs")
    cfg = _lora_peft_config(lora.rank, lora.alpha, lora.dropout, lora.target_modules, is_moe=False, tp_size=1)
    assert cfg.target_modules == ["*", "*_projs"]


def test_moe_with_tensor_parallel_is_rejected_up_front():
    # AutoModel's safe custom-MoE TP path replicates non-expert modules and rejects PEFT only
    # after the weight load; fail here instead so the user isn't billed minutes of loading.
    with pytest.raises(ValueError, match="requires --fsdp.tp_size 1"):
        _lora_peft_config(8, 32, 0.0, None, is_moe=True, tp_size=2)


def test_moe_without_tensor_parallel_is_allowed():
    # EP is the supported way to scale MoE under LoRA, so ep-only must not be blocked.
    assert _lora_peft_config(8, 32, 0.0, ["*"], is_moe=True, tp_size=1) is not None


def test_tensor_parallel_without_moe_is_allowed():
    # Dense TP+LoRA only costs AutoModel's Triton LoRA kernel, so it stays enabled.
    assert _lora_peft_config(8, 32, 0.0, None, is_moe=False, tp_size=8) is not None
