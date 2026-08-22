# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from molt.cli.common_args import add_fsdp_args


def test_fsdp_args_reject_fp16(capsys):
    parser = argparse.ArgumentParser()
    add_fsdp_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--fsdp.param_dtype", "fp16"])

    assert "invalid choice: 'fp16' (choose from 'bf16')" in capsys.readouterr().err


def test_fsdp_args_reject_removed_optimizer_only_offload(capsys):
    parser = argparse.ArgumentParser()
    add_fsdp_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--fsdp.offload", "optimizer"])

    assert "invalid choice: 'optimizer'" in capsys.readouterr().err
