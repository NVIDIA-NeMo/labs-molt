# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for :pyfile:`nemo_automodel/components/distributed/context_parallel/mamba.py`.

Tests mock the distributed process group so they can run on CPU-only CI
systems while still verifying dimension calculations, parameter slicing,
group replication logic, and activation shape transformations.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.distributed.context_parallel.mamba import MambaContextParallel

# ---------------------------------------------------------------------------
# Lightweight stubs for torch.distributed.ProcessGroup
# ---------------------------------------------------------------------------


class _FakeProcessGroup:
    """Minimal stub emulating ``torch.distributed.ProcessGroup`` for unit tests.

    Only ``size()`` and ``rank()`` are used by ``MambaContextParallel.__init__``
    and the parameter-slicing helpers.
    """

    def __init__(self, size: int, rank: int = 0):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


# ---------------------------------------------------------------------------
# Helpers to build a MambaContextParallel instance with dummy parameters
# ---------------------------------------------------------------------------


def _make_conv1d(d_inner: int, n_groups: int, d_state: int, kernel_size: int = 4) -> nn.Conv1d:
    """Create a Conv1d whose weight/bias values are deterministic (arange-based)."""
    conv_dim = d_inner + 2 * n_groups * d_state
    conv = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=kernel_size, groups=conv_dim, bias=True)
    # Fill weight with sequential values for easy verification.
    with torch.no_grad():
        conv.weight.copy_(torch.arange(conv_dim * kernel_size, dtype=torch.float32).reshape(conv_dim, 1, kernel_size))
        conv.bias.copy_(torch.arange(conv_dim, dtype=torch.float32))
    return conv


class _FakeMixer:
    """Minimal object that exposes the attributes MambaContextParallel needs."""

    def __init__(self, conv1d, dt_bias, A_log, D):
        self.conv1d = conv1d
        self.dt_bias = dt_bias
        self.A_log = A_log
        self.D = D


def _make_mamba_cp(
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    cp_size: int,
    cp_rank: int = 0,
    kernel_size: int = 4,
) -> MambaContextParallel:
    """Construct a ``MambaContextParallel`` with deterministic dummy parameters."""
    pg = _FakeProcessGroup(size=cp_size, rank=cp_rank)
    conv1d = _make_conv1d(num_heads * head_dim, n_groups, d_state, kernel_size)
    dt_bias = torch.arange(num_heads, dtype=torch.float32)
    A_log = torch.arange(num_heads, dtype=torch.float32) + 100.0
    D = torch.arange(num_heads, dtype=torch.float32) + 200.0
    mixer = _FakeMixer(conv1d, dt_bias, A_log, D)
    return MambaContextParallel(
        cp_group=pg,
        num_heads=num_heads,
        head_dim=head_dim,
        n_groups=n_groups,
        d_state=d_state,
        mixer=mixer,
    )


@pytest.mark.parametrize(
    "num_heads, n_groups, cp_size, expected_heads_local, expected_d_inner_local, expected_n_groups_local, expected_repeat",
    [
        # Basic: groups == heads, evenly divisible
        (8, 8, 2, 4, 4 * 2, 4, 1),
        (8, 8, 4, 2, 2 * 2, 2, 1),
        # More groups than cp_size
        (16, 8, 4, 4, 4 * 2, 2, 1),
        # n_groups < cp_size -> replication
        (8, 1, 2, 4, 4 * 2, 1, 2),
        (8, 2, 4, 2, 2 * 2, 1, 2),
        (16, 1, 4, 4, 4 * 2, 1, 4),
        # cp_size == 1 (no parallelism)
        (8, 4, 1, 8, 8 * 2, 4, 1),
    ],
    ids=[
        "groups_eq_heads_cp2",
        "groups_eq_heads_cp4",
        "more_groups_cp4",
        "1group_cp2",
        "2groups_cp4",
        "1group_cp4",
        "cp1_noop",
    ],
)
def test_dimension_calculations(
    num_heads,
    n_groups,
    cp_size,
    expected_heads_local,
    expected_d_inner_local,
    expected_n_groups_local,
    expected_repeat,
):
    """Verify computed per-rank dimensions for various (num_heads, n_groups, cp_size) combos."""
    head_dim = 2
    mcp = _make_mamba_cp(num_heads=num_heads, head_dim=head_dim, n_groups=n_groups, d_state=4, cp_size=cp_size)

    assert mcp.num_heads_local == expected_heads_local
    assert mcp.d_inner_local == expected_d_inner_local
    assert mcp.n_groups_local == expected_n_groups_local
    assert mcp.group_repeat_count == expected_repeat
    assert mcp.d_inner == num_heads * head_dim


def test_validation_error_heads_not_divisible_by_cp():
    """num_heads % cp_size != 0 must raise AssertionError."""
    with pytest.raises(AssertionError, match="num_heads.*must be divisible by cp_size"):
        _make_mamba_cp(num_heads=7, head_dim=2, n_groups=1, d_state=4, cp_size=4)


def test_validation_error_cp_not_divisible_by_groups():
    """When n_groups < cp_size, cp_size % n_groups != 0 must raise."""
    with pytest.raises(AssertionError, match="cp_size.*must be divisible by n_groups"):
        _make_mamba_cp(num_heads=12, head_dim=2, n_groups=3, d_state=4, cp_size=4)


def test_validation_error_groups_not_divisible_by_cp():
    """When n_groups >= cp_size, n_groups % cp_size != 0 must raise."""
    with pytest.raises(AssertionError, match="n_groups.*must be divisible by cp_size"):
        _make_mamba_cp(num_heads=12, head_dim=2, n_groups=5, d_state=4, cp_size=4)


class TestParameterSlicing:
    """Verify that get_conv1d_weight, get_conv1d_bias, get_dt_bias, get_A_log, get_D
    return correct slices per CP rank."""

    NUM_HEADS = 8
    HEAD_DIM = 2
    N_GROUPS = 4
    D_STATE = 3
    KERNEL_SIZE = 4
    CP_SIZE = 2

    @property
    def d_inner(self) -> int:
        return self.NUM_HEADS * self.HEAD_DIM

    @property
    def groups_state_size(self) -> int:
        return self.N_GROUPS * self.D_STATE

    @property
    def conv_dim(self) -> int:
        return self.d_inner + 2 * self.groups_state_size

    def _build(self, rank: int) -> MambaContextParallel:
        return _make_mamba_cp(
            num_heads=self.NUM_HEADS,
            head_dim=self.HEAD_DIM,
            n_groups=self.N_GROUPS,
            d_state=self.D_STATE,
            cp_size=self.CP_SIZE,
            cp_rank=rank,
            kernel_size=self.KERNEL_SIZE,
        )

    def test_dt_bias_slicing(self):
        """dt_bias[num_heads] should be sliced into contiguous chunks per rank."""
        for rank in range(self.CP_SIZE):
            mcp = self._build(rank)
            sliced = mcp.get_dt_bias()
            expected_start = rank * (self.NUM_HEADS // self.CP_SIZE)
            expected = torch.arange(self.NUM_HEADS, dtype=torch.float32)[
                expected_start : expected_start + mcp.num_heads_local
            ]
            assert torch.equal(sliced, expected), f"dt_bias mismatch on rank {rank}"

    def test_A_log_slicing(self):
        """A_log[num_heads] should be sliced into contiguous chunks per rank."""
        for rank in range(self.CP_SIZE):
            mcp = self._build(rank)
            sliced = mcp.get_A_log()
            expected_start = rank * mcp.num_heads_local
            expected = (
                torch.arange(self.NUM_HEADS, dtype=torch.float32)[expected_start : expected_start + mcp.num_heads_local]
                + 100.0
            )
            assert torch.equal(sliced, expected), f"A_log mismatch on rank {rank}"

    def test_D_slicing(self):
        """D[num_heads] should be sliced into contiguous chunks per rank."""
        for rank in range(self.CP_SIZE):
            mcp = self._build(rank)
            sliced = mcp.get_D()
            expected_start = rank * mcp.num_heads_local
            expected = (
                torch.arange(self.NUM_HEADS, dtype=torch.float32)[expected_start : expected_start + mcp.num_heads_local]
                + 200.0
            )
            assert torch.equal(sliced, expected), f"D mismatch on rank {rank}"

    def test_conv1d_weight_slicing_shape(self):
        """conv1d weight [conv_dim, 1, K] -> [conv_dim_local, K] per rank (squeezed)."""
        for rank in range(self.CP_SIZE):
            mcp = self._build(rank)
            w = mcp.get_conv1d_weight()
            d_inner_local = self.d_inner // self.CP_SIZE
            n_groups_local = self.N_GROUPS // self.CP_SIZE
            conv_dim_local = d_inner_local + 2 * n_groups_local * self.D_STATE
            assert w.shape == (conv_dim_local, self.KERNEL_SIZE), f"Weight shape mismatch rank {rank}"

    def test_conv1d_bias_slicing_shape(self):
        """conv1d bias [conv_dim] -> [conv_dim_local] per rank."""
        for rank in range(self.CP_SIZE):
            mcp = self._build(rank)
            b = mcp.get_conv1d_bias()
            d_inner_local = self.d_inner // self.CP_SIZE
            n_groups_local = self.N_GROUPS // self.CP_SIZE
            conv_dim_local = d_inner_local + 2 * n_groups_local * self.D_STATE
            assert b.shape == (conv_dim_local,), f"Bias shape mismatch rank {rank}"

    def test_conv1d_weight_slicing_values(self):
        """Verify that the x-portion of conv1d weight is correctly sliced per rank."""
        # The full weight is filled with arange(conv_dim * K).reshape(conv_dim, 1, K).
        # x-portion occupies rows [0, d_inner).  Each rank gets d_inner/cp_size rows.
        # get_conv1d_weight() squeezes dim-1, so expected shape is [rows, K].
        full_weight = torch.arange(self.conv_dim * self.KERNEL_SIZE, dtype=torch.float32).reshape(
            self.conv_dim, self.KERNEL_SIZE
        )
        d_inner_local = self.d_inner // self.CP_SIZE

        for rank in range(self.CP_SIZE):
            mcp = self._build(rank)
            sliced = mcp.get_conv1d_weight()
            x_start = rank * d_inner_local
            x_expected = full_weight[x_start : x_start + d_inner_local]
            x_actual = sliced[:d_inner_local]
            assert torch.equal(x_actual, x_expected), f"Weight x-portion mismatch rank {rank}"

    def test_conv1d_bias_none(self):
        """When conv1d has no bias, get_conv1d_bias() returns None."""
        pg = _FakeProcessGroup(size=2, rank=0)
        conv = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.KERNEL_SIZE,
            groups=self.conv_dim,
            bias=False,
        )
        dt_bias = torch.zeros(self.NUM_HEADS)
        A_log = torch.zeros(self.NUM_HEADS)
        D = torch.zeros(self.NUM_HEADS)
        mixer = _FakeMixer(conv, dt_bias, A_log, D)
        mcp = MambaContextParallel(
            cp_group=pg,
            num_heads=self.NUM_HEADS,
            head_dim=self.HEAD_DIM,
            n_groups=self.N_GROUPS,
            d_state=self.D_STATE,
            mixer=mixer,
        )
        assert mcp.get_conv1d_bias() is None


class TestParameterSlicingWithReplication:
    """When n_groups < cp_size, B/C conv param slicing uses group_repeat_count."""

    NUM_HEADS = 8
    HEAD_DIM = 2
    N_GROUPS = 1  # < cp_size -> replication
    D_STATE = 3
    KERNEL_SIZE = 4
    CP_SIZE = 2

    @property
    def d_inner(self) -> int:
        return self.NUM_HEADS * self.HEAD_DIM

    @property
    def groups_state_size(self) -> int:
        return self.N_GROUPS * self.D_STATE

    @property
    def conv_dim(self) -> int:
        return self.d_inner + 2 * self.groups_state_size

    def _build(self, rank: int) -> MambaContextParallel:
        return _make_mamba_cp(
            num_heads=self.NUM_HEADS,
            head_dim=self.HEAD_DIM,
            n_groups=self.N_GROUPS,
            d_state=self.D_STATE,
            cp_size=self.CP_SIZE,
            cp_rank=rank,
            kernel_size=self.KERNEL_SIZE,
        )

    def test_replicated_groups_conv_weight(self):
        """With n_groups=1, cp_size=2 all ranks should get the same B/C slice."""
        slices = []
        for rank in range(self.CP_SIZE):
            mcp = self._build(rank)
            w = mcp.get_conv1d_weight()
            d_inner_local = self.d_inner // self.CP_SIZE
            bc_portion = w[d_inner_local:]
            slices.append(bc_portion)

        # Both ranks replicate the single group, so B/C slices must be identical.
        assert torch.equal(slices[0], slices[1]), (
            "With n_groups < cp_size, B/C conv param slices should be identical across ranks"
        )

    def test_replicated_groups_conv_weight_shape(self):
        """Verify conv weight shape with replication."""
        mcp = self._build(0)
        w = mcp.get_conv1d_weight()
        d_inner_local = self.d_inner // self.CP_SIZE
        bc_size_local = mcp.n_groups_local * self.D_STATE
        expected_conv_dim_local = d_inner_local + 2 * bc_size_local
        assert w.shape == (expected_conv_dim_local, self.KERNEL_SIZE)


class TestGroupReplication:
    """Verify B/C state replication via expand+reshape when n_groups < cp_size."""

    def test_bc_state_expansion(self):
        """When n_groups=1 and cp_size=2, B/C states should be doubled before all-to-all."""
        num_heads = 8
        head_dim = 2
        n_groups = 1
        d_state = 3
        cp_size = 2
        d_inner = num_heads * head_dim
        groups_state_size = n_groups * d_state

        mcp = _make_mamba_cp(
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            d_state=d_state,
            cp_size=cp_size,
            cp_rank=0,
        )

        B, L_local = 2, 4
        proj_dim = d_inner + d_inner + groups_state_size + groups_state_size + num_heads
        projected = torch.randn(B, L_local, proj_dim)

        captured_calls = []

        def fake_cp2hp(tensor, cp_group, batch_size):
            captured_calls.append(tensor.clone())
            H = tensor.shape[-1]
            H_local = H // cp_size
            return torch.randn(batch_size, L_local * cp_size, H_local)

        with patch(
            "nemo_automodel.components.distributed.context_parallel.mamba._all_to_all_cp2hp", side_effect=fake_cp2hp
        ):
            mcp.pre_conv_ssm(projected)

        assert len(captured_calls) == 5, f"Expected 5 all-to-all calls, got {len(captured_calls)}"

        b_state_input = captured_calls[2]
        c_state_input = captured_calls[3]
        expected_expanded_dim = n_groups * mcp.group_repeat_count * d_state  # 1*2*3 = 6
        assert b_state_input.shape == (B, L_local, expected_expanded_dim), (
            f"B_state should be expanded to dim {expected_expanded_dim}, got {b_state_input.shape[-1]}"
        )
        assert c_state_input.shape == (B, L_local, expected_expanded_dim), (
            f"C_state should be expanded to dim {expected_expanded_dim}, got {c_state_input.shape[-1]}"
        )

    def test_no_expansion_when_groups_ge_cp(self):
        """When n_groups >= cp_size, B/C states should NOT be expanded."""
        num_heads = 8
        head_dim = 2
        n_groups = 4
        d_state = 3
        cp_size = 2
        d_inner = num_heads * head_dim
        groups_state_size = n_groups * d_state

        mcp = _make_mamba_cp(
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            d_state=d_state,
            cp_size=cp_size,
            cp_rank=0,
        )
        assert mcp.group_repeat_count == 1

        B, L_local = 2, 4
        proj_dim = d_inner + d_inner + groups_state_size + groups_state_size + num_heads
        projected = torch.randn(B, L_local, proj_dim)

        captured_calls = []

        def fake_cp2hp(tensor, cp_group, batch_size):
            captured_calls.append(tensor.clone())
            H = tensor.shape[-1]
            H_local = H // cp_size
            return torch.randn(batch_size, L_local * cp_size, H_local)

        with patch(
            "nemo_automodel.components.distributed.context_parallel.mamba._all_to_all_cp2hp", side_effect=fake_cp2hp
        ):
            mcp.pre_conv_ssm(projected)

        b_state_input = captured_calls[2]
        assert b_state_input.shape == (B, L_local, groups_state_size)


class TestPrePostConvSsmShapes:
    """Verify shape transformations of pre_conv_ssm and post_conv_ssm.

    All-to-all calls are mocked to avoid needing real distributed backends.
    The mock simulates the correct shape transformation that all-to-all would
    perform (sequence-sharded <-> hidden-sharded).
    """

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_pre_conv_ssm_output_shape(self, cp_size):
        """pre_conv_ssm: [B, L/cp, proj_dim] -> [B, L, proj_dim/cp]."""
        num_heads = 8
        head_dim = 2
        n_groups = 4
        d_state = 3
        d_inner = num_heads * head_dim
        groups_state_size = n_groups * d_state

        mcp = _make_mamba_cp(
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            d_state=d_state,
            cp_size=cp_size,
            cp_rank=0,
        )

        B = 2
        L_local = 8
        L_global = L_local * cp_size
        proj_dim = d_inner + d_inner + groups_state_size + groups_state_size + num_heads
        projected = torch.randn(B, L_local, proj_dim)

        def fake_cp2hp(tensor, cp_group, batch_size):
            B_t, L_t, H_t = tensor.shape
            H_local = H_t // cp_size
            return torch.randn(B_t, L_t * cp_size, H_local)

        with patch(
            "nemo_automodel.components.distributed.context_parallel.mamba._all_to_all_cp2hp", side_effect=fake_cp2hp
        ):
            output = mcp.pre_conv_ssm(projected)

        d_inner_local = d_inner // cp_size
        n_groups_local = n_groups // cp_size
        groups_state_local = n_groups_local * d_state
        num_heads_local = num_heads // cp_size
        proj_dim_local = d_inner_local + d_inner_local + groups_state_local + groups_state_local + num_heads_local

        assert output.shape == (B, L_global, proj_dim_local), (
            f"Expected ({B}, {L_global}, {proj_dim_local}), got {output.shape}"
        )

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_post_conv_ssm_output_shape(self, cp_size):
        """post_conv_ssm: [B, L, d_inner/cp] -> [B, L/cp, d_inner]."""
        num_heads = 8
        head_dim = 2
        n_groups = 4
        d_state = 3
        d_inner = num_heads * head_dim

        mcp = _make_mamba_cp(
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            d_state=d_state,
            cp_size=cp_size,
            cp_rank=0,
        )

        B = 2
        L_global = 16
        L_local = L_global // cp_size
        d_inner_local = d_inner // cp_size
        ssm_output = torch.randn(B, L_global, d_inner_local)

        def fake_hp2cp(tensor, cp_group, batch_size):
            B_t, L_t, H_t = tensor.shape
            L_out = L_t // cp_size
            H_out = H_t * cp_size
            return torch.randn(B_t, L_out, H_out)

        with patch(
            "nemo_automodel.components.distributed.context_parallel.mamba._all_to_all_hp2cp", side_effect=fake_hp2cp
        ):
            output = mcp.post_conv_ssm(ssm_output)

        assert output.shape == (B, L_local, d_inner), f"Expected ({B}, {L_local}, {d_inner}), got {output.shape}"

    def test_pre_conv_ssm_noop_cp1(self):
        """When cp_size == 1, pre_conv_ssm should return the input unchanged."""
        mcp = _make_mamba_cp(num_heads=4, head_dim=2, n_groups=2, d_state=3, cp_size=1, cp_rank=0)
        inp = torch.randn(2, 8, 4 * 2 + 4 * 2 + 2 * 3 + 2 * 3 + 4)
        out = mcp.pre_conv_ssm(inp)
        assert out is inp, "pre_conv_ssm should be identity when cp_size==1"

    def test_post_conv_ssm_noop_cp1(self):
        """When cp_size == 1, post_conv_ssm should return the input unchanged."""
        mcp = _make_mamba_cp(num_heads=4, head_dim=2, n_groups=2, d_state=3, cp_size=1, cp_rank=0)
        inp = torch.randn(2, 8, 4 * 2)
        out = mcp.post_conv_ssm(inp)
        assert out is inp, "post_conv_ssm should be identity when cp_size==1"


class TestAllToAllLayoutTransforms:
    """Test _all_to_all_cp2hp and _all_to_all_hp2cp with mocked all-to-all.

    We mock _all_to_all (the autograd wrapper) to simulate an identity all-to-all
    (single rank), which lets us verify the reshape/permute logic without a real PG.
    """

    def test_cp2hp_shape(self):
        """Verify _all_to_all_cp2hp output shape with identity all-to-all."""
        from nemo_automodel.components.distributed.context_parallel.mamba import _all_to_all_cp2hp

        cp_size = 2
        B, L_local, H = 2, 4, 8
        pg = _FakeProcessGroup(size=cp_size, rank=0)

        inp = torch.randn(B, L_local, H)

        with patch(
            "nemo_automodel.components.distributed.context_parallel.mamba._all_to_all", side_effect=lambda t, g: t
        ):
            out = _all_to_all_cp2hp(inp, pg, B)

        assert out.shape == (B, L_local * cp_size, H // cp_size)

    def test_hp2cp_shape(self):
        """Verify _all_to_all_hp2cp output shape with identity all-to-all."""
        from nemo_automodel.components.distributed.context_parallel.mamba import _all_to_all_hp2cp

        cp_size = 2
        B, L_global, H_local = 2, 8, 4
        pg = _FakeProcessGroup(size=cp_size, rank=0)

        inp = torch.randn(B, L_global, H_local)

        with patch(
            "nemo_automodel.components.distributed.context_parallel.mamba._all_to_all", side_effect=lambda t, g: t
        ):
            out = _all_to_all_hp2cp(inp, pg, B)

        assert out.shape == (B, L_global // cp_size, H_local * cp_size)


def test_cp_size_1_is_identity():
    """When cp_size == 1, all dimension calculations should match the unpartitioned case."""
    mcp = _make_mamba_cp(num_heads=8, head_dim=4, n_groups=4, d_state=16, cp_size=1, cp_rank=0)

    assert mcp.num_heads_local == 8
    assert mcp.d_inner_local == 32
    assert mcp.n_groups_local == 4
    assert mcp.group_repeat_count == 1

    assert mcp.get_dt_bias().shape == (8,)
    assert mcp.get_A_log().shape == (8,)
    assert mcp.get_D().shape == (8,)

    w = mcp.get_conv1d_weight()
    assert w.shape[0] == 32 + 2 * 4 * 16


def test_parameter_slices_allow_gradient_flow():
    """Sliced parameters should maintain gradient connectivity to the originals."""
    mcp = _make_mamba_cp(num_heads=4, head_dim=2, n_groups=2, d_state=3, cp_size=2, cp_rank=0)

    dt_slice = mcp.get_dt_bias()
    assert dt_slice.data_ptr() == mcp._mixer.dt_bias[:2].data_ptr()

    a_slice = mcp.get_A_log()
    assert a_slice.data_ptr() == mcp._mixer.A_log[:2].data_ptr()

    d_slice = mcp.get_D()
    assert d_slice.data_ptr() == mcp._mixer.D[:2].data_ptr()
