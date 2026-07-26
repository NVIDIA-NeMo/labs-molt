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
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.


class WorkerWrap:
    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl"):
        """Init torch process group for model weights update"""
        import torch

        from molt.utils.distributed_util import stateless_init_process_group

        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        # One rank per vLLM worker GPU. The mp executor places an engine's whole
        # TP*DP worker set in a single torch world (get_rank() is global across the
        # data-parallel replicas), so the plain offset already gives every worker a
        # unique weight-sync rank.
        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weights_packed(self, metas):
        """Receive ONE packed broadcast carrying many weights.

        ``metas`` is a list of ``(name, dtype, shape)``. Producer (rank 0 in
        the trainer) cats all tensors into a single uint8 buffer in the same
        order; here we split + reinterpret-cast back. Replaces thousands of
        per-tensor RPC+broadcast pairs with a handful of ~1 GiB ones.

        Dtype-faithful: each meta carries the sender's own per-param dtype, which
        may differ from ``model_config.dtype`` (e.g. an fp32-kept MoE router/gate).
        We reconstruct each tensor at its *sent* dtype
        (per-meta ``dtype.itemsize`` / ``view(dtype)``) and hand it to vLLM's
        ``load_weights``, which casts to that param's target dtype via
        ``param.data.copy_()``. We must therefore NOT assert a single uniform dtype
        here — the old assert forced every weight through bf16 and silently
        downcast fp32-kept params, corrupting routing.
        """
        import math

        import torch

        sizes = [math.prod(shape) * dtype.itemsize for _, dtype, shape in metas]

        buf = torch.empty(sum(sizes), dtype=torch.uint8, device="cuda")
        self._model_update_group.broadcast(buf, src=0, stream=torch.cuda.current_stream())

        weights = [
            (name, part.view(dtype).view(*shape)) for (name, dtype, shape), part in zip(metas, buf.split(sizes))
        ]
        loaded = self.model_runner.model.load_weights(weights=weights)
        # Warn on EVERY refit flush that vLLM ignored entirely (loaded nothing) -- a real
        # name-format break silently drops those updates -> stale rollout weights.
        # `load_weights` returns the set of *vLLM-internal* param names it assigned, which
        # differ from the HF names we send (vLLM's WeightsMapper strips the outer `model.`
        # prefix and fuses qkv/gate_up), so a per-name diff against our sent names would
        # false-positive on every remapped/fused weight. Keying off "loaded 0 of N" avoids
        # that: a healthy flush maps to >0 params; only a genuine mismatch maps to none.
        # No other refit logging.
        if loaded is not None and len(loaded) == 0 and weights:
            print(
                f"[refit] WARNING: vLLM loaded 0 of {len(weights)} refit weights in a flush "
                f"(names unrecognized -> dropped, rollout stays stale); sample sent: "
                f"{[name for name, _ in weights][:10]}",
                flush=True,
            )
        del buf

    def weight_energy_by_layer(self):
        """This worker's float-param energy (sum of squares) per decoder layer.

        Energy is what makes the trainer's weights and the engine's comparable at all: vLLM
        concatenates (gate_proj|up_proj -> w13_weight, q|k|v -> qkv_proj) and shards params
        across TP/EP workers, and neither transform changes the multiset of values — so the
        per-layer total is identical on both sides iff the engine holds the weights that were
        sent. Names cannot be compared; these numbers can. Layer -1 holds everything outside
        the decoder stack.
        """
        import re

        import torch

        energy: dict[int, float] = {}
        with torch.no_grad():
            for name, param in self.model_runner.model.named_parameters():
                if not param.is_floating_point():
                    continue
                match = re.search(r"layers\.(\d+)", name)
                layer = int(match.group(1)) if match else -1
                energy[layer] = energy.get(layer, 0.0) + float(param.detach().float().square().sum())
        return energy
