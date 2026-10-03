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

def _inline_qwen2_first_layer(model):
    """Inline Qwen2's first decoder layer while preserving its normal outputs."""
    import types
    from itertools import islice

    from vllm.distributed import get_pp_group
    from vllm.sequence import IntermediateTensors

    backbone = getattr(model, "model", None)
    if type(backbone).__name__ != "Qwen2Model":
        raise RuntimeError("first-layer inlining requires vLLM Qwen2Model")
    if getattr(backbone, "_molt_first_layer_inlined", False):
        return

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
            if idx:
                hidden_states, residual = layer(positions, hidden_states, residual)
                continue
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)
            hidden_states = layer.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = layer.mlp(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    backbone.forward = types.MethodType(forward, backbone)
    backbone._molt_first_layer_inlined = True


def _install_qwen2_first_layer_inline_patch():
    """Install the optional Qwen2 call-boundary alignment before compilation."""
    import os

    if os.environ.get("MOLT_VLLM_INLINE_QWEN_FIRST_LAYER") != "1":
        return

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_molt_first_layer_inline_patch", False):
        return
    load_model = GPUModelRunner.load_model

    def patched_load_model(self, *args, **kwargs):
        result = load_model(self, *args, **kwargs)
        _inline_qwen2_first_layer(self.model)
        return result

    GPUModelRunner.load_model = patched_load_model
    GPUModelRunner._molt_first_layer_inline_patch = True


def _install_vllm_cuda_rope_patch():
    """Use vLLM's native RoPE kernel for full-dimension static rotation."""
    import os

    if (
        os.environ.get("MOLT_USE_VLLM_CUDA_ROPE") != "1"
        and os.environ.get("MOLT_ALIGNMENT_VLLM_CUDA_ROPE") != "1"
    ):
        return

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    if getattr(RotaryEmbedding, "_molt_cuda_static_rope", False):
        return
    original_forward_static = RotaryEmbedding.forward_static

    def forward_static(
        positions, query, key, head_size, rotary_dim, cos_sin_cache, is_neox_style
    ):
        if rotary_dim != head_size:
            return original_forward_static(
                positions,
                query,
                key,
                head_size,
                rotary_dim,
                cos_sin_cache,
                is_neox_style,
            )
        ops.rotary_embedding(
            positions.flatten(),
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox_style,
        )
        return query, key

    RotaryEmbedding.forward_static = staticmethod(forward_static)
    RotaryEmbedding._molt_cuda_static_rope = True
    print("[Alignment] enabled vLLM CUDA static RoPE.")


_install_qwen2_first_layer_inline_patch()
_install_vllm_cuda_rope_patch()


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
        # Collect the names vLLM says it assigned, for the exact by-name coverage check
        # (--train.check_weight_update_equal). Only armed between reset/report calls.
        if getattr(self, "_weight_update_loaded", None) is not None and loaded:
            self._weight_update_loaded.update(loaded)
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

    def reset_weight_update_check(self):
        """Start collecting the param names ``load_weights`` assigns in the coming broadcast."""
        self._weight_update_loaded = set()

    def weight_update_missing(self):
        """This worker's float params that the broadcast never assigned, by exact name.

        ``None`` when the names are not comparable — either the model reports nothing, or it
        reports names from a different namespace than ``named_parameters()`` (vLLM models are
        free to return the pre-mapping or the fused name). The subset test is what makes this
        safe: without it, a namespace mismatch reads as "the whole model is stale", which is
        how a by-name check false-alarms.

        A weight the refit skips on purpose is listed too — a tied ``lm_head`` is never sent
        because it reaches vLLM through ``embed_tokens`` — so read the names, not the count.
        """
        loaded, self._weight_update_loaded = getattr(self, "_weight_update_loaded", None), None
        if not loaded:
            return None
        held = {name for name, param in self.model_runner.model.named_parameters() if param.is_floating_point()}
        return sorted(held - loaded) if loaded <= held else None
