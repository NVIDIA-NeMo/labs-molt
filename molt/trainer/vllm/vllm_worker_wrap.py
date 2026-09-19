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


def _enable_automodel_residual_norm(model):
    """Match AutoModel's add-then-RMSNorm residual convention for Qwen."""
    import os

    if os.environ.get("MOLT_VLLM_MATCH_AUTOMODEL_RESIDUAL_NORM") != "1":
        return
    backbone = getattr(model, "model", model)
    layers = list(getattr(backbone, "layers", ()))
    if not layers:
        raise RuntimeError("AutoModel residual-norm matching requires model.layers")
    if getattr(backbone, "_molt_automodel_residual_norm", False):
        return

    trace = None
    trace_stages = None
    copy_trace = None
    trace_layer_index = 0
    trace_token_index = 0
    if os.environ.get("MOLT_ALIGNMENT_CUDAGRAPH_TRACE") == "1":
        import torch

        trace_layer_index = int(
            os.environ.get("MOLT_ALIGNMENT_CUDAGRAPH_TRACE_LAYER", "0")
        )
        trace_token_index = int(
            os.environ.get("MOLT_ALIGNMENT_CUDAGRAPH_TRACE_TOKEN_INDEX", "0")
        )
        if not 0 <= trace_layer_index < len(layers):
            raise ValueError("MOLT_ALIGNMENT_CUDAGRAPH_TRACE_LAYER is out of range")
        if trace_token_index < 0:
            raise ValueError("MOLT_ALIGNMENT_CUDAGRAPH_TRACE_TOKEN_INDEX must be >= 0")
        trace = torch.zeros(
            (len(layers), backbone.config.hidden_size),
            dtype=backbone.embed_tokens.weight.dtype,
            device=backbone.embed_tokens.weight.device,
        )
        trace_stages = torch.zeros(
            (7, backbone.config.hidden_size), dtype=trace.dtype, device=trace.device
        )
        trace_embedding = torch.zeros(
            2, trace.shape[1], dtype=trace.dtype, device=trace.device
        )
        trace_metadata = torch.zeros(2, dtype=torch.int64, device=trace.device)
        trace_pending = torch.zeros(1, dtype=torch.bool, device=trace.device)
        trace_unconditional = (
            os.environ.get("MOLT_ALIGNMENT_CUDAGRAPH_TRACE_UNCONDITIONAL") == "1"
        )

        def copy_trace(destination, value):
            if trace_unconditional:
                destination.copy_(value)
            else:
                destination.copy_(torch.where(trace_pending, value, destination))

        backbone._molt_alignment_cudagraph_trace = trace
        backbone._molt_alignment_cudagraph_trace_stages = trace_stages
        backbone._molt_alignment_cudagraph_trace_embedding = trace_embedding
        backbone._molt_alignment_cudagraph_trace_metadata = trace_metadata
        backbone._molt_alignment_cudagraph_trace_pending = trace_pending
        backbone._molt_alignment_cudagraph_trace_layer_index = trace_layer_index
        backbone._molt_alignment_cudagraph_trace_token_index = trace_token_index
        original_backbone_forward = backbone.forward
        original_embed_input_ids = backbone.embed_input_ids

        def traced_embed_input_ids(input_ids):
            hidden_states = original_embed_input_ids(input_ids)
            if input_ids.shape[0] > trace_token_index:
                copy_trace(trace_embedding[0], hidden_states[trace_token_index])
                copy_trace(
                    trace_embedding[1],
                    backbone.embed_tokens.weight[input_ids[trace_token_index]],
                )
            return hidden_states

        def traced_backbone_forward(input_ids, positions, *args, **kwargs):
            if input_ids.shape[0] > trace_token_index:
                copy_trace(
                    trace_metadata[0:1],
                    input_ids[trace_token_index : trace_token_index + 1],
                )
                copy_trace(
                    trace_metadata[1:2],
                    positions[trace_token_index : trace_token_index + 1],
                )
            return original_backbone_forward(input_ids, positions, *args, **kwargs)

        backbone.embed_input_ids = traced_embed_input_ids
        backbone.forward = traced_backbone_forward

    def make_layer_forward(layer, layer_index):
        def forward(positions, hidden_states, residual):
            if (
                trace_stages is not None
                and layer_index == trace_layer_index
                and hidden_states.shape[0] > trace_token_index
            ):
                copy_trace(trace_stages[0], hidden_states[trace_token_index])
            if residual is None:
                residual = hidden_states
            else:
                residual = hidden_states + residual
            hidden_states = layer.input_layernorm(residual)
            if (
                trace_stages is not None
                and layer_index == trace_layer_index
                and hidden_states.shape[0] > trace_token_index
            ):
                copy_trace(trace_stages[1], hidden_states[trace_token_index])
            hidden_states = layer.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
            if (
                trace_stages is not None
                and layer_index == trace_layer_index
                and hidden_states.shape[0] > trace_token_index
            ):
                copy_trace(trace_stages[2], hidden_states[trace_token_index])
            residual = hidden_states + residual
            if (
                trace_stages is not None
                and layer_index == trace_layer_index
                and hidden_states.shape[0] > trace_token_index
            ):
                copy_trace(trace_stages[3], residual[trace_token_index])
            hidden_states = layer.post_attention_layernorm(residual)
            if (
                trace_stages is not None
                and layer_index == trace_layer_index
                and hidden_states.shape[0] > trace_token_index
            ):
                copy_trace(trace_stages[4], hidden_states[trace_token_index])
            hidden_states = layer.mlp(hidden_states)
            if (
                trace_stages is not None
                and layer_index == trace_layer_index
                and hidden_states.shape[0] > trace_token_index
            ):
                copy_trace(trace_stages[5], hidden_states[trace_token_index])
            if trace is not None and hidden_states.shape[0] > trace_token_index:
                copy_trace(
                    trace[layer_index],
                    hidden_states[trace_token_index] + residual[trace_token_index],
                )
                if trace_stages is not None and layer_index == trace_layer_index:
                    copy_trace(
                        trace_stages[6],
                        hidden_states[trace_token_index]
                        + residual[trace_token_index],
                    )
                if layer_index == len(layers) - 1:
                    trace_pending.zero_()
            return hidden_states, residual

        return forward

    for layer_index, layer in enumerate(layers):
        if not all(
            hasattr(layer, name)
            for name in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
        ):
            raise ValueError("AutoModel residual-norm matching supports Qwen decoder layers only.")
        layer.forward = make_layer_forward(layer, layer_index)

    original_final_norm = backbone.norm.forward

    def final_norm(hidden_states, residual=None):
        if residual is None:
            return original_final_norm(hidden_states)
        residual = hidden_states + residual
        return original_final_norm(residual), residual

    backbone.norm.forward = final_norm
    backbone._molt_automodel_residual_norm = True
    print(
        f"[Alignment] enabled AutoModel residual-norm convention in {len(layers)} vLLM layers.",
        flush=True,
    )


def _install_precompile_residual_norm_patch():
    """Apply the opt-in residual convention before vLLM's first model trace."""
    import os

    if os.environ.get("MOLT_VLLM_PRECOMPILE_RESIDUAL_NORM") != "1":
        return

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_molt_precompile_residual_norm", False):
        return

    original_load_model = GPUModelRunner.load_model

    def load_model(self, *args, **kwargs):
        result = original_load_model(self, *args, **kwargs)
        _enable_automodel_residual_norm(self.model)
        return result

    GPUModelRunner.load_model = load_model
    GPUModelRunner._molt_precompile_residual_norm = True


_install_precompile_residual_norm_patch()


def _install_class_residual_norm_patch():
    """Install the Qwen residual convention before vLLM constructs the model."""
    import os

    if os.environ.get("MOLT_VLLM_CLASS_RESIDUAL_NORM") != "1":
        return

    from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer

    if getattr(Qwen2DecoderLayer, "_molt_automodel_residual_norm", False):
        return

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
        else:
            residual = hidden_states + residual
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        residual = hidden_states + residual
        hidden_states = self.post_attention_layernorm(residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    Qwen2DecoderLayer.forward = forward
    Qwen2DecoderLayer._molt_automodel_residual_norm = True


_install_class_residual_norm_patch()


class WorkerWrap:
    def _enable_automodel_residual_norm(self):
        _enable_automodel_residual_norm(self.model_runner.model)

    def dump_cudagraph_alignment_trace(self, path):
        """Persist the small graph-safe layer trace after a replay completes."""
        import os

        import torch

        model = self.model_runner.model
        backbone = getattr(model, "model", model)
        trace = getattr(backbone, "_molt_alignment_cudagraph_trace", None)
        if trace is None:
            raise RuntimeError("CUDA-graph trace was not enabled before model compilation")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "layers": trace.detach().cpu(),
                "stages": backbone._molt_alignment_cudagraph_trace_stages.detach().cpu(),
                "embedding": backbone._molt_alignment_cudagraph_trace_embedding.detach().cpu(),
                "metadata": backbone._molt_alignment_cudagraph_trace_metadata.detach().cpu(),
                "layer_index": backbone._molt_alignment_cudagraph_trace_layer_index,
                "token_index": backbone._molt_alignment_cudagraph_trace_token_index,
            },
            path,
        )
        return {"layers": trace.shape[0], "hidden_size": trace.shape[1], "path": path}

    def dump_embedding_weight_row(self, token_id, path):
        """Persist one embedding row outside the compiled model forward."""
        import os

        import torch

        model = self.model_runner.model
        backbone = getattr(model, "model", model)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(backbone.embed_tokens.weight[token_id].detach().cpu(), path)
        return {"token_id": token_id, "path": path}

    def reset_cudagraph_alignment_trace(self):
        """Arm the graph-safe trace for exactly the next model forward."""
        model = self.model_runner.model
        backbone = getattr(model, "model", model)
        pending = getattr(backbone, "_molt_alignment_cudagraph_trace_pending", None)
        if pending is None:
            raise RuntimeError("CUDA-graph trace was not enabled before model compilation")
        pending.fill_(True)
        return True

    def arm_alignment_trace(self, trace_dir):
        """Save one prefill's sequence-start hidden states for parity debugging."""
        import os

        import torch

        model = self.model_runner.model
        backbone = getattr(model, "model", model)
        layers = list(getattr(backbone, "layers", ()))
        if not layers:
            raise RuntimeError("alignment trace requires a decoder model with model.layers")
        trace_layer = int(os.environ.get("MOLT_ALIGNMENT_TRACE_LAYER", "0"))
        trace_offset = int(os.environ.get("MOLT_ALIGNMENT_TRACE_TOKEN_OFFSET", "0"))
        prefix_length = max(
            64,
            trace_offset + 1,
            int(os.environ.get("MOLT_ALIGNMENT_TRACE_PREFIX_LENGTH", "0")),
        )
        trace_logits = os.environ.get("MOLT_ALIGNMENT_TRACE_LOGITS") == "1"
        trace_decode_steps = int(os.environ.get("MOLT_ALIGNMENT_TRACE_DECODE_STEPS", "0"))
        full_prefix_stages = {
            name
            for name in os.environ.get(
                "MOLT_ALIGNMENT_TRACE_FULL_PREFIX_STAGES", ""
            ).split(",")
            if name
        }
        if not 0 <= trace_layer < len(layers):
            raise ValueError(f"invalid MOLT_ALIGNMENT_TRACE_LAYER={trace_layer}")
        if trace_offset < 0:
            raise ValueError(f"invalid MOLT_ALIGNMENT_TRACE_TOKEN_OFFSET={trace_offset}")
        if trace_decode_steps < 0:
            raise ValueError(f"invalid MOLT_ALIGNMENT_TRACE_DECODE_STEPS={trace_decode_steps}")

        os.makedirs(trace_dir, exist_ok=True)
        state = {
            "input_ids": None,
            "layers": {},
            "stages": {},
            "handles": [],
            "prefill_captured": False,
            "reported_inputs": False,
        }

        def finalize_trace():
            if state.get("saved") or state["input_ids"] is None:
                return
            starts = state["prefix_starts"]
            input_ids = state["input_ids"]
            boundaries = torch.cat((starts[1:], starts.new_tensor([input_ids.numel()])))
            prefixes = [
                input_ids[start : min(int(end), int(start) + prefix_length)].detach().cpu()
                for start, end in zip(starts.tolist(), boundaries.tolist())
            ]
            torch.save(
                {"prefixes": prefixes, "layers": state["layers"], "stages": state["stages"]},
                os.path.join(trace_dir, f"vllm-{os.getpid()}.pt"),
            )
            state["saved"] = True
            for handle in state["handles"]:
                handle.remove()
            state["handles"].clear()
            backbone.forward = state["original_forward"]

        def capture_value(name, value):
            if state["input_ids"] is None or state["prefill_captured"]:
                return
            if isinstance(value, tuple):
                value = value[0]
            if not isinstance(value, torch.Tensor):
                return
            if name in full_prefix_stages:
                prefixes = [
                    value[start : start + prefix_length]
                    for start in state["prefix_starts"].tolist()
                ]
                state["stages"][name] = torch.stack(prefixes).detach().cpu()
            else:
                state["stages"][name] = value.index_select(0, state["starts"]).detach().cpu()

        def capture_attention_value(name, value):
            if (
                state["input_ids"] is None
                or state["prefill_captured"]
                or not isinstance(value, torch.Tensor)
            ):
                return
            if name in full_prefix_stages:
                prefixes = [
                    value[start : start + prefix_length].flatten(1)
                    for start in state["prefix_starts"].tolist()
                ]
                state["stages"][name] = torch.stack(prefixes).detach().cpu()
            else:
                state["stages"][name] = (
                    value.index_select(0, state["starts"]).flatten(1).detach().cpu()
                )

        def capture_inputs(args, kwargs):
            if not state["reported_inputs"]:
                state["reported_inputs"] = True
                arg_types = [None if value is None else type(value).__name__ for value in args]
                arg_shapes = [None if value is None else getattr(value, "shape", None) for value in args]
                print(
                    f"[alignment_trace] vLLM backbone forward types={arg_types} shapes={arg_shapes}",
                    flush=True,
                )
            if state["input_ids"] is not None:
                return
            input_ids = args[0] if args else kwargs.get("input_ids")
            positions = args[1] if len(args) > 1 else kwargs.get("positions")
            if input_ids is None or positions is None:
                return
            prefix_starts = positions.eq(0).nonzero(as_tuple=False).flatten()
            boundaries = torch.cat(
                (prefix_starts[1:], prefix_starts.new_tensor([input_ids.numel()]))
            )
            starts = prefix_starts + trace_offset
            valid = starts < boundaries
            starts = starts[valid]
            prefix_starts = prefix_starts[valid]
            if not len(starts):
                return
            # The V1 model runner reuses this input buffer for subsequent decode
            # steps. The trace finalizes after those steps, so retain a snapshot
            # rather than a view into mutable runner storage.
            state["input_ids"] = input_ids.detach().clone()
            state["starts"] = starts.clone()
            state["prefix_starts"] = prefix_starts.clone()

        def capture_layer(index):
            def hook(_, __, output):
                if state["input_ids"] is None or state["prefill_captured"]:
                    return
                hidden, residual = output
                state["layers"][index] = (
                    (hidden + residual).index_select(0, state["starts"]).detach().cpu()
                )
                if index != len(layers) - 1 or trace_logits:
                    return
                finalize_trace()

            return hook

        state["original_forward"] = backbone.forward

        def traced_forward(*args, **kwargs):
            capture_inputs(args, kwargs)
            return state["original_forward"](*args, **kwargs)

        backbone.forward = traced_forward
        state["handles"].extend(layer.register_forward_hook(capture_layer(i)) for i, layer in enumerate(layers))
        traced_layer = layers[trace_layer]
        state["handles"].append(
            traced_layer.register_forward_pre_hook(lambda _, args: capture_value("input", args[1]))
        )
        state["handles"].append(
            traced_layer.input_layernorm.register_forward_pre_hook(
                lambda _, args: capture_value("pre_norm1", args[0])
            )
        )
        for name, module in (
            ("norm1", traced_layer.input_layernorm),
            ("attn", traced_layer.self_attn),
            ("norm2", traced_layer.post_attention_layernorm),
            ("mlp", traced_layer.mlp),
        ):
            state["handles"].append(module.register_forward_hook(lambda _, __, output, name=name: capture_value(name, output)))

        def capture_qkv(_, __, output):
            qkv = output[0] if isinstance(output, tuple) else output
            q_size = traced_layer.self_attn.q_size
            kv_size = traced_layer.self_attn.kv_size
            capture_value("q_raw", qkv[..., :q_size])
            capture_value("k_raw", qkv[..., q_size : q_size + kv_size])
            capture_value("v_raw", qkv[..., q_size + kv_size :])

        state["handles"].append(traced_layer.self_attn.qkv_proj.register_forward_hook(capture_qkv))
        def capture_post_rope(_, args):
            capture_attention_value("q_post_rope", args[0])
            capture_attention_value("k_post_rope", args[1])

        state["handles"].append(traced_layer.self_attn.attn.register_forward_pre_hook(capture_post_rope))

        def capture_rope_cache(_, args):
            if state["input_ids"] is None or state["prefill_captured"]:
                return
            positions = args[0]
            cos_sin = traced_layer.self_attn.rotary_emb.cos_sin_cache.index_select(
                0, positions.index_select(0, state["starts"])
            )
            rotary_dim = cos_sin.shape[-1] // 2
            state["stages"]["rope_cos"] = cos_sin[..., :rotary_dim].detach().cpu()
            state["stages"]["rope_sin"] = cos_sin[..., rotary_dim:].detach().cpu()

        state["handles"].append(
            traced_layer.self_attn.rotary_emb.register_forward_pre_hook(capture_rope_cache)
        )
        state["handles"].append(
            traced_layer.self_attn.o_proj.register_forward_pre_hook(
                lambda _, args: capture_value("attn_pre_o_proj", args[0])
            )
        )
        if trace_logits:
            def capture_final_norm(_, __, output):
                if trace_decode_steps:
                    if isinstance(output, tuple):
                        output = output[0]
                    if output.ndim != 2:
                        raise RuntimeError(
                            "decode trace requires vLLM's flattened hidden-state layout."
                        )
                    if "final_norm_prefill" not in state["stages"]:
                        state["stages"]["final_norm_prefill"] = (
                            output.index_select(0, state["starts"]).detach().cpu()
                        )
                        state["prefill_captured"] = True
                    chunks = state["stages"].setdefault("final_norm_sequence", [])
                    chunks.append(output[-1:].detach().cpu())
                    if len(chunks) == trace_decode_steps:
                        state["stages"]["final_norm_sequence"] = torch.cat(chunks)
                        finalize_trace()
                    return
                capture_value("final_norm", output)
                finalize_trace()

            state["handles"].append(backbone.norm.register_forward_hook(capture_final_norm))

        def capture_gate_up(_, __, output):
            gate_up = output[0] if isinstance(output, tuple) else output
            gate_size = gate_up.shape[-1] // 2
            capture_value("gate_raw", gate_up[..., :gate_size])
            capture_value("up_raw", gate_up[..., gate_size:])

        state["handles"].append(traced_layer.mlp.gate_up_proj.register_forward_hook(capture_gate_up))
        print(
            f"[alignment_trace] armed vLLM worker pid={os.getpid()} "
            f"model={type(model).__name__} backbone={type(backbone).__name__} "
            f"layers={len(layers)} trace_layer={trace_layer} offset={trace_offset} path={trace_dir}",
            flush=True,
        )

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl"):
        """Init torch process group for model weights update"""
        import torch

        from molt.utils.distributed_util import stateless_init_process_group

        self._enable_automodel_residual_norm()
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
