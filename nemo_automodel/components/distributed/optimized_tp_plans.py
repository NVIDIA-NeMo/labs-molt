# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Model-specific parallel plans for tensor parallelism.

This module contains optimized tensor parallel plans for the Qwen and Muse Glimmer
model families (HF and NeMo-native classes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Union, cast

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.parallel_styles import ReplicatedWithGradAllReduce

# Annotation-only imports: importing transformers model modules at module scope
# drags in the whole model zoo, so PARALLELIZE_FUNCTIONS keys are spelled as literals.
if TYPE_CHECKING:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


class SequenceParallelAllGatherActivation(SequenceParallel):
    """SequenceParallel that all-gathers activations for sequence parallelism."""

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        """Prepare outputs by redistributing sharded DTensors to replicated placement."""
        # If output is a DTensor with Shard placement, redistribute to Replicate
        if isinstance(outputs, DTensor):
            if any(isinstance(p, Shard) for p in outputs.placements):
                # Redistribute to replicated placement (performs all-gather)
                outputs = outputs.redistribute(device_mesh=device_mesh, placements=[Replicate()])
        else:
            raise ValueError(f"Expected output to be a DTensor, but got {type(outputs)}")

        # Call the parent's prepare_output_fn to handle use_local_output
        return SequenceParallel._prepare_output_fn(use_local_output, mod, outputs, device_mesh)


class VocabParallelEmbedding(RowwiseParallel):
    """``RowwiseParallel`` for ``nn.Embedding`` with a ``MaskPartial`` mask-buffer fixup.

    Some PyTorch versions have a DTensor bug where the ``MaskPartial``
    placement's ``mask_buffer`` is not populated during the embedding
    dispatch, leading to::

        AssertionError: assert self.mask_buffer.data is not None

    This subclass works around the issue by:

    1. Saving the *original* (un-adjusted) ``input_ids`` in a pre-hook.
    2. Recomputing and populating the ``mask_buffer`` in the post-hook
       when the DTensor dispatch failed to do so.

    In PyTorch versions where the dispatch works correctly the mask buffer
    is already populated and the fixup is a no-op.
    """

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        # Save the original input_ids (before DTensor index-adjustment)
        # so we can recompute the mask in the output hook if needed.
        input_tensor = inputs[0]
        if isinstance(input_tensor, DTensor):
            mod._vocab_parallel_saved_ids = input_tensor.to_local().clone()
        else:
            mod._vocab_parallel_saved_ids = input_tensor.clone()

        return RowwiseParallel._prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        saved_ids = getattr(mod, "_vocab_parallel_saved_ids", None)
        if saved_ids is not None:
            delattr(mod, "_vocab_parallel_saved_ids")

        # If the output is a DTensor whose MaskPartial placement has an
        # empty mask_buffer, compute and materialise the mask so that the
        # subsequent ``_reduce_value`` / ``_reduce_shard_value`` succeeds.
        if isinstance(outputs, DTensor) and saved_ids is not None:
            placement = outputs.placements[0]
            mb = getattr(placement, "mask_buffer", None)
            if mb is not None and getattr(mb, "data", ...) is None:
                vocab_size = getattr(mod, "num_embeddings", None) or mod.weight.shape[0]
                tp_size = device_mesh.size()
                rank = device_mesh.get_local_rank()

                chunk = vocab_size // tp_size
                rem = vocab_size % tp_size
                if rank < rem:
                    local_size = chunk + 1
                    local_off = rank * (chunk + 1)
                else:
                    local_size = chunk
                    local_off = rem * (chunk + 1) + (rank - rem) * chunk

                mask = (saved_ids < local_off) | (saved_ids >= local_off + local_size)
                mb.materialize_mask(mask)

        return RowwiseParallel._prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh)


def _parallelize_muse_glimmer(
    model,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """TP plan for the native MuseGlimmer dense VLM.

    The vision tower stays replicated. The language backbone and vocabulary
    matrices contain nearly all trainable parameters and are tensor-sharded.
    MuseGlimmer has two KV heads, so the model strategy limits this complete
    Q/K/V-sharding plan to TP1 or TP2.
    """
    if sequence_parallel:
        import warnings

        warnings.warn(
            "sequence_parallel=True is not yet supported for MuseGlimmer and will be ignored.",
            stacklevel=2,
        )

    plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.output_gate_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    return cast(dict[str, ParallelStyle], plan)


def _parallelize_qwen(
    model: Union[Qwen2ForCausalLM, Qwen3ForCausalLM],
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelizes a Qwen2/Qwen3 causal LM across data and tensor parallel dimensions."""

    if sequence_parallel:
        base_model_tp_plan = {
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1),
                use_local_output=False,
            ),
            "model.embed_tokens": VocabParallelEmbedding(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
                # Keep DTensor outputs so HF modeling code (e.g. cache_position) can
                # observe the *global* sequence length via DTensor.shape.
                use_local_output=False,
            ),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallelAllGatherActivation(),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.q_norm": ReplicatedWithGradAllReduce(),
            "model.layers.*.self_attn.k_norm": ReplicatedWithGradAllReduce(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
            # Rowwise projections reduce-scatter back to sequence-sharded activations.
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
            # Qwen3 q_norm/k_norm operate independently on head-sharded Q/K.
            # Their parameters stay replicated, while partial-head gradients sum.
            "model.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        }

    else:
        base_model_tp_plan = {
            "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
            "model.embed_tokens": VocabParallelEmbedding(
                input_layouts=Replicate(),
            ),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.q_norm": ReplicatedWithGradAllReduce(),
            "model.layers.*.self_attn.k_norm": ReplicatedWithGradAllReduce(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(),
        }

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


# Named TP plan for use with tp_shard_plan="llama_nemotron_super_tp_plan" in parallelizer
LLAMA_NEMOTRON_SUPER_TP_PLAN_NAME = "llama_nemotron_super_tp_plan"


def _get_class_qualname(cls: type) -> str:
    """Return the fully qualified name of a class as ``module.qualname``.

    Used as a stable dict key for PARALLELIZE_FUNCTIONS instead of the class
    object itself.

    When NeMo-RL uses automodel, ``force_hf=True`` is auto-set for models
    (e.g. ``LlamaForCausalLM``) whose adapter does not implement
    ``convert_single_tensor_to_hf``. This causes ``_get_mixin_wrapped_class``
    in ``model_init.py`` to create a new class via ``type(...)`` that wraps
    the original with ``HFCheckpointingMixin``. The wrapper copies
    ``__module__`` and ``__qualname__`` from the original but is a **different
    Python object**, so ``type(model) in PARALLELIZE_FUNCTIONS`` (identity
    check) returns ``False`` and the default plan is used instead of the
    optimized one.

    String comparison on ``module.qualname`` survives this wrapping and
    correctly identifies the model class.
    """
    return f"{cls.__module__}.{cls.__qualname__}"


def _parallelize_qwen3_5_vlm(
    model,
    sequence_parallel: bool = False,
) -> Dict[str, ParallelStyle]:
    """Parallelize Qwen3.5 VLM by reusing transformers' base_model_tp_plan.

    Qwen3.5 has mixed attention: full self_attn (every 4th layer) + linear_attn
    (GatedDeltaNet). The transformers-provided base_model_tp_plan covers only
    self_attn + MLP — linear_attn is not TP-shardable with stock kernels.
    """
    from nemo_automodel.components.distributed.parallelizer import get_hf_tp_shard_plan

    return get_hf_tp_shard_plan(model)


# Keyed by qualified class name — see _get_class_qualname for why.
PARALLELIZE_FUNCTIONS: Dict[str, Callable[..., Dict[str, ParallelStyle]]] = {
    "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM": _parallelize_qwen,
    "transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM": _parallelize_qwen,
    # Hard-coded qualname to avoid eagerly importing transformers.models.qwen3_5.
    "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForConditionalGeneration": _parallelize_qwen3_5_vlm,
    # NeMo-native Qwen3.5 dense (custom-model port): same plan — shard self_attn +
    # MLP, leave the GatedDeltaNet (linear_attn) replicated.
    "nemo_automodel.components.models.qwen3_5.model.Qwen3_5ForConditionalGeneration": _parallelize_qwen3_5_vlm,
    "nemo_automodel.components.models.qwen3_5.model.Qwen3_5ForCausalLM": _parallelize_qwen3_5_vlm,
    # Register native Qwen classes without importing their checkpoint adapters into the distributed component.
    "nemo_automodel.components.models.qwen2.model.Qwen2ForCausalLM": _parallelize_qwen,
    "nemo_automodel.components.models.qwen3.model.Qwen3ForCausalLM": _parallelize_qwen,
    "nemo_automodel.components.models.muse_glimmer.model.MuseGlimmerForConditionalGeneration": _parallelize_muse_glimmer,
}
