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

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA, GroupedExpertsLoRA
from nemo_automodel.components._peft.module_matcher import ModuleMatcher
from nemo_automodel.components.moe.layers import GroupedExperts, GroupedExpertsDeepEP, GroupedExpertsTE
from nemo_automodel.shared.import_utils import safe_import_te
from nemo_automodel.shared.tp_linear import tp_linear_forward
from nemo_automodel.shared.utils import dtype_from_str

HAS_TE, transformer_engine = safe_import_te()

logger = logging.getLogger(__name__)


@dataclass
class PeftConfig:
    target_modules: list = field(default_factory=list)
    exclude_modules: list = field(default_factory=list)
    match_all_linear: bool = False
    dim: int = 8
    alpha: int = 32
    # Note: we currently support DoRA for nn.Linear only.
    use_dora: bool = False
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "post"
    lora_A_init: str = "xavier"
    lora_dtype: torch.dtype | None = None
    use_memory_efficient_lora: bool = True
    use_triton: bool = False
    moe_rank_scaling: bool = False

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        return cls(
            target_modules=d.get("target_modules", []),
            exclude_modules=d.get("exclude_modules", []),
            match_all_linear=d.get("match_all_linear", False),
            dim=d.get("dim", 8),
            alpha=d.get("alpha", 32),
            use_dora=d.get("use_dora", False),
            dropout=d.get("dropout", 0.0),
            dropout_position=d.get("dropout_position", "post"),
            lora_A_init=d.get("lora_A_init", "xavier"),
            lora_dtype=d.get("lora_dtype", None),
            use_memory_efficient_lora=d.get("use_memory_efficient_lora", True),
            use_triton=d.get("use_triton", False),
            moe_rank_scaling=d.get("moe_rank_scaling", False),
        )


class LinearLoRA(nn.Linear):
    """
    Linear + LoRA, maintains ckpts structure (i.e. Linear's weight/bias remain at the same FQN).

    The _init_wrapper and _forward methods provide the LoRA functionality. We want to be able to
    use those inside LinearLoRA but also for monkey-patching modules, without repeating the
    same code -> therefore those are decorated with @staticmethod.
    """

    def __init__(
        self,
        orig_linear,
        dim=8,
        alpha=32,
        use_dora: bool = False,
        dropout=0.0,
        dropout_position="post",
        lora_A_init_method="xavier",
        lora_dtype=None,
        use_memory_efficient_lora=True,
    ):
        """
        LinearLora constructor.

        Args:
            orig_linear (nn.Module): the linear module to augment.
            dim (int): lora's dim in_features -> dim -> out_features.
            alpha (int): lora's scaling alpha.
            dropout (float): dropout prob (default: 0.0).
            dropout_position (str): where to apply dropout rel. to lora (choices= ['pre', 'post'], default=post)
            lora_A_init_method (str): init method for lora_A (choices= ['xavier', 'uniform'])
            lora_dtype (torch.dtype): weight's dtype, by default will use orig_linear's but if they
            are quantized weights (e.g. 4bit) needs to be specified explicitly.
        """
        assert isinstance(orig_linear, nn.Linear)
        super(LinearLoRA, self).__init__(
            in_features=orig_linear.in_features,
            out_features=orig_linear.out_features,
            bias=orig_linear.bias is not None,
            device=orig_linear.weight.device,
            dtype=orig_linear.weight.dtype,
        )
        # copy weights
        self.weight.data.copy_(orig_linear.weight.data)
        if orig_linear.bias is not None:
            self.bias.data.copy_(orig_linear.bias.data)
        # initialize the adapte
        LinearLoRA._init_adapter(
            self,
            dim=dim,
            alpha=alpha,
            use_dora=use_dora,
            dropout=dropout,
            dropout_position=dropout_position,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
            use_memory_efficient_lora=use_memory_efficient_lora,
        )

    @torch.no_grad
    def init_lora_weights(self, init_method: str):
        """
        Initialize the LoRA weights.

        Args:
            init_method (str): Method to initialize the LoRA weights.
        """
        if init_method == "xavier":
            nn.init.xavier_normal_(self.lora_A.weight.data)
        else:
            nn.init.kaiming_uniform_(self.lora_A.weight.data, a=math.sqrt(5))
        self.lora_B.weight.data.fill_(0)

    @torch.no_grad
    @staticmethod
    def _init_adapter(
        obj,
        dim=8,
        alpha=32,
        use_dora: bool = False,
        dropout=0.0,
        dropout_position="post",
        lora_A_init_method="xavier",
        lora_dtype=None,
        use_memory_efficient_lora=True,
    ):
        """
        Adds LoRA weights to obj. Obj is either a LinearLoRA or an nn.Module (when monkey-patching).

        Args:
            obj (LinearLoRA | nn.Module): input module to adapt.
            dim (int): lora's dim in_features -> dim -> out_features.
            alpha (int): lora's scaling alpha.
            dropout (float): dropout prob (default: 0.0).
            dropout_position (str): where to apply dropout rel. to lora (choices= ['pre', 'post'], default=post)
            lora_A_init_method (str): init method for lora_A (choices= ['xavier', 'uniform'])
            lora_dtype (torch.dtype): weight's dtype, by default will use orig_linear's but if they
            are quantized weights (e.g. 4bit) needs to be specified explicitly.
        """
        obj.dim = dim
        obj.scale = alpha / dim
        obj.use_dora = bool(use_dora)
        obj.use_memory_efficient_lora = bool(use_memory_efficient_lora)

        # Freezer
        device = obj.weight.device
        obj.weight.requires_grad = False
        if obj.bias is not None:
            obj.bias.requires_grad = False

        in_features = obj.in_features
        out_features = obj.out_features
        if isinstance(lora_dtype, str):
            lora_dtype = dtype_from_str(lora_dtype)
        assert lora_dtype is None or isinstance(lora_dtype, torch.dtype)
        dtype = lora_dtype or obj.weight.dtype

        if HAS_TE and isinstance(obj, transformer_engine.pytorch.Linear):
            obj.lora_A = transformer_engine.pytorch.Linear(
                in_features=in_features, out_features=dim, bias=False, device=device, params_dtype=dtype
            )
            obj.lora_B = transformer_engine.pytorch.Linear(
                in_features=dim, out_features=out_features, bias=False, device=device, params_dtype=dtype
            )
        else:
            obj.lora_A = nn.Linear(in_features, dim, bias=False, dtype=dtype, device=device)
            obj.lora_B = nn.Linear(dim, out_features, bias=False, dtype=dtype, device=device)
        LinearLoRA.init_lora_weights(obj, lora_A_init_method)
        obj.dropout_p = dropout
        assert dropout_position in ["pre", "post"], ("dropout position can only be pre/post", dropout_position)
        obj.dropout_position = dropout_position

        if obj.use_dora:
            # initialize DoRA magnitude vector to ||W|| (row-wise L2 norm).
            with torch.no_grad():
                weight_norm = torch.linalg.norm(obj.weight.data, dim=1).to(dtype=dtype, device=device)
            obj.lora_magnitude = nn.Parameter(weight_norm, requires_grad=True)

    def _dora_weight_norm(self) -> torch.Tensor:
        """
        Compute the detached weight norm used by DoRA.
        """
        # ΔW = B @ A, shapes: [out, dim] @ [dim, in] => [out, in]
        delta_w = (self.lora_B.weight @ self.lora_A.weight).detach().to(self.weight.dtype)
        weight = self.weight.to(self.weight.dtype)
        weight_norm = torch.linalg.norm(weight + self.scale * delta_w, dim=1).to(weight.dtype)
        return weight_norm.detach()

    def materialize_effective_weight(self) -> torch.Tensor:
        """Return the differentiable dense weight represented by this LoRA layer.

        Returns:
            Tensor of shape [out_features, in_features] containing the frozen base
            weight plus the scaled LoRA update.

        Raises:
            RuntimeError: If training-time dropout makes one fixed effective weight
                unable to represent the layer's stochastic forward pass.
            NotImplementedError: If DoRA, a delegated linear implementation, or an
                unsupported quantized or non-strided weight layout is active.
        """
        if self.training and self.dropout_p > 0.0:
            raise RuntimeError("materialize_effective_weight does not support active LoRA training dropout")
        if self.use_dora:
            raise NotImplementedError("materialize_effective_weight does not support DoRA")
        if getattr(self, "super_fwd", None) is not None:
            raise NotImplementedError(
                "materialize_effective_weight supports only ordinary torch linear weights, not delegated "
                "linear implementations"
            )
        if self.weight.layout != torch.strided or self.weight.is_quantized:
            raise NotImplementedError(
                "materialize_effective_weight supports only dense, strided, non-quantized linear weights"
            )
        return self.weight + self.scale * (self.lora_B.weight @ self.lora_A.weight)

    def _should_use_memory_efficient_lora(self, x: torch.Tensor) -> bool:
        """Return whether this LoRA branch can use the custom autograd path."""
        if not getattr(self, "use_memory_efficient_lora", False):
            return False
        if isinstance(x, DTensor):
            return False
        if isinstance(getattr(self.lora_A, "weight", None), DTensor):
            return False
        if isinstance(getattr(self.lora_B, "weight", None), DTensor):
            return False
        if torch.compiler.is_compiling():
            return False
        if HAS_TE and isinstance(getattr(self, "lora_A", None), transformer_engine.pytorch.Linear):
            return False
        return True

    def forward(self, x):
        """
        Forward pass through the original linear layer augmented with the LoRA pathway.

        Applies LoRA either before or after the dropout, depending on the configuration.
        The result of the original linear transformation is combined with the LoRA output.

        Args:
            x (Tensor): Input activations of shape ``[B, S, in_features]``
                (``B`` = batch, ``S`` = sequence) or ``[N, in_features]``
                (``N`` = flattened tokens).  May be a DTensor: a 3-D DTensor
                sharded on dim 0 or 1 (e.g. ``Shard(1)`` from sequence
                parallelism) routes the base projection through ``torch.bmm``;
                replicated or last-dimension-sharded inputs (``Shard(2)`` or
                ``Shard(-1)``) take ``F.linear``, which under async-TP tracing
                is the fusable native linear graph.

        Returns:
            Tensor: Output of shape ``[..., out_features]`` with the same
            leading dimensions as ``x``; a DTensor if ``x`` and the weights
            are DTensors.
        """
        # pylint: disable=C0115,C0116
        # If LinearLoRA is used to monkey-patch a nn.Linear module, we want to use nn.Linear's
        # forward in the case where it uses quantized weights. We store a reference to nn.Linear's
        # forward in `super_fwd` attribute. If the attribute does not exist we do the usual linear.
        if (fwd := getattr(self, "super_fwd", None)) is not None:
            assert fwd != self.forward
            res = fwd(x)
        else:
            # TE Linear can expose an empty .bias tensor (numel()==0) when bias=False; treat as no bias.
            bias = self.bias
            if bias is not None and bias.numel() == 0:
                bias = None
            res = tp_linear_forward(x, self.weight, bias, mm_for_2d_compile=False)

        if not self.use_dora:
            if self.dropout_position == "pre":
                x = F.dropout(x, p=self.dropout_p, training=self.training)

            # Apply scale before lora_B to keep lora_res as a Partial tensor.
            # This allows both res and lora_res to remain Partial, so only one reduce-scatter is needed after addition.
            # Multiplying after lora_B would convert Partial to Replicate, causing an extra reduce-scatter operation.
            use_memory_efficient_lora = self._should_use_memory_efficient_lora(x)
            if use_memory_efficient_lora:
                if self.dropout_position == "pre" or not self.training or self.dropout_p == 0.0:
                    return apply_memory_efficient_lora(
                        x, self.lora_A.weight, self.lora_B.weight, self.scale, False, res
                    )
                lora_res = apply_memory_efficient_lora(x, self.lora_A.weight, self.lora_B.weight, self.scale, False)
            else:
                lora_res = self.lora_B(self.lora_A(x) * self.scale)
            if self.dropout_position == "post":
                lora_res = F.dropout(lora_res, p=self.dropout_p, training=self.training)
            if use_memory_efficient_lora:
                return lora_res.add_(res)
            return res + lora_res

        if getattr(self, "lora_magnitude", None) is None:
            raise RuntimeError("use_dora=True but lora_magnitude was not initialized")

        if self.dropout_position == "pre" and self.training and self.dropout_p > 0.0:
            x_lora = F.dropout(x, p=self.dropout_p, training=True)
            base_result = None
        else:
            x_lora = x
            base_result = res

        lora_result = self.lora_B(self.lora_A(x_lora))
        if self.dropout_position == "post":
            lora_result = F.dropout(lora_result, p=self.dropout_p, training=self.training)

        # Compute DoRA scaling factor.
        weight_norm = self._dora_weight_norm()
        mag = self.lora_magnitude.to(x.dtype)
        weight_norm = weight_norm.to(x.dtype)

        # Broadcast magnitude scaling across batch/sequence dimensions.
        mag_norm_scale = mag / weight_norm
        if res.dim() == 3:
            mag_norm_scale = mag_norm_scale.view(1, 1, -1)
        else:
            mag_norm_scale = mag_norm_scale.view(1, -1)

        # HF PEFT subtracts bias from base_result before applying scaling terms.
        if base_result is not None:
            bias = self.bias
            if bias is not None and bias.numel() > 0:
                base_no_bias = base_result - bias
            else:
                base_no_bias = base_result
        else:
            # Recompute base linear output without bias on x_lora (see HF PEFT DoraLinearLayer.forward).
            base_no_bias = F.linear(x_lora, self.weight, None)

        dora_extra = (mag_norm_scale - 1) * base_no_bias + mag_norm_scale * lora_result * self.scale
        return res + dora_extra


def patch_linear_module(
    orig_linear,
    dim=8,
    alpha=32,
    use_dora: bool = False,
    dropout=0.0,
    dropout_position="post",
    lora_A_init_method="xavier",
    lora_dtype=None,
    use_memory_efficient_lora=True,
    use_triton=True,
    layer_name=None,
):
    """
    Monkey-patches a nn.Linear (orig_linear param) to be a LinearLoRA.

    The orig_linear might not contain valid weights, for example, the given orig_linear was
    initialized within a context-manager that uses a "meta" device. Therefore, we cannot copy
    the weight/bias from the orig_linear to the LinearLoRA, since those have not been allocated,

    To circumvent this scenario, LinearLoRA's additional functionality (_init_adapter, _forward)
    is based on static functions, so that we can use them for patching or when allocating a
    new LinearLoRA object.

    Args:
        orig_linear (nn.Linear): the module we add adapter to.
        dim (int, optional): Lora dim. Defaults to 8.
        alpha (int, optional): Lora alpha scale. Defaults to 32.
        dropout (float, optional): dropout prob. Defaults to 0.0.
        dropout_position (str, optional): location to apply dropout wrt lora.
            Defaults to 'post' (choices: 'pre', 'post').
        lora_A_init_method (str, optional): lora_a init method. Defaults to 'xavier'.
        lora_dtype (_type_, optional): Lora weights' dtype. By default will use orig_linear's dtype
            but orig_linear might use non-trainable dtype (e.g., 4bit), in which case the user must
            specify the dtype manually. Defaults to None.
        use_memory_efficient_lora (bool, optional): Use the custom autograd implementation for standard LoRA.
            When Triton is enabled this uses Triton kernels; otherwise it uses PyTorch matmuls. Defaults to True.
        use_triton (bool, optional): By default we use the triton kernel LoRA implementation.

    Returns:
        (nn.Module): the monkey-patched (nn.Linear + LoRA) nn.Module
    """
    linear_types = [nn.Linear]
    if HAS_TE:
        linear_types.append(transformer_engine.pytorch.Linear)
    if not isinstance(orig_linear, tuple(linear_types)):
        raise NotImplementedError("Expected isinstance(orig_linear, nn.Linear)")
    assert not hasattr(orig_linear, "super_fwd"), orig_linear.super_fwd

    if use_dora:
        if HAS_TE and isinstance(orig_linear, transformer_engine.pytorch.Linear):
            raise ValueError("DoRA is not supported for transformer_engine.pytorch.Linear layers.")

    linear_lora_cls = LinearLoRA
    linear_lora_cls._init_adapter(
        orig_linear,
        dim=dim,
        alpha=alpha,
        use_dora=use_dora,
        dropout=dropout,
        dropout_position=dropout_position,
        lora_A_init_method=lora_A_init_method,
        lora_dtype=lora_dtype,
        use_memory_efficient_lora=use_memory_efficient_lora,
    )
    cls = orig_linear.__class__
    new_cls = type("PatchedLinearLoRA", (linear_lora_cls, cls), {})

    if HAS_TE and isinstance(orig_linear, transformer_engine.pytorch.Linear):
        # Delegate base computation to TE's forward so TE kernels (including FP8)
        # are used instead of falling back to F.linear().
        orig_linear.super_fwd = orig_linear.forward

    orig_linear.__class__ = new_cls
    if layer_name is not None:
        orig_linear._layer_name = layer_name
    return orig_linear


def patch_moe_module(
    orig_module,
    dim=8,
    alpha=32,
    lora_A_init_method="xavier",
    lora_dtype=None,
):
    """
    Patches a custom MoE module (GroupedExperts or GroupedExpertsDeepEP) with LoRA.

    Args:
        orig_module (nn.Module): The original MoE module to be patched.
        dim (int, optional): LoRA rank (dimension). Defaults to 8.
        alpha (int, optional): LoRA scaling factor. Defaults to 32.
        lora_A_init_method (str, optional): Initialization method for LoRA A matrix. Defaults to "xavier".
        lora_dtype (torch.dtype or str, optional): Data type for LoRA weights. Defaults to None.

    Returns:
        nn.Module: The LoRA-wrapped MoE module (GroupedExpertsLoRA or GroupedExpertsDeepEPLoRA).
    """
    if isinstance(orig_module, GroupedExpertsTE):
        raise NotImplementedError("LoRA is not supported for Transformer Engine (TE) expert modules.")
    elif isinstance(orig_module, GroupedExpertsDeepEP):
        new_module = GroupedExpertsDeepEPLoRA(
            orig_module,
            lora_dim=dim,
            alpha=alpha,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )
    elif isinstance(orig_module, GroupedExperts):
        new_module = GroupedExpertsLoRA(
            orig_module,
            lora_dim=dim,
            alpha=alpha,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )
    else:
        raise NotImplementedError(f"Unsupported MoE module type: {type(orig_module)}")

    return new_module


# patch a model in-place
def apply_lora_to_linear_modules(
    model: nn.Module,
    peft_config: PeftConfig,
    quantization_config=None,
    skip_freeze: bool = False,
) -> int:
    """
    Replace selected nn.Linear layers with LinearLoRA layers (in-place).

    Args:
        model: The model to apply LoRA to.
        peft_config: PEFT configuration for LoRA parameters.
        quantization_config: Optional separate QLoRA quantization configuration.
        skip_freeze: If True, skip the global parameter freeze (caller will handle it later).

    Returns:
        Number of modules that were modified with LoRA.

    Note:
        target_modules accepts wildcard fragments, e.g. ["q_proj", "k_proj", ".*fc.*"].

        When ``use_memory_efficient_lora`` is enabled, after the linear layers are patched this also
        fuses SiLU-SwiGLU (gate/up/down) and ReLU² (up/down) MLPs whose projections were all
        LoRA-patched: their forward is swapped to a single memory-efficient autograd op (see
        ``install_fused_lora_mlp``) that recomputes the activation in backward. It transparently
        falls back to the per-linear path under tensor/expert parallelism (DTensor), DoRA, or active
        dropout.
    """
    # Freeze base model parameters
    if not skip_freeze:
        for w in model.parameters():
            w.requires_grad_(False)

    is_causal_lm = False
    try:
        if (
            hasattr(model, "config")
            and model.config.architectures is not None
            and len(model.config.architectures) > 0
            and "CausalLM" in model.config.architectures[0]
        ):
            # for example, LlamaForCausalLM
            is_causal_lm = True
    except (AttributeError, TypeError):
        is_causal_lm = False

    matcher = ModuleMatcher(
        peft_config.target_modules, peft_config.exclude_modules, peft_config.match_all_linear, is_causal_lm
    )
    num_modules_matched = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, (GroupedExperts, GroupedExpertsDeepEP, GroupedExpertsTE)):
            if matcher.match(module, name):
                if peft_config.use_dora:
                    raise NotImplementedError("DoRA is not supported for MoE expert modules in Automodel yet.")
                num_modules_matched += 1
                lora_dtype = peft_config.lora_dtype
                if quantization_config is not None and lora_dtype is None:
                    lora_dtype = torch.bfloat16

                # Compute effective LoRA rank for MoE modules
                moe_dim = peft_config.dim
                if peft_config.moe_rank_scaling:
                    n_act = module.config.n_activated_experts
                    moe_dim = peft_config.dim // n_act
                    if moe_dim < 1:
                        raise ValueError(
                            f"moe_rank_scaling: dim={peft_config.dim} // n_activated_experts={n_act} "
                            f"gives rank {moe_dim}. Increase dim to at least n_activated_experts."
                        )
                    if peft_config.dim % n_act != 0:
                        logger.warning(
                            "moe_rank_scaling: dim=%d is not evenly divisible by n_activated_experts=%d; "
                            "using floor division rank=%d.",
                            peft_config.dim,
                            n_act,
                            moe_dim,
                        )

                # Replace the module in the model
                new_module = patch_moe_module(
                    module,
                    dim=moe_dim,
                    alpha=peft_config.alpha,
                    lora_A_init_method=peft_config.lora_A_init,
                    lora_dtype=lora_dtype,
                )

                # Find parent and replace
                if "." not in name:
                    setattr(model, name, new_module)
                else:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, new_module)
        else:
            # Standard Linear patching
            linear_types = [nn.Linear] + ([transformer_engine.pytorch.Linear] if HAS_TE else [])
            if isinstance(module, tuple(linear_types)) and matcher.match(module, name):
                num_modules_matched += 1
                # For QLora, set lora_dtype to float16/bfloat16 since base weights are quantized
                lora_dtype = peft_config.lora_dtype
                if quantization_config is not None and lora_dtype is None:
                    lora_dtype = torch.bfloat16

                patch_linear_module(
                    module,
                    dim=peft_config.dim,
                    alpha=peft_config.alpha,
                    use_dora=peft_config.use_dora,
                    dropout=peft_config.dropout,
                    dropout_position=peft_config.dropout_position,
                    lora_A_init_method=peft_config.lora_A_init,
                    lora_dtype=lora_dtype,
                    use_memory_efficient_lora=getattr(peft_config, "use_memory_efficient_lora", True),
                    use_triton=peft_config.use_triton,
                    layer_name=name,
                )

    return num_modules_matched


class LoRATritonFunction(torch.autograd.Function):
    """
    Autograd function that avoids saving the LoRA A activation.

    The default path calls Triton kernel wrappers for forward and backward. Callers can pass
    ``use_triton_kernel=False`` to use PyTorch matmuls while keeping the same memory-efficient
    saved tensor behavior.
    """

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Stores context for LoRA backward pass.
        """
        x, lora_A, lora_B, scale, dtype, *rest = inputs
        ctx.save_for_backward(x, lora_A, lora_B)
        ctx.scale = scale
        ctx.dtype = dtype
        ctx.use_triton_kernel = bool(rest[0]) if rest else True
        ctx.has_residual = len(rest) > 1 and rest[1] is not None
        ctx.num_inputs = len(inputs)

    @staticmethod
    def forward(x, lora_A, lora_B, scale, dtype, use_triton_kernel=True, res=None):
        """
        Forward method for memory-efficient LoRA.

        Reshapes 3D tensors into 2D and then calls either Triton kernels or PyTorch matmuls. When ``res`` is
        provided, the residual is added in-place into the LoRA output to avoid allocating a separate add result.

        Always returns a **2D** tensor; the caller restores the original leading dimensions. Keeping the
        ``(N, out) -> (bs, seq, out)`` reshape *outside* this ``autograd.Function`` means the Function's output
        is never a view, so a downstream consumer may safely mutate the LoRA output in place (the reshape done
        by the caller is an ordinary autograd view, which supports in-place ops).
        """
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
            if res is not None:
                res = res.reshape(-1, res.shape[-1])

        lora_res = F.linear(F.linear(x, lora_A) * scale, lora_B)

        if res is not None:
            lora_res.add_(res)

        return lora_res

    @staticmethod
    def backward(ctx, d_y):
        """Compute gradients for memory-efficient LoRA.

        Reshapes 3D tensors into 2D and then updates d_lora_a, d_lora_b, and dx. The PyTorch matmul
        path recomputes ``x @ lora_A.T`` here instead of saving it from forward.

        Args:
            ctx: Autograd context containing the saved input and LoRA weights.
            d_y: Tensor of shape ``[tokens, out_features]`` containing the output gradient in the forward compute
                dtype.

        Returns:
            Gradients for the custom autograd inputs. The input gradient has shape ``[tokens, in_features]`` or
            ``[batch, sequence, in_features]`` and matches the original input dtype. LoRA weight gradients have
            shapes ``[rank, in_features]`` and ``[out_features, rank]`` and match their weight dtypes. When a
            residual input is present, its gradient matches the original residual shape.
        """
        x, lora_A, lora_B = ctx.saved_tensors
        scale = ctx.scale

        reshape = x.dim() == 3
        if reshape:
            bs, seq_len, d = x.shape
        # forward now returns a 2D output, so d_y arrives 2D; the residual input kept its original
        # (possibly 3D) shape, so its gradient must be reshaped back to match that input.
        d_res = None
        if ctx.has_residual and ctx.needs_input_grad[6]:
            d_res = d_y.reshape(bs, seq_len, -1) if reshape else d_y
        if reshape:
            d_y = d_y.reshape(-1, d_y.shape[-1])
            x = x.reshape(-1, d)

        d_x = d_lora_A = d_lora_B = None
        needs_x, needs_lora_A, needs_lora_B = ctx.needs_input_grad[:3]
        # Custom backward executes outside autocast. Reuse d_y's forward compute dtype for the recomputed
        # matmuls, then cast each returned gradient to the dtype of its corresponding input.
        x_compute = x.to(d_y.dtype) if needs_lora_A or needs_lora_B else x
        lora_A_compute = lora_A.to(d_y.dtype) if needs_x or needs_lora_B else lora_A
        lora_B_compute = lora_B.to(d_y.dtype) if needs_x or needs_lora_A else lora_B
        if needs_x or needs_lora_A:
            d_y_lora_B = torch.matmul(d_y, lora_B_compute)
            d_y_lora_B.mul_(scale)
            if needs_x:
                d_x = torch.matmul(d_y_lora_B, lora_A_compute).to(x.dtype)
            if needs_lora_A:
                d_lora_A = torch.matmul(d_y_lora_B.t(), x_compute).to(lora_A.dtype)

        if needs_lora_B:
            x_lora_A = F.linear(x_compute, lora_A_compute)
            x_lora_A.mul_(scale)
            d_lora_B = torch.matmul(d_y.t(), x_lora_A).to(lora_B.dtype)

        if reshape and d_x is not None:
            d_x = d_x.view(bs, seq_len, d)

        gradients = (d_x, d_lora_A, d_lora_B, None, None)
        if ctx.num_inputs == 7:
            return gradients + (None, d_res)
        if ctx.num_inputs == 6:
            return gradients + (None,)
        return gradients


def apply_memory_efficient_lora(x, lora_A, lora_B, scale, use_triton_kernel, res=None):
    """Run :class:`LoRATritonFunction` and restore the input's leading dimensions.

    ``LoRATritonFunction.forward`` returns a 2D tensor (its reshape is intentionally kept outside the
    autograd Function so the output is never a view). Reshape back to the input rank here; the result
    is an ordinary autograd view, which — unlike a custom-Function output view — a downstream consumer
    may mutate in place (e.g. transformers' gemma3n ``project_per_layer_inputs``).

    Args:
        x: Activation tensor of shape ``[tokens, in_features]`` or ``[batch, sequence, in_features]``.
        lora_A: Tensor of shape ``[rank, in_features]``.
        lora_B: Tensor of shape ``[out_features, rank]``.
        scale: LoRA scaling factor (``alpha / rank``).
        use_triton_kernel: Request the Triton kernels; declined when their dtype precondition
            does not hold (see below).
        res: Optional base-projection output to fold in, in ``x``'s leading shape with
            ``out_features`` trailing.

    Returns:
        Tensor with ``x``'s leading dimensions and ``out_features`` trailing.
    """
    out = LoRATritonFunction.apply(x, lora_A, lora_B, scale, x.dtype, use_triton_kernel, res)
    if x.dim() == 3:
        out = out.reshape(*x.shape[:-1], -1)
    return out
