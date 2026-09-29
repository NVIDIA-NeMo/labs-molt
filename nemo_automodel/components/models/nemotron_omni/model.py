# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""NemotronOmni (NemotronH_Nano_Omni_Reasoning_V3) custom model for Nemo Automodel.

This model is a VLM (vision-language model) with:
- Vision encoder: RADIO v2.5-H (ViT-Huge, patch_size=16) -- loaded from HF
- Audio encoder: Parakeet (FastConformer-based) -- loaded from HF
- LLM: NemotronH (hybrid Mamba+Attention MoE) -- reuses nemotron_v3 custom implementation
- Projectors: MLP projectors for vision->LLM and audio->LLM

Architecture name: "NemotronH_Nano_Omni_Reasoning_V3" (from config.json)
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    round_robin_local_indices,
    shard_batch_aux_only,
    shard_sequence_for_cp_round_robin,
)
from nemo_automodel.components.distributed.context_parallel.utils import cp_dispatcher_suspended
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.nemotron_v3.model import (
    NemotronHForCausalLM as NemotronV3ForCausalLM,
)
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .state_dict_adapter import NemotronOmniStateDictAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helper modules (vision projector, sound projector)
# These match the HF checkpoint exactly.
# ---------------------------------------------------------------------------


class SquaredReLU(nn.Module):
    """Squared ReLU activation: ReLU(x)^2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(torch.nn.functional.relu(x), 2)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


def _pixel_shuffle(x: torch.Tensor, scale_factor: float, ps_version: str) -> torch.Tensor:
    """Rearrange spatial patches into channels with the checkpoint layout.

    Args:
        x: Features shaped [batch, grid_height, grid_width, hidden_size].
        scale_factor: Spatial downsampling ratio.
        ps_version: Pixel-shuffle layout version.

    Returns:
        Features with downsampled spatial axes and expanded channels.
    """
    n, w, h, c = x.size()
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(
        n,
        int(h * scale_factor),
        int(w * scale_factor),
        int(c / (scale_factor * scale_factor)),
    )
    if ps_version == "v1":
        warnings.warn(
            "In ps_version 'v1', the height and width have not been swapped back, which results in a transposed image."
        )
    else:
        x = x.permute(0, 2, 1, 3).contiguous()
    return x


class VisionProjector(nn.Module):
    """MLP projector from vision encoder to LLM hidden space.

    HF checkpoint structure (mlp1):
        mlp1.0.weight  ->  RMSNorm weight  (vit_hidden_size * pixel_shuffle_factor^2,)
        mlp1.1.weight  ->  Linear1 weight  (projector_hidden_size, vit_hidden_size * pixel_shuffle_factor^2)
        mlp1.3.weight  ->  Linear2 weight  (llm_hidden_size, projector_hidden_size)

    Between linear1 and linear2 there is a SquaredReLU activation (index 2 in Sequential,
    but it has no weight). Super 3.5 also stores a vision_final_layernorm here;
    forward applies it before pixel shuffle.
    """

    def __init__(
        self,
        vit_hidden_size: int,
        projector_hidden_size: int,
        llm_hidden_size: int,
        downsample_ratio: float = 0.5,
        *,
        vision_final_layernorm_eps: float | None = None,
    ) -> None:
        """Build the projector and optional checkpoint-owned vision LayerNorm.

        Args:
            vit_hidden_size: Feature width before pixel shuffle.
            projector_hidden_size: Hidden width of the projector MLP.
            llm_hidden_size: Output width of the language model.
            downsample_ratio: Spatial scale applied by pixel shuffle.
            vision_final_layernorm_eps: LayerNorm epsilon, or None for models
                without a final vision norm.
        """
        super().__init__()
        self.vision_final_layernorm = (
            nn.LayerNorm(vit_hidden_size, eps=vision_final_layernorm_eps)
            if vision_final_layernorm_eps is not None
            else None
        )
        self.downsample_ratio = downsample_ratio
        pixel_shuffle_channels = vit_hidden_size * int(1 / downsample_ratio) ** 2
        self.norm = RMSNorm(pixel_shuffle_channels, eps=1e-5)
        self.linear1 = nn.Linear(pixel_shuffle_channels, projector_hidden_size, bias=False)
        self.activation = SquaredReLU()
        self.linear2 = nn.Linear(projector_hidden_size, llm_hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        patch_grid_shapes: list[tuple[int, int]] | None = None,
        ps_version: str = "v2",
    ) -> torch.Tensor:
        """Normalize raw patch features, shuffle each image, then project.

        Args:
            x: Raw features [batch, sum_patches, vit_hidden_size] when
                patch_grid_shapes is provided; otherwise already shuffled
                features [batch, image_tokens, pixel_shuffle_channels].
            patch_grid_shapes: Per-image (height, width) in patches.
            ps_version: Checkpoint pixel-shuffle layout version.

        Returns:
            Projected features [batch, image_tokens, llm_hidden_size].
        """
        # Keep all projector-owned parameters inside this forward so FSDP
        # materializes the final LayerNorm weights before they are accessed.
        if patch_grid_shapes is not None:
            if self.vision_final_layernorm is not None:
                x = self.vision_final_layernorm(x)
            chunks = torch.split(x, [h * w for h, w in patch_grid_shapes], dim=1)
            x = torch.cat(
                [
                    _pixel_shuffle(chunk.reshape(chunk.shape[0], h, w, -1), self.downsample_ratio, ps_version).flatten(
                        1, 2
                    )
                    for chunk, (h, w) in zip(chunks, patch_grid_shapes)
                ],
                dim=1,
            )
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class SoundProjection(nn.Module):
    """MLP projector from sound encoder to LLM hidden space.

    HF checkpoint structure:
        sound_projection.norm.weight       -> RMSNorm weight  (sound_hidden_size,)
        sound_projection.linear1.weight    -> Linear1 weight  (projection_hidden_size, sound_hidden_size)
        sound_projection.linear2.weight    -> Linear2 weight  (llm_hidden_size, projection_hidden_size)
    """

    def __init__(
        self,
        sound_hidden_size: int,
        projection_hidden_size: int,
        llm_hidden_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.norm = RMSNorm(sound_hidden_size, eps=1e-5)
        self.linear1 = nn.Linear(sound_hidden_size, projection_hidden_size, bias=bias)
        self.activation = SquaredReLU()
        self.linear2 = nn.Linear(projection_hidden_size, llm_hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------


class NemotronOmniConfig(PretrainedConfig):
    """Configuration for the NemotronOmni (NemotronH_Nano_Omni_Reasoning_V3) model.

    This wraps the HF config and provides easy access to sub-configs.
    """

    model_type = "NemotronH_Nano_Omni_Reasoning_V3"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        sound_config=None,
        force_image_size=512,
        downsample_ratio=0.5,
        patch_size=16,
        template=None,
        ps_version="v2",
        image_tag_type="internvl",
        projector_hidden_size=20480,
        vit_hidden_size=1280,
        img_context_token_id=18,
        video_context_token_id=131081,
        sound_context_token_id=27,
        video_pruning_rate=0.7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_config = vision_config
        self.llm_config = llm_config
        self.sound_config = sound_config
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.patch_size = patch_size
        self.template = template
        self.ps_version = ps_version
        self.image_tag_type = image_tag_type
        self.projector_hidden_size = projector_hidden_size
        self.vit_hidden_size = vit_hidden_size
        self.img_context_token_id = img_context_token_id
        self.video_context_token_id = video_context_token_id
        self.sound_context_token_id = sound_context_token_id
        self.video_pruning_rate = video_pruning_rate


# ---------------------------------------------------------------------------
# Model proxy for MoE parallelizer compatibility
# ---------------------------------------------------------------------------


class _ModelProxy:
    """Thin proxy so the MoE parallelizer can navigate model.model.moe_config
    and model.model -> get_text_module -> .layers without changing the weight
    hierarchy.

    The parallelizer (parallelizer.py) expects:
        model.model.moe_config           (for expert-count validation)
        model.model -> get_text_module()  (finds language_model attr) -> .layers

    By setting self.model = _ModelProxy(self.language_model) on the VLM:
        model.model.moe_config            -> language_model.model.moe_config  OK
        get_text_module(model.model)       -> model.model.language_model
                                           == language_model.model (NemotronV3Model)
                                           -> .layers                          OK
    """

    def __init__(self, llm: "NemotronV3ForCausalLM"):
        # llm is NemotronHForCausalLM which has .model = NemotronV3Model
        self.moe_config = llm.model.moe_config
        # Expose the inner NemotronV3Model as 'language_model' so that
        # get_text_module() can find it and access .layers
        self.language_model = llm.model


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class NemotronOmniForConditionalGeneration(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """NemotronOmni VLM model for conditional generation (training).

    Wraps:
    - Vision encoder (RADIO v2.5-H) -- HF implementation via trust_remote_code
    - Audio encoder (Parakeet) -- HF implementation via trust_remote_code
    - Vision projector (MLP: RMSNorm -> Linear -> SquaredReLU -> Linear)
    - Sound projector (MLP: RMSNorm -> Linear -> SquaredReLU -> Linear)
    - Language model (NemotronH hybrid Mamba+Attention MoE) -- nemotron_v3 custom impl

    The LLM part reuses the nemotron_v3 implementation (NemotronHForCausalLM) which
    has custom DTensor parallelism for the Mamba+Attention hybrid MoE architecture.
    """

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    # Same fp32 keep-list as NemotronHForCausalLM: ``cast_model_to_dtype`` reads these
    # attributes from the *outermost* module, so without them the wrapper-level cast in
    # ``initialize_weights`` turns the router's ``e_score_correction_bias`` buffers into
    # bf16. The checkpoint stores them in fp32 (values ~3.97 +- 0.024, none bf16-exact) and
    # the bf16 rounding (ulp 0.0156 at 4.0) is larger than the sigmoid routing margins, so
    # top-k expert selection diverged from the HF reference on ~85% of tokens.
    _keep_in_fp32_modules_strict = ["e_score_correction_bias", "_fp32_params"]
    # CP submesh, installed by the MoE parallelizer's apply_cp when context
    # parallelism is active; None means the forward embeds and shards nothing for CP.
    cp_mesh = None

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = False
        supports_ep: bool = True
        # MTP under CP delegates to the NemotronV3 LM's globally-prepared per-depth
        # targets (prepare_mtp_inputs_for_cp) sharded by the same sharder as the inputs.
        supports_mtp_cp: bool = True

    @classmethod
    def from_config(
        cls,
        config,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        """Create model from config.

        Args:
            config: NemotronH_Nano_Omni_Reasoning_V3 config (HF config with trust_remote_code)
            backend: Backend configuration
            **kwargs: Additional arguments

        Returns:
            NemotronOmniForConditionalGeneration instance
        """
        return cls(config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        """Load pretrained model.

        Args:
            pretrained_model_name_or_path: Path or name of pretrained model
            *model_args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            NemotronOmniForConditionalGeneration instance
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        """Initialize NemotronOmniForConditionalGeneration.

        Args:
            config: NemotronH_Nano_Omni_Reasoning_V3 config
            backend: Backend configuration
            **kwargs: Additional arguments
        """
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.backend = backend or BackendConfig()

        # ---------------------------------------------------------------
        # Extract sub-configs
        # ---------------------------------------------------------------
        llm_config = config.llm_config
        vision_config = config.vision_config
        sound_config = getattr(config, "sound_config", None)

        # Store key VLM parameters
        self.force_image_size = getattr(config, "force_image_size", 512)
        self.patch_size = getattr(config, "patch_size", 16)
        self.downsample_ratio = getattr(config, "downsample_ratio", 0.5)
        self.ps_version = getattr(config, "ps_version", "v2")
        self.img_context_token_id = getattr(config, "img_context_token_id", 18)
        self.video_context_token_id = getattr(config, "video_context_token_id", 131081)
        self.sound_context_token_id = getattr(config, "sound_context_token_id", 27)

        self.num_image_token = int((self.force_image_size // self.patch_size) ** 2 * (self.downsample_ratio**2))
        logger.info(f"NemotronOmni: num_image_token={self.num_image_token}")
        logger.info(f"NemotronOmni: ps_version={self.ps_version}")
        logger.info(f"NemotronOmni: img_context_token_id={self.img_context_token_id}")

        vit_hidden_size = getattr(config, "vit_hidden_size", 1280)
        projector_hidden_size = getattr(config, "projector_hidden_size", 20480)
        llm_hidden_size = llm_config.hidden_size

        # ---------------------------------------------------------------
        # 1. Language Model (reuses nemotron_v3 custom implementation)
        # ---------------------------------------------------------------
        logger.info("NemotronOmni: Creating NemotronV3 LLM backbone...")
        self.language_model = NemotronV3ForCausalLM(llm_config, backend=self.backend, **kwargs)
        logger.info(
            f"NemotronOmni: LLM created with {llm_config.num_hidden_layers} layers, "
            f"hidden_size={llm_config.hidden_size}, vocab_size={llm_config.vocab_size}"
        )

        # ---------------------------------------------------------------
        # 2. Vision Encoder (RADIO v2.5-H from HF)
        # ---------------------------------------------------------------
        logger.info("NemotronOmni: Creating RADIO vision encoder from HF config...")
        dtype = get_dtype(getattr(llm_config, "torch_dtype", None), torch.bfloat16)
        # FIX: Force timm to use eager (math) attention instead of fused SDPA
        # for the RADIO ViT. This ensures numerical parity with the HF model
        # which also uses eager attention. The timm Attention class reads this
        # global flag at __init__ time, so it must be set BEFORE model creation.
        from timm.layers.config import set_fused_attn as _timm_set_fused_attn

        _timm_set_fused_attn(False)
        # Resolve RadioModel directly from THIS checkpoint's own remote-code module rather
        # than through the generic `AutoModel.from_config(..., trust_remote_code=True)`
        # factory. That factory dispatches on `model_type` ("radio") through a *global*
        # registry: with multiple RADIO/omni checkpoints' remote code cached under the same
        # $HF_HOME (each with its own "radio" registration -- e.g. a legacy timm-based
        # implementation for NemotronH_Nano_Omni_Reasoning_V3 vs. the newer
        # transformers-native embeddings/encoder RadioModel some checkpoints ship, such as
        # NemotronH_Omni_Reasoning_V3), whichever module happened to register "radio" first
        # in this process wins -- nondeterministically resolving the wrong class/module tree
        # for the checkpoint actually being loaded. Explicit dynamic-module resolution is
        # deterministic and always matches the checkpoint's own weights.
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            radio_model_cls = get_class_from_dynamic_module("modeling_radio.RadioModel", config.name_or_path)
            self.vision_model = radio_model_cls(vision_config)
        except (ImportError, OSError, AttributeError):
            # Checkpoints that don't ship a "modeling_radio.RadioModel" module/class under
            # this exact name/path fall back to the generic, registry-based resolution.
            self.vision_model = AutoModel.from_config(vision_config, trust_remote_code=True)
        _timm_set_fused_attn(True)  # Restore default for any subsequent timm usage
        # WAR for transformers issue 38358
        if hasattr(self.vision_model, "model") and hasattr(self.vision_model.model, "_init_weights"):
            self.vision_model.model._initialize_weights = self.vision_model.model._init_weights
        # Make preprocessor external (required by RADIO): the image processor already
        # normalizes pixel values, so the model-side input conditioner must become an
        # Identity -- exactly what the checkpoint's own NemotronH_Omni_Reasoning_V3.__init__
        # does. Legacy checkpoints expose it under `radio_model`; the transformers-native
        # RadioModel exposes it directly. Leaving it active normalizes twice and corrupts
        # the vision features (RADIO output cosine ~0.32 vs the HF reference).
        if hasattr(self.vision_model, "radio_model"):
            self.vision_model.radio_model.make_preprocessor_external()
        elif hasattr(self.vision_model, "make_preprocessor_external"):
            self.vision_model.make_preprocessor_external()

        # 3D patch projector for temporally-packed video frames. Only present when the
        # checkpoint ships a `patch_generator.video_embedder` weight (i.e. v3+).
        self.video_temporal_patch_dim = getattr(config, "video_temporal_patch_size", None)
        if self.video_temporal_patch_dim is not None and hasattr(self.vision_model, "radio_model"):
            pg = self.vision_model.radio_model.model.patch_generator
            pg.video_embedder = nn.Linear(
                in_features=self.video_temporal_patch_dim * 3 * pg.patch_size * pg.patch_size,
                out_features=pg.embed_dim,
                bias=False,
            )

        # The native RadioModel's RadioLayerScale always creates a learnable
        # `lambda1` parameter, even when `layerscale_value == 1.0` (a pure
        # multiplicative no-op: `x * 1.0 == x`). Checkpoints trained with
        # layerscale disabled (this value) never save these tensors, so
        # checkpoint loading would otherwise fail with a spurious "missing key"
        # error for a parameter whose only correct value is the identity.
        # Replacing with nn.Identity() is exact (not an approximation) and
        # removes both the memory and the checkpoint-key requirement.
        if hasattr(self.vision_model, "encoder") and getattr(vision_config, "layerscale_value", None) == 1.0:
            num_replaced = 0
            for layer in self.vision_model.encoder.layer:
                if hasattr(layer, "layer_scale1"):
                    layer.layer_scale1 = nn.Identity()
                    num_replaced += 1
                if hasattr(layer, "layer_scale2"):
                    layer.layer_scale2 = nn.Identity()
                    num_replaced += 1
            if num_replaced:
                logger.info(f"NemotronOmni: Replaced {num_replaced} no-op RADIO layer_scale modules with nn.Identity()")

        self.vision_model = self.vision_model.to(dtype)

        # Convert RADIO buffers that are NOT in the HF checkpoint to
        # non-persistent so the DCP loader doesn't expect them on disk.
        self._make_missing_buffers_non_persistent(self.vision_model)
        logger.info("NemotronOmni: Vision encoder created (RADIO v2.5-H)")

        # ---------------------------------------------------------------
        # 3. Vision Projector (MLP: RMSNorm -> Linear -> SquaredReLU -> Linear)
        # ---------------------------------------------------------------
        self.vision_projector = VisionProjector(
            vit_hidden_size=vit_hidden_size,
            projector_hidden_size=projector_hidden_size,
            llm_hidden_size=llm_hidden_size,
            downsample_ratio=self.downsample_ratio,
            # This norm belongs to the pretrained Super 3.5 vision architecture.
            # Disabling the language MTP objective must not remove it.
            vision_final_layernorm_eps=(
                vision_config.layer_norm_eps
                if "NemotronH_Omni_Reasoning_V3" in (getattr(config, "architectures", None) or [])
                else None
            ),
        ).to(dtype)
        logger.info(
            f"NemotronOmni: Vision projector created "
            f"(vit_hidden={vit_hidden_size} -> proj_hidden={projector_hidden_size} -> llm_hidden={llm_hidden_size})"
        )

        # ---------------------------------------------------------------
        # 4. Audio Encoder (Parakeet from HF) + Sound Projector
        # ---------------------------------------------------------------
        if sound_config is not None:
            sound_hidden_size = getattr(sound_config, "hidden_size", 1024)
            sound_proj_hidden_size = getattr(sound_config, "projection_hidden_size", 4096)
            sound_proj_bias = getattr(sound_config, "projection_bias", False)

            logger.info("NemotronOmni: Creating Parakeet sound encoder...")
            try:
                from transformers import ParakeetEncoder, ParakeetEncoderConfig

                # Build ParakeetEncoderConfig from sound_config
                parakeet_config_dict = {
                    "attention_bias": getattr(sound_config, "attention_bias", False),
                    "hidden_size": sound_hidden_size,
                    "num_attention_heads": getattr(sound_config, "num_attention_heads", 8),
                    "num_hidden_layers": getattr(sound_config, "num_hidden_layers", 24),
                    "intermediate_size": getattr(sound_config, "intermediate_size", 4096),
                    "conv_kernel_size": getattr(sound_config, "conv_kernel_size", 9),
                    "convolution_bias": getattr(sound_config, "convolution_bias", False),
                    "subsampling_conv_channels": getattr(sound_config, "subsampling_conv_channels", 256),
                    "subsampling_conv_kernel_size": getattr(sound_config, "subsampling_conv_kernel_size", 3),
                    "subsampling_conv_stride": getattr(sound_config, "subsampling_conv_stride", 2),
                    "subsampling_factor": getattr(sound_config, "subsampling_factor", 8),
                    "num_mel_bins": getattr(sound_config, "num_mel_bins", 128),
                }
                parakeet_config = ParakeetEncoderConfig(**parakeet_config_dict)
                self.sound_encoder = ParakeetEncoder(parakeet_config).to(dtype)
                logger.info(f"NemotronOmni: Sound encoder created (hidden_size={sound_hidden_size})")
            except ImportError:
                logger.warning(
                    "NemotronOmni: ParakeetEncoder not available in transformers. Sound encoder will not be loaded."
                )
                self.sound_encoder = None

            self.sound_projection = SoundProjection(
                sound_hidden_size=sound_hidden_size,
                projection_hidden_size=sound_proj_hidden_size,
                llm_hidden_size=llm_hidden_size,
                bias=sound_proj_bias,
            ).to(dtype)
            logger.info(
                f"NemotronOmni: Sound projector created "
                f"(sound_hidden={sound_hidden_size} -> proj_hidden={sound_proj_hidden_size} -> llm_hidden={llm_hidden_size})"
            )
        else:
            self.sound_encoder = None
            self.sound_projection = None
            logger.info("NemotronOmni: No sound config, audio encoder disabled.")

        # ---------------------------------------------------------------
        # 5. Model proxy for MoE parallelizer compatibility
        # ---------------------------------------------------------------
        # The MoE parallelizer (parallelizer.py) expects model.model.moe_config
        # and apply_ep navigates: model.model -> get_text_module() -> .layers.
        # We create a thin _ModelProxy that exposes these attributes:
        #   self.model.moe_config  -> language_model.model.moe_config
        #   self.model.language_model -> language_model.model (NemotronV3Model with .layers)
        self.model = _ModelProxy(self.language_model)
        logger.info("NemotronOmni: Model proxy created for parallelizer compatibility")

        # ---------------------------------------------------------------
        # 6. State dict adapter
        # ---------------------------------------------------------------
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = NemotronOmniStateDictAdapter(
                config=config,
                llm_config=llm_config,
                moe_config=self.language_model.model.moe_config,
                backend=self.backend,
                dtype=dtype,
                # Checkpoints whose remote code ships the transformers-native RadioModel
                # (embeddings/encoder module tree, e.g. NemotronH_Omni_Reasoning_V3) save
                # weights under the older timm-style `radio_model.model.blocks` naming and
                # need the same key-rename + fused-QKV split transformers itself applies via
                # `register_checkpoint_conversion_mapping("radio", ...)` during
                # `from_pretrained`. Checkpoints whose remote code ships the legacy
                # `radio_model`-wrapped class (identified by `hasattr(self.vision_model,
                # "radio_model")`, e.g. NemotronH_Nano_Omni_Reasoning_V3) already match the
                # checkpoint's native layout and need no remap.
                vision_uses_native_radio=not hasattr(self.vision_model, "radio_model"),
            )
            logger.info("NemotronOmni: State dict adapter created")

    # ------------------------------------------------------------------
    # Buffer management helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_missing_buffers_non_persistent(module: nn.Module) -> None:
        """Convert persistent buffers that are NOT saved in HF checkpoints
        to non-persistent buffers.

        The RADIO vision encoder registers some buffers (e.g. ``summary_idxs``)
        as persistent, but the HF checkpoint does not contain them.  When the DCP
        loader builds its load plan it expects every persistent buffer to appear
        in the checkpoint and raises ``RuntimeError: Missing key`` otherwise.

        This method re-registers such buffers as non-persistent so they are
        kept at their init-time values and not expected on disk.
        """
        # Known buffers not in the HF RADIO checkpoint
        _NON_CHECKPOINT_BUFFERS = {"summary_idxs"}

        for name, sub in module.named_modules():
            for buf_name in list(sub._buffers.keys()):
                if buf_name in _NON_CHECKPOINT_BUFFERS:
                    buf = sub._buffers[buf_name]
                    # Re-register as non-persistent (keeps the tensor, removes
                    # it from state_dict())
                    sub.register_buffer(buf_name, buf, persistent=False)
                    logger.info(
                        f"NemotronOmni: Converted buffer '{name}.{buf_name}' to non-persistent (not in HF checkpoint)"
                    )

    # ------------------------------------------------------------------
    # Embedding access (required by VLM training infrastructure)
    # ------------------------------------------------------------------

    def get_input_embeddings(self):
        """Return the input embeddings from the language model."""
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set the input embeddings of the language model."""
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        """Return the output embeddings (lm_head) from the language model."""
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings (lm_head) of the language model."""
        self.language_model.set_output_embeddings(new_embeddings)

    @property
    def lm_head(self) -> nn.Module | None:
        """Return the nested language-model output head without re-registering it."""
        return self.language_model.lm_head

    # ------------------------------------------------------------------
    # Vision feature extraction
    # ------------------------------------------------------------------

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        """Pixel shuffle for downsampling spatial resolution while increasing channels.

        Args:
            x: Input tensor [N, W, H, C]
            scale_factor: Downsampling ratio (default 0.5 = halve spatial dims)

        Returns:
            Shuffled tensor [N, W*scale, H*scale, C/(scale^2)]
        """
        return _pixel_shuffle(x, scale_factor, self.ps_version)

    def extract_feature(self, pixel_values: "torch.Tensor | list[torch.Tensor]") -> "torch.Tensor | list[torch.Tensor]":
        """Extract vision features from pixel values through RADIO + projector.

        Args:
            pixel_values: Image tensors [num_tiles, C, H, W], or a list of
                per-image tensors ([C, H, W] or [num_tiles, C, H, W]) when the
                batch mixes resolutions and cannot be stacked. The collate fn
                (`nemotron_omni_collate_fn`) emits the list form whenever the
                dataset is variable-resolution and `local_batch_size > 1`.

        Returns:
            Vision embeddings [num_tiles, num_tokens, llm_hidden_size], or, for
            list input, one such tensor per list element. Each element keeps its
            own `num_tokens` because that depends on the image resolution.
        """
        # Force vision model to eval mode for deterministic spectral reparam.
        # RADIO uses spectral reparameterization with power iteration that is
        # non-deterministic in train mode (random _u/_v init). Since the vision
        # tower is frozen during training, eval mode is correct and produces
        # reproducible outputs.
        was_training = self.vision_model.training
        self.vision_model.eval()
        try:
            if isinstance(pixel_values, (list, tuple)):
                # RADIO only accepts a dense (B, C, H, W) tensor, so variable
                # resolutions have to be run one at a time.
                return [self._extract_feature_dense(pv[None] if pv.dim() == 3 else pv) for pv in pixel_values]
            return self._extract_feature_dense(pixel_values)
        finally:
            if was_training:
                self.vision_model.train()

    def _extract_feature_dense(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a dense image batch, applying the final norm before pixel shuffle.

        Args:
            pixel_values: Images shaped [batch, channels, height, width].

        Returns:
            Projected features shaped [batch, image_tokens, llm_hidden_size].
        """
        vit_embeds = self.vision_model(pixel_values).features
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)

        # Patch grid comes from input dims so non-square dynamic-res tiles also work.
        # Native RadioModel (embeddings/encoder tree) exposes `patch_size` directly; the
        # legacy `radio_model`-wrapped class only has it nested under patch_generator.
        if hasattr(self.vision_model, "radio_model"):
            patch_size = self.vision_model.radio_model.model.patch_generator.patch_size
        else:
            patch_size = self.vision_model.patch_size
        _, _, H, W = pixel_values.shape
        return self.vision_projector(
            vit_embeds, patch_grid_shapes=[(H // patch_size, W // patch_size)], ps_version=self.ps_version
        )

    def extract_feature_dynamic(
        self,
        pixel_values: torch.Tensor,
        imgs_sizes: "torch.Tensor | list[tuple[int, int]]",
    ) -> torch.Tensor:
        """Dynamic-resolution feature extraction (no tile splitting).

        Matches vLLM's dynamic-resolution vision path for Nano v3 VL /
        Nemotron-Omni (see 3rdparty/vllm/vllm/model_executor/models/
        nano_nemotron_vl.py). Required when the rollout uses
        DynamicResolutionImageTiler — tile-based `extract_feature` would
        produce different embeddings and break rollout/train logprob
        agreement.

        Unlike vLLM's RADIO port (which supports packed `imgs_sizes=` inputs),
        the HF RADIO from nvidia/C-RADIOv2-H only accepts a dense
        `(B, C, H, W)` tensor. We crop each padded image back to its real
        size and run the vision model per-image, then concatenate features.

        Args:
            pixel_values: [num_images, C, H_padded, W_padded] batch of
                dynamically-resized images padded to the batch max (h, w).
            imgs_sizes: [num_images, 2] actual (h, w) per image (torch tensor
                of ints) or an equivalent list of tuples.

        Returns:
            Vision embeddings [sum_num_embeddings_after_pixel_shuffle,
            llm_hidden_size].
        """
        if isinstance(imgs_sizes, torch.Tensor):
            imgs_sizes_list: list[tuple[int, int]] = [
                (int(imgs_sizes[i, 0].item()), int(imgs_sizes[i, 1].item())) for i in range(imgs_sizes.shape[0])
            ]
        else:
            imgs_sizes_list = [(int(h), int(w)) for (h, w) in imgs_sizes]

        was_training = self.vision_model.training
        self.vision_model.eval()

        # Cast to the vision model's expected dtype at the boundary (not at
        # dataset-load) to match vLLM's normalization-in-fp32 →
        # cast-to-model-dtype-at-boundary order exactly. Pre-casting in the
        # data pipeline produced a subtle per-token systematic bias that
        # showed up as `sampling_importance_ratio` ~0.80 vs mbridge ~0.99.
        vm_dtype = next(
            (p.dtype for p in self.vision_model.parameters()),
            pixel_values.dtype,
        )
        if pixel_values.dtype != vm_dtype:
            pixel_values = pixel_values.to(dtype=vm_dtype)

        per_image_feats: list[torch.Tensor] = []
        for i, (h, w) in enumerate(imgs_sizes_list):
            # Crop back to the real resolution before calling RADIO — the
            # pixel_values tensor is padded to the per-batch (H_padded,
            # W_padded). Slice both spatial dims.
            img = pixel_values[i : i + 1, :, :h, :w]
            out = self.vision_model(img)
            # HF RADIO returns either a RadioOutput namedtuple or a dict when
            # adaptors are configured. `.features` is the per-patch features
            # (N, L, C) layout with feature_fmt='NLC'.
            feats = getattr(out, "features", None)
            if feats is None:
                # Backbone dict variant.
                feats = out["backbone"].features if isinstance(out, dict) else out[1]
            feats = feats.to(dtype=torch.bfloat16)
            # feats: [1, (h//p)*(w//p), C_feat]
            per_image_feats.append(feats)

        if was_training:
            self.vision_model.train()

        vit_embeds = torch.cat(per_image_feats, dim=-2)
        return self.vision_projector(
            vit_embeds,
            patch_grid_shapes=[(h // self.patch_size, w // self.patch_size) for h, w in imgs_sizes_list],
            ps_version=self.ps_version,
        )

    def _pixel_shuffle_dynamic_res(self, x: torch.Tensor, imgs_sizes: list[tuple[int, int]]) -> torch.Tensor:
        """Per-image pixel-shuffle for dynamic-resolution outputs.

        Ported from vLLM's `NanoNemotronVLMultimodal.pixel_shuffle_dynamic_res`.
        Splits `x` along the sequence dim by per-image patch counts, reshapes
        each split to (N, H_patches, W_patches, C_feat), applies pixel_shuffle
        with `downsample_ratio`, and flattens back to a concatenated (N, L', C).
        """
        patch_dim = self.patch_size
        seq_lens = [(h // patch_dim) * (w // patch_dim) for (h, w) in imgs_sizes]
        splits = torch.split(x, seq_lens, dim=-2)
        out = []
        for i, sv in enumerate(splits):
            h = imgs_sizes[i][0] // patch_dim
            w = imgs_sizes[i][1] // patch_dim
            sv = sv.reshape(sv.shape[0], h, w, -1)
            sv = self.pixel_shuffle(sv, scale_factor=self.downsample_ratio)
            sv = sv.flatten(1, 2)
            out.append(sv)
        return torch.cat(out, dim=-2)

    def extract_video_feature(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """Pack ``T = video_temporal_patch_dim`` frames into channels and run the ViT.

        Returns embeddings shaped like ``extract_feature`` output, but with
        ``ceil(N_frames / T)`` rows instead of one row per frame.
        """
        assert self.video_temporal_patch_dim is not None, "video_temporal_patch_size missing from config"
        pg = self.vision_model.radio_model.model.patch_generator
        T = self.video_temporal_patch_dim
        N, C, H, W = pixel_values_videos.shape

        if N % T != 0:
            pad = pixel_values_videos[-1:].expand(T - (N % T), -1, -1, -1)
            pixel_values_videos = torch.cat([pixel_values_videos, pad], dim=0)
            N = pixel_values_videos.shape[0]
        num_groups = N // T

        # Per-patch feature order ends up `[t=0,c=0..C-1, t=1,c=0..C-1, ...]`, which is
        # the layout the checkpoint's `video_embedder.weight` expects.
        x = pixel_values_videos.reshape(num_groups, T * C, H, W)

        was_training = self.vision_model.training
        self.vision_model.eval()
        orig_embedder = pg.embedder
        pg.embedder = pg.video_embedder
        try:
            vit_embeds = self.vision_model(x).features
        finally:
            pg.embedder = orig_embedder
            if was_training:
                self.vision_model.train()

        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
        patch_size = pg.patch_size
        return self.vision_projector(
            vit_embeds, patch_grid_shapes=[(H // patch_size, W // patch_size)], ps_version=self.ps_version
        )

    def extract_sound_feature(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract and project sound features from audio input.

        Args:
            input_features: Mel spectrogram features [batch, seq_len, feature_dim]
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Sound embeddings projected to LLM hidden size
        """
        if self.sound_encoder is None:
            raise RuntimeError("Sound encoder not initialized.")
        outputs = self.sound_encoder(
            input_features=input_features,
            attention_mask=attention_mask,
        )
        sound_embeds = outputs.last_hidden_state
        sound_embeds = sound_embeds.to(dtype=torch.bfloat16)
        sound_embeds = self.sound_projection(sound_embeds)
        return sound_embeds

    # ------------------------------------------------------------------
    # Context parallelism pre-processing
    # ------------------------------------------------------------------

    @property
    def mtp(self):
        """The LM's MTP head (so the MoE parallelizer / FSDP block iteration find its MoE sublayers)."""
        return getattr(self.language_model, "mtp", None)

    @property
    def mtp_config(self):
        """MTP head configuration of the wrapped NemotronV3 LM (drives ``supports.mtp_enabled``)."""
        return getattr(self.language_model, "mtp_config", None)

    def prepare_mtp_inputs_for_cp(self, batch: dict[str, Any], *, ignore_index: int = -100):
        """Globally ordered MTP future-token inputs/targets, prepared before CP sharding.

        Delegates to the NemotronV3 LM (the batch is still the full text-token stream;
        media placeholders are ordinary tokens at this stage).
        """
        return self.language_model.prepare_mtp_inputs_for_cp(batch, ignore_index=ignore_index)

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        num_chunks: int = 1,
    ) -> dict[str, Any]:
        """Return a sharder-only CP backend; embed + splice + shard happen in forward.

        Embedding and the image/video/audio multimodal scatter now run inside
        ``forward`` per microbatch (the existing ``inputs_embeds is None`` block),
        which then round-robin shards the result with
        :func:`shard_sequence_for_cp_round_robin`. The returned
        :class:`ContextParallelSharder` round-robin-shards only the no-grad aux
        streams (labels/position_ids/loss_mask/padding_mask) and leaves
        ``input_ids`` and the media inputs full-length for the forward. NemotronOmni
        uses plain 1-D positions, so no ``position_ids`` are computed here.

        Args:
            batch: The full-sequence batch (with ``input_ids`` ``[batch,
                sequence]``); left intact.
            num_chunks: Accepted for hook-signature parity; unused (round-robin CP).
        """
        del num_chunks
        if batch.get("qkv_format") == "thd":
            # Packed THD: defer to the framework TE THD sharder, which partitions
            # input_ids/labels/position_ids per document (tex.thd_get_partitioned_indices)
            # before embedding -- the layout TE's THD CP attention and the Mamba CP
            # helpers require. Media inputs stay full-length; record which GLOBAL
            # positions are media placeholders so forward can pick this rank's
            # slice of the encoder features (see _select_local_media_features).
            input_ids = batch.get("input_ids")
            if input_ids is None:
                return {}
            flat = input_ids.reshape(-1)
            prepared = {"_nemotron_omni_global_image_mask": flat == self.img_context_token_id}
            if self.sound_context_token_id is not None:
                prepared["_nemotron_omni_global_sound_mask"] = flat == self.sound_context_token_id
            return prepared
        return {
            "cp_sharder": ContextParallelSharder(
                shard_batch=shard_batch_aux_only,
                local_token_global_indices=round_robin_local_indices,
            )
        }

    @staticmethod
    def _select_local_media_features(
        features: torch.Tensor,
        global_mask: torch.Tensor,
        local_selected: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cp_size: int,
        cp_rank: int,
    ) -> torch.Tensor:
        """Pick this CP rank's slice of full-sequence media features (packed THD CP).

        The framework TE THD sharder hands forward an ``input_ids`` shard (TE's
        per-document partition) while the media encoders ran on every image /
        clip, so ``features`` is ordered by GLOBAL placeholder position. Map each
        local token to its global position, then to its feature row.

        Args:
            features: Tensor of shape [num_global_media_tokens, hidden], ordered by
                media-placeholder position in the global packed sequence.
            global_mask: Boolean Tensor of shape [global_tokens], flattened from the
                global packed sequence, with true entries at media placeholders.
            local_selected: Boolean Tensor of shape [local_tokens], with true entries
                at media placeholders in this rank's TE THD partition.
            cu_seqlens: Int32 Tensor of shape [num_documents + 1] containing cumulative
                unpadded document boundaries for the global packed sequence.
            cp_size: Number of context-parallel ranks.
            cp_rank: This rank's zero-based index in the context-parallel group.

        Returns:
            Tensor of shape [num_local_media_tokens, hidden] containing this rank's
            media-feature rows in local placeholder order.
        """
        import transformer_engine_torch as tex  # noqa: PLC0415

        cu = cu_seqlens.to(dtype=torch.int32)
        local_indices = tex.thd_get_partitioned_indices(cu, int(cu[-1].item()), cp_size, cp_rank).to(torch.long)
        local_selected = local_selected.reshape(-1)
        if local_indices.numel() != local_selected.numel():
            raise ValueError(
                f"NemotronOmni packed CP: {local_indices.numel()} partition indices for "
                f"{local_selected.numel()} local tokens."
            )
        feature_index_by_token = global_mask.reshape(-1).to(device=local_indices.device, dtype=torch.long).cumsum(0) - 1
        local_feature_indices = feature_index_by_token.index_select(0, local_indices)[
            local_selected.to(local_indices.device)
        ]
        return features.index_select(0, local_feature_indices.to(features.device))

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_flags: torch.LongTensor | None = None,
        imgs_sizes: torch.LongTensor | None = None,
        past_key_values: List[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        sound_features: torch.FloatTensor | None = None,
        sound_attention_mask: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[dict, Tuple, CausalLMOutputWithPast]:
        """Forward pass for training.

        This follows the same pattern as the HF NemotronH_Nano_Omni_Reasoning_V3.forward():
        1. Get text embeddings from LLM embed_tokens
        2. Extract vision features from pixel_values
        3. Replace image token embeddings with vision embeddings
        4. Run LLM forward pass
        5. Compute loss if labels provided

        Args:
            pixel_values: Image pixel values [num_tiles, C, H, W]
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position IDs (ignored by the RoPE-free backbone; forwarded to the MTP head)
            image_flags: Flags indicating real images vs padding [num_tiles, 1]
            labels: Token IDs for loss computation [batch, seq_len]
            inputs_embeds: Pre-computed input embeddings (optional)
            use_cache: Whether to use caching (not used in training)
            output_hidden_states: Whether the returned output should carry the
                final decoder hidden states (required for fused linear
                cross-entropy / cut-CE). Defaults to the text sub-config's
                ``output_hidden_states`` when ``None``.
            logits_to_keep: If 0 (default), compute logits for all positions;
                if > 0, only compute logits for the last ``logits_to_keep``
                positions (used by fused linear cross-entropy to avoid the full
                logit matrix). Forwarded to the language-model lm_head gating.
            **kwargs: Additional arguments

        Returns:
            CausalLMOutputWithPast with loss and logits
        """
        return_dict = return_dict if return_dict is not None else True
        # Resolve from the text/decoder sub-config (the top-level NemotronOmni
        # config has no output_hidden_states; the recipe toggles it on llm_config).
        if output_hidden_states is None:
            llm_config = getattr(getattr(self, "config", None), "llm_config", None)
            output_hidden_states = getattr(llm_config, "output_hidden_states", False)

        # Caller pre-supplied inputs_embeds (CP path: prepare_model_inputs_for_cp
        # ran the multimodal scatter on the un-sharded sequence before
        # context_parallel sharded the tensors). In that case skip the embed +
        # multimodal-replacement block entirely; the shards are already correct.
        _embeds_pre_built = inputs_embeds is not None
        # Packed THD under CP: input_ids arrive already partitioned per document by
        # the framework TE THD sharder (cu_seqlens/cp_size/cp_rank come with it);
        # media features must be narrowed to this rank's placeholders.
        _global_image_mask = kwargs.pop("_nemotron_omni_global_image_mask", None)
        _global_sound_mask = kwargs.pop("_nemotron_omni_global_sound_mask", None)
        _thd_cp = (
            kwargs.get("qkv_format") == "thd"
            and kwargs.get("cu_seqlens") is not None
            and int(kwargs.get("cp_size", 1)) > 1
        )

        def _local_media(
            features: torch.Tensor,
            global_mask: torch.Tensor | None,
            selected: torch.Tensor,
        ) -> torch.Tensor:
            """Select local packed-THD media rows when context parallelism is active.

            Args:
                features: Tensor of shape [num_global_media_tokens, hidden].
                global_mask: Optional Boolean Tensor of shape [global_tokens] marking
                    media placeholders in the global packed sequence.
                selected: Boolean Tensor of shape [local_tokens] marking media
                    placeholders in this rank's local sequence.

            Returns:
                Tensor of shape [num_local_media_tokens, hidden] for packed THD CP,
                or ``features`` unchanged outside that path.
            """
            if not _thd_cp or global_mask is None:
                return features
            return self._select_local_media_features(
                features, global_mask, selected, kwargs["cu_seqlens"], int(kwargs["cp_size"]), int(kwargs["cp_rank"])
            )

        # Get text embeddings
        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Process vision inputs. We support two mutually-exclusive paths:
        #
        #   1) dynamic-resolution (imgs_sizes is not None)
        #      — matches vLLM's DynamicResolutionImageTiler. Each image is a
        #        single variable-resolution tensor; extract_feature_dynamic
        #        emits exactly one contiguous run of embeddings per image.
        #
        #   2) tile-based (image_flags is not None)
        #      — static InternVL-style tiling with one flag per tile. Keeps
        #        backward compatibility with callers that stick with the
        #        checkpoint's bundled tile processor.
        #
        # When both are None (or pixel_values is None), we skip image
        # injection and run the LM path on text embeddings only.
        if not _embeds_pre_built and pixel_values is not None and imgs_sizes is not None:
            _embeds_shape = inputs_embeds.shape  # [B, N, C] or THD-flattened [T, C]
            C = _embeds_shape[-1]
            inputs_embeds = inputs_embeds.reshape(-1, C)
            input_ids_flat = input_ids.reshape(-1)
            selected = input_ids_flat == self.img_context_token_id

            # Vision/audio encoders are not CP-sharded; suspend the ring dispatcher
            # so their non-causal attention is not intercepted by the CP ring SDPA.
            with cp_dispatcher_suspended(self.cp_mesh):
                vit_embeds = self.extract_feature_dynamic(pixel_values, imgs_sizes)
            vit_embeds = _local_media(vit_embeds.reshape(-1, C), _global_image_mask, selected)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                logger.info(
                    f"NemotronOmni (dynamic-res): images={pixel_values.shape[0]}, "
                    f"imgs_sizes={imgs_sizes.tolist() if isinstance(imgs_sizes, torch.Tensor) else list(imgs_sizes)}, "
                    f"vit_embeds.shape={tuple(vit_embeds.shape)}, "
                    f"num_<image>_positions={int(selected.sum().item())}"
                )

            try:
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds
            except Exception as e:
                logger.warning(
                    f"Shape mismatch (dynamic-res): {e}, "
                    f"inputs_embeds[selected].shape={inputs_embeds[selected].shape}, "
                    f"vit_embeds.shape={vit_embeds.shape}"
                )
                n_token = int(selected.sum().item())
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds[:n_token]

            inputs_embeds = inputs_embeds.reshape(_embeds_shape)
        elif not _embeds_pre_built and pixel_values is not None:
            if image_flags is None:
                # Packed samples carry pixel_values without tile flags: every image is real.
                n_imgs = len(pixel_values) if isinstance(pixel_values, (list, tuple)) else pixel_values.shape[0]
                image_flags = torch.ones(n_imgs, dtype=torch.long, device=inputs_embeds.device)
            image_flags = image_flags.reshape(-1)

            _embeds_shape = inputs_embeds.shape  # [B, N, C] or THD-flattened [T, C]
            C = _embeds_shape[-1]
            B = _embeds_shape[0] if inputs_embeds.dim() == 3 else 1
            inputs_embeds = inputs_embeds.reshape(-1, C)
            input_ids_flat = input_ids.reshape(-1)

            selected = input_ids_flat == self.img_context_token_id

            # Mixed-resolution batches arrive as a list of per-image tensors
            # (they cannot be stacked), so there is no leading batch dim.
            pv_is_list = isinstance(pixel_values, (list, tuple))
            vit_batch_size = len(pixel_values) if pv_is_list else pixel_values.shape[0]
            with cp_dispatcher_suspended(self.cp_mesh):
                vit_embeds = self.extract_feature(pixel_values)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                logger.info(
                    f"NemotronOmni: dynamic ViT batch size: {vit_batch_size}, "
                    f"images per sample: {vit_batch_size / B}, "
                    f"tokens: {inputs_embeds.shape[0]}"
                )

            # Filter by image_flags (1 = real image, 0 = padding)
            if pv_is_list:
                # Token count per image varies with resolution, so the features
                # can't be stacked — select then concatenate along the token dim.
                kept = [e.reshape(-1, C) for e, flag in zip(vit_embeds, image_flags.tolist()) if flag == 1]
                vit_embeds = torch.cat(kept, dim=0) if kept else vit_embeds[0].new_zeros((0, C))
            else:
                vit_embeds = vit_embeds[image_flags == 1]
            vit_embeds = _local_media(vit_embeds.reshape(-1, C), _global_image_mask, selected)

            try:
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                logger.warning(
                    f"Shape mismatch: {e}, "
                    f"inputs_embeds[selected].shape={inputs_embeds[selected].shape}, "
                    f"vit_embeds.shape={vit_embeds.shape}"
                )
                n_token = selected.sum()
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds[:n_token]

            inputs_embeds = inputs_embeds.reshape(_embeds_shape)

        # Image and video both expand to `img_context_token_id` in the prompt, so a
        # single sample can carry only one of `pixel_values` / `pixel_values_videos`.
        if not _embeds_pre_built and pixel_values_videos is not None:
            assert pixel_values is None, "pixel_values and pixel_values_videos are mutually exclusive"
            _embeds_shape_v = inputs_embeds.shape
            C_v = _embeds_shape_v[-1]
            inputs_embeds = inputs_embeds.reshape(-1, C_v)
            video_selected = input_ids.reshape(-1) == self.img_context_token_id
            with cp_dispatcher_suspended(self.cp_mesh):
                video_embeds = self.extract_video_feature(pixel_values_videos)
            video_embeds = _local_media(video_embeds.reshape(-1, C_v), _global_image_mask, video_selected)
            inputs_embeds[video_selected] = inputs_embeds[video_selected] * 0.0 + video_embeds
            inputs_embeds = inputs_embeds.reshape(_embeds_shape_v)

        # --- Sound/audio token replacement ---
        has_sound = (
            not _embeds_pre_built
            and sound_features is not None
            and self.sound_encoder is not None
            and self.sound_context_token_id is not None
        )
        if has_sound:
            _embeds_shape_s = inputs_embeds.shape
            C_s = _embeds_shape_s[-1]
            inputs_embeds = inputs_embeds.reshape(-1, C_s)
            input_ids_flat_sound = input_ids.reshape(-1)

            sound_selected = input_ids_flat_sound == self.sound_context_token_id
            num_sound_tokens = sound_selected.sum().item()

            if num_sound_tokens > 0:
                # Move sound features to correct device/dtype
                target_dtype = inputs_embeds.dtype
                sound_features = sound_features.to(dtype=target_dtype, device=inputs_embeds.device)
                if sound_attention_mask is not None:
                    sound_attention_mask = sound_attention_mask.to(device=inputs_embeds.device)

                # Extract and project sound features
                with cp_dispatcher_suspended(self.cp_mesh):
                    sound_embeds = self.extract_sound_feature(sound_features, sound_attention_mask)
                sound_embeds_flat = _local_media(sound_embeds.reshape(-1, C_s), _global_sound_mask, sound_selected)

                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    logger.info(
                        f"NemotronOmni: sound tokens: {num_sound_tokens}, "
                        f"sound_embeds shape: {sound_embeds.shape}, "
                        f"sound_features shape: {sound_features.shape}"
                    )

                try:
                    inputs_embeds[sound_selected] = inputs_embeds[sound_selected] * 0.0 + sound_embeds_flat.to(
                        inputs_embeds.dtype
                    )
                except Exception as e:
                    logger.warning(
                        f"Sound shape mismatch: {e}, "
                        f"inputs_embeds[sound_selected].shape={inputs_embeds[sound_selected].shape}, "
                        f"sound_embeds_flat.shape={sound_embeds_flat.shape}"
                    )
                    inputs_embeds[sound_selected] = inputs_embeds[sound_selected] * 0.0 + sound_embeds_flat[
                        :num_sound_tokens
                    ].to(inputs_embeds.dtype)

                del sound_embeds, sound_embeds_flat

            inputs_embeds = inputs_embeds.reshape(_embeds_shape_s)

        # packed_sequence_thd_vlm_collater / packed_sequence_thd_collater emit
        # seq_lens/seq_lens_padded (not cu_seqlens): the generic TE-THD CP sharder
        # that would normally derive cu_seqlens (thd_utils.process_input_for_thd)
        # never runs for this model, since prepare_model_inputs_for_cp's aux-only
        # sharder takes priority once CP is active. Without cu_seqlens, TE's
        # DotProductAttention falls back to its constructor-default qkv_format
        # ("bshd") and rejects the THD-squeezed 2D/3D q/k/v downstream. Derive the
        # GLOBAL (pre-CP-shard) cu_seqlens here so TE's ring/P2P CP attention -
        # configured on this model's attention layers via MoE parallelizer's
        # apply_cp (block.self_attn) - gets real sequence-boundary metadata.
        if kwargs.get("qkv_format") == "thd" and "cu_seqlens" not in kwargs and "seq_lens" in kwargs:
            _seq_lens = kwargs.pop("seq_lens")
            _seq_lens_padded = kwargs.pop("seq_lens_padded", None)
            _seq_lens_flat = _seq_lens.reshape(-1)
            _valid_seq_lens = _seq_lens_flat[_seq_lens_flat != -1000].to(torch.int32)
            _cu_seqlens = torch.cat(
                [torch.zeros(1, dtype=torch.int32, device=_valid_seq_lens.device), _valid_seq_lens.cumsum(0)]
            ).to(torch.int32)
            kwargs["cu_seqlens"] = _cu_seqlens
            if _seq_lens_padded is not None:
                _seq_lens_padded_flat = _seq_lens_padded.reshape(-1)
                _valid_seq_lens_padded = _seq_lens_padded_flat[_seq_lens_padded_flat != -1000].to(torch.int32)
                _cu_seqlens_padded = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int32, device=_valid_seq_lens_padded.device),
                        _valid_seq_lens_padded.cumsum(0),
                    ]
                ).to(torch.int32)
                if not torch.equal(_cu_seqlens_padded, _cu_seqlens):
                    kwargs["cu_seqlens_padded"] = _cu_seqlens_padded
            if _cu_seqlens.numel() > 1:
                kwargs["max_seqlen"] = (_cu_seqlens[1:] - _cu_seqlens[:-1]).max().to(torch.int32)

        # Context-parallel: keep this rank's round-robin chunk pair of the freshly
        # embedded + spliced full sequence (aux streams aligned by
        # shard_batch_aux_only), so the LM shard matches the old dispatch-level
        # pre-embed and stays differentiable. Skipped when inputs_embeds is pre-sharded.
        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        if cp_size > 1 and not _embeds_pre_built:
            if not _thd_cp:
                # BSHD: ring-SDPA / all-gather CP uses the whole-row head-tail layout.
                # (Packed THD is already this rank's per-document shard: the framework
                # TE THD sharder partitioned input_ids before embedding.)
                inputs_embeds, _, _ = shard_sequence_for_cp_round_robin(self.cp_mesh, inputs_embeds, seq_dim=1)
            # shard_sequence_for_cp_round_robin only shards inputs_embeds; a THD/packed
            # attention_mask (still the pre-shard, full-length seq_idx/padding tensor)
            # would now mismatch the sharded sequence length. qkv_format="thd" (set by
            # packed_sequence_thd_vlm_collater / packed_sequence_thd_collater) plus
            # seq_lens/seq_lens_padded already fully describe packing structure, so drop
            # the stale mask rather than let it desync the THD-vs-BSHD branch in the
            # LLM's attention layers.
            if kwargs.get("qkv_format") == "thd" or "cu_seqlens" in kwargs or "cu_seqlens_q" in kwargs:
                attention_mask = None

        # Forward through the LLM. ``logits_to_keep`` gates the lm_head projection
        # (0 -> all positions; N -> last N) and ``output_hidden_states`` makes the
        # returned NemotronHCausalLMOutputWithPast carry the final, full-sequence
        # decoder hidden states (consumed by fused linear cross-entropy / cut-CE).
        kwargs.pop("seq_lens", None)
        kwargs.pop("seq_lens_padded", None)
        # Context-parallel MTP: the recipe supplies globally shifted per-depth token ids
        # (prepare_mtp_inputs_for_cp) sharded like the inputs. The LM receives
        # inputs_embeds from us, so hand it per-depth EMBEDDINGS instead (text-token
        # embeddings; media placeholders keep their placeholder embedding).
        mtp_per_depth_input_ids = kwargs.pop("mtp_per_depth_input_ids", None)
        mtp_embed_inputs: tuple = ()
        if mtp_per_depth_input_ids is not None:
            embed = self.language_model.get_input_embeddings()
            mtp_embed_inputs = tuple(embed(ids) for ids in mtp_per_depth_input_ids)
        lm_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,  # unused by the RoPE-free backbone; consumed by the MTP head
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        if mtp_embed_inputs:
            # NemotronV3 takes the per-depth MTP embeddings as positional varargs after input_ids.
            outputs = self.language_model(None, *mtp_embed_inputs, **lm_kwargs)
        else:
            outputs = self.language_model(input_ids=None, **lm_kwargs)  # inputs_embeds carry the tokens

        return outputs

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize model weights.

        Args:
            buffer_device: Device to use for buffer initialization
            dtype: Target dtype for model weights
        """
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            # Initialize LLM weights
            self.language_model.initialize_weights(buffer_device=buffer_device, dtype=dtype)

        # Vision model and projectors are loaded from checkpoint
        # Cast everything to target dtype
        cast_model_to_dtype(self, dtype, skip_modules=("_fp32_params",))


ModelClass = NemotronOmniForConditionalGeneration
