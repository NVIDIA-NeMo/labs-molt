# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Tiny-config numerical parity check: NeMo Inkling MoE vs HuggingFace Inkling.

Builds a small 4-layer Inkling config (2 dense + 2 sparse MoE layers), loads the
same random weights into both implementations via the state-dict adapter, runs a
forward pass on CPU/float32, and compares the language-model logits with an
allclose check plus a KL divergence, which must be at the 1e-3 level.

Run directly:  python tests/unit_tests/models/inkling/parity_check_inkling.py
"""

import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling.configuration import InklingConfig
from nemo_automodel.components.models.inkling.model import InklingForConditionalGeneration


def build_tiny_config() -> InklingConfig:
    text = dict(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=16,
        sliding_window_size=8,
        d_rel=4,
        rel_extent=16,
        vocab_size=128,
        unpadded_vocab_size=None,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=2,
        route_scale=8.0,
        dense_intermediate_size=96,
        dense_mlp_idx=2,
        conv_kernel_size=4,
        max_position_embeddings=256,
        logits_mup_width_multiplier=4.0,
    )
    vision = dict(patch_size=8, temporal_patch_size=2, num_channels=3, n_layers=2)
    audio = dict(n_mel_bins=8, mel_vocab_size=16)
    return InklingConfig(
        text_config=text,
        vision_config=vision,
        audio_config=audio,
        image_token_id=126,
        audio_token_id=127,
        torch_dtype="float32",
        _attn_implementation="eager",
    )


def main() -> None:
    try:
        from transformers.models.inkling.configuration_inkling import InklingConfig as HFInklingConfig
        from transformers.models.inkling.modeling_inkling import (
            InklingForConditionalGeneration as HFInklingForConditionalGeneration,
        )
    except ImportError as exc:
        raise RuntimeError("The development parity script requires Transformers >=5.14 as a reference") from exc

    torch.manual_seed(0)
    device, dtype = torch.device("cpu"), torch.float32
    cfg = build_tiny_config()

    hf_config = HFInklingConfig.from_dict(cfg.to_dict())
    hf_model = HFInklingForConditionalGeneration(hf_config).to(device=device, dtype=dtype).eval()
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch")
    nemo_model = InklingForConditionalGeneration.from_config(cfg, backend=backend).to(device=device, dtype=dtype).eval()

    # --- Level 1: state-dict adapter round-trip -------------------------------
    adapter = nemo_model.state_dict_adapter
    hf_sd = hf_model.state_dict()
    native_sd = adapter.from_hf(hf_sd)
    raw_sd = adapter.to_hf(native_sd)
    roundtrip = adapter.from_hf(raw_sd)
    assert set(roundtrip.keys()) == set(native_sd.keys()), (
        f"round-trip key mismatch\n  missing: {set(native_sd) - set(roundtrip)}"
        f"\n  extra: {set(roundtrip) - set(native_sd)}"
    )
    max_rt = max((native_sd[k] - roundtrip[k]).abs().max().item() for k in native_sd)
    assert max_rt == 0.0, f"round-trip not exact: max_diff={max_rt}"
    print(f"[L1] state-dict adapter round-trip exact (max_diff={max_rt}).")

    # --- Load HF weights into the NeMo model ----------------------------------
    missing, unexpected = nemo_model.load_state_dict(native_sd, strict=False)
    missing = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
    unexpected = [k for k in unexpected if "rotary" not in k and "inv_freq" not in k]
    assert not missing, f"missing keys when loading into NeMo model: {missing}"
    assert not unexpected, f"unexpected keys when loading into NeMo model: {unexpected}"
    print("[load] HF weights loaded into NeMo model (no missing/unexpected keys).")

    # --- Level 3: forward logit parity ----------------------------------------
    input_ids = torch.randint(0, cfg.text_config.vocab_size, (1, 24), device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        nemo_logits = nemo_model(input_ids=input_ids, attention_mask=attention_mask).logits.float()

    assert hf_logits.shape == nemo_logits.shape, (hf_logits.shape, nemo_logits.shape)
    max_diff = (hf_logits - nemo_logits).abs().max().item()

    hf_logp = F.log_softmax(hf_logits, dim=-1)
    nemo_logp = F.log_softmax(nemo_logits, dim=-1)
    # KL(HF || NeMo), averaged over tokens.
    kl = F.kl_div(nemo_logp, hf_logp, log_target=True, reduction="batchmean").item()

    print(f"[L3] logits max_abs_diff={max_diff:.3e}  KL(HF||NeMo)={kl:.3e}")
    assert kl < 1e-3, f"KL divergence too high: {kl}"
    assert max_diff < 1e-2, f"max abs logit diff too high: {max_diff}"
    print("PARITY PASSED.")


if __name__ == "__main__":
    main()
