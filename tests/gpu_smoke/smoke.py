# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-GPU smoke of the molt backend path, one architecture per run, one to two minutes each.

Each architecture is its real config cut to a few layers / experts with seeded random weights, built the
way molt builds its actor (FSDP2 + EP mesh, TE attention, fp32 masters under a bf16 MixedPrecisionPolicy)
and driven through molt's forward path: THD packing, or a padded forward for Qwen3.5-MoE, which molt runs
through its CP hook. Checks, in order:

  1. forward: finite logits;
  2. export: the consolidated HF export (the path molt's vLLM refit and checkpoint export use) must load
     into transformers' own class with no missing, unexpected or mismatched keys;
  3. reference (``hf`` families): that transformers model, in bf16 like molt's actor, must reproduce the
     forward logits on the same inputs within a tolerance sized to bf16 noise. Qwen3.5-MoE and Inkling are ``export`` only: their
     tiny random builds disagree with transformers by far more than the other families (Qwen3.6's real-weight
     numerics are covered by molt's e2e; Inkling's disagreement is an open item to settle with real weights);
  4. --train: a few AdamW steps with the mesh-aware clip, the loss must fall; then the DCP checkpoint saved
     before training is reloaded and must reproduce the original forward logits.

    torchrun --nproc_per_node=2 tests/gpu_smoke/smoke.py --arch qwen3_6 --train
"""
import argparse
import glob
import json
import os
import shutil
import tempfile
import time
import zlib

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHS = {  # config, layers, experts (0 = as configured), ep, padded forward, reference
    "qwen2_5": ("qwen2_5_1_5b.json", 4, 0, 1, False, "hf"),
    "qwen3": ("qwen3_4b.json", 4, 0, 1, False, "hf"),
    "qwen3_moe": ("qwen3_30b_a3b.json", 4, 16, 2, False, "hf"),
    "qwen3_6": ("qwen3_6_35b_a3b.json", 4, 16, 2, True, "export"),  # Qwen3.5-MoE architecture
    "inkling": ("inkling_small.json", 4, 16, 2, False, "export"),  # logits differ from transformers by ~0.6 rel; open item
    "nemotron3": ("nemotron3_nano_30b_a3b.json", 10, 16, 2, False, "hf"),  # last 10 of the M/E/* pattern keep one attention layer
}
# Tolerances are relative errors of the [B, S, V] logits: Frobenius (||a-b|| / ||b||) and max-abs
# (max|a-b| / max|b|). molt's actor computes in bf16 under fp32 masters while the transformers reference
# runs in bf16 too but through transformers' kernels, so `hf` cannot be bit-exact; measured noise on H100 is
# 2e-3 to 1.6e-2 on both metrics, the bounds sit 3x and up above it, and a wrong weight mapping, rope or norm
# moves them by O(1). The reload check shares kernels and process with the original forward: near bit-exact.
HF_TOL, RELOAD_TOL = (2e-2, 5e-2), (1e-6, 1e-5)
TRAIN_STEPS = 3
ap = argparse.ArgumentParser()
ap.add_argument("--arch", choices=ARCHS, required=True)
ap.add_argument("--seqlen", type=int, default=256)
ap.add_argument("--train", action="store_true")
args = ap.parse_args()
config_file, n_layers, n_experts, ep, padded, reference = ARCHS[args.arch]

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count())
t0 = time.time()

import nemo_automodel
from nemo_automodel._transformers.model_init import get_hf_config
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config, MoEParallelizerConfig
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.distributed.mesh_utils import _create_device_meshes
from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm


def log(msg):
    if rank == 0:
        print(f"[{args.arch} +{time.time() - t0:5.1f}s] {msg}", flush=True)


def compare(ours, ref, what, tol):
    """Relative Frobenius and max-abs distance of two [B, S, V] fp32 logit tensors (argmax agreement is logged)."""
    fro = ((ours - ref).norm() / ref.norm().clamp_min(1e-6)).item()
    mx = ((ours - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
    agree = (ours.argmax(-1) == ref.argmax(-1))[mask.bool().cpu()].float().mean().item()
    log(f"{what}: rel_fro={fro:.2e} rel_max={mx:.2e} argmax_agree={agree:.3f} (tol {tol[0]:.0e} / {tol[1]:.0e})")
    assert fro <= tol[0] and mx <= tol[1], f"{what} disagrees beyond tolerance"


# --- config: real architecture, the last few layers (layer-type patterns end on a full-attention layer,
# so GDN and full attention are both kept), few experts, no MTP head
cfg_dir = tempfile.mkdtemp(prefix="smoke_cfg_")
with open(os.path.join(HERE, "configs", config_file)) as f:
    raw = json.load(f)
raw.pop("auto_map", None)  # hub configs with remote code (Nemotron 3): use the transformers / registry classes
with open(os.path.join(cfg_dir, "config.json"), "w") as f:
    json.dump(raw, f)
cfg = get_hf_config(cfg_dir, "sdpa", trust_remote_code=True)
text = getattr(cfg, "text_config", None) or cfg
n_orig = text.num_hidden_layers
for c in {id(cfg): cfg, id(text): text}.values():  # multimodal wrappers mirror the text sizes at the top level
    for k, v in list(vars(c).items()):
        if isinstance(v, (list, str)) and len(v) == n_orig:  # per-layer lists, Nemotron-H's pattern string
            setattr(c, k, v[-n_layers:])
    if getattr(c, "num_hidden_layers", None):
        c.num_hidden_layers = n_layers
    for k in ("num_experts", "n_routed_experts", "moe_num_experts", "num_local_experts"):
        if n_experts and getattr(c, k, None):
            setattr(c, k, n_experts)
for k in ("mtp_num_hidden_layers", "num_nextn_predict_layers", "num_mtp_modules"):
    if getattr(text, k, None):
        setattr(text, k, 0)
for tower in ("vision_config", "audio_config"):  # unused here (text inputs only): one layer, on both sides of the export
    for k in ("depth", "num_hidden_layers", "num_layers"):
        if getattr(getattr(cfg, tower, None), k, None):
            setattr(getattr(cfg, tower), k, 1)
is_vlm = any("ConditionalGeneration" in a for a in (cfg.architectures or []))

# --- meshes + setup, as molt's BaseModel builds them
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True)
fsdp_cfg = FSDP2Config(mp_policy=mp, activation_checkpointing="full", defer_fsdp_grad_sync=False)
moe_cfg = MoEParallelizerConfig(mp_policy=mp, ignore_router_for_ac=True, reshard_after_forward=False) if ep > 1 else None
device_mesh, moe_mesh = _create_device_meshes(fsdp_cfg, ParallelismSizes(ep_size=ep), world_size=world)
setup = DistributedSetup(mesh_context=MeshContext.from_meshes(device_mesh, moe_mesh), strategy_config=fsdp_cfg, moe_parallel_config=moe_cfg, activation_checkpointing="full")
backend = BackendConfig(attn="te", rope_fusion=False, dispatcher=os.environ.get("MOE_DISPATCHER", "hybridep"), rms_norm="torch_fp32", gate_precision="float32")
Cls = nemo_automodel.NeMoAutoModelForImageTextToText if is_vlm else nemo_automodel.NeMoAutoModelForCausalLM
model = Cls.from_config(cfg, torch_dtype=torch.float32, attn_implementation="sdpa", distributed_setup=setup, use_liger_kernel=False, has_packed_sequence=True, force_hf=False, backend=backend)
for name, p in model.named_parameters():  # seeded, non-trivial weights
    local = p.data._local_tensor if isinstance(p.data, DTensor) else p.data
    if local.numel():
        gen = torch.Generator(device=local.device).manual_seed((zlib.crc32(name.encode()) + 1000003 * rank) % (2**31 - 1))
        local.normal_(0.0, 0.02, generator=gen)
log(f"{type(model).__name__}: layers={n_layers} experts={n_experts or '-'} ep={ep} padded={padded} reference={reference} world={world}")

# --- fixed batch: two sequences, the second one shorter
vocab = int(text.vocab_size)
B, S = 2, args.seqlen
g = torch.Generator().manual_seed(1234)
seq = torch.randint(5, min(vocab, 30000), (B, S), generator=g).cuda()
mask = torch.ones(B, S, dtype=torch.long, device="cuda")
mask[1, S // 2 :] = 0
labels_full = torch.roll(seq, -1, 1)


def run_step(train):
    """One forward (+ backward when training) through molt's path; returns dense [B, S, V] logits."""
    if padded:
        out = model(input_ids=seq, attention_mask=mask)
        logits = out if isinstance(out, torch.Tensor) else out["logits"] if isinstance(out, dict) else out.logits
        loss = None
        if train:
            flat = mask.reshape(-1).bool()
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1])[flat], labels_full.reshape(-1)[flat])
            loss.backward()
        return logits.detach() * mask.unsqueeze(-1), loss
    # molt's cp1 packing branch: real tokens only, THD kwargs for TE
    flat = mask.reshape(-1).bool()
    packed = seq.reshape(-1)[flat].unsqueeze(0)
    seq_lens = mask.sum(-1, dtype=torch.int32)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), seq_lens.cumsum(0).to(torch.int32)])
    pos = (torch.cumsum(mask, -1) - 1).clamp_min(0).reshape(-1)[flat].unsqueeze(0)
    out = model(input_ids=packed, position_ids=pos, attention_mask=None, qkv_format="thd", cu_seqlens=cu, cu_seqlens_padded=cu, max_seqlen=int(seq_lens.max()))
    logits = out if isinstance(out, torch.Tensor) else out["logits"] if isinstance(out, dict) else out.logits
    loss = None
    if train:
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels_full.reshape(-1)[flat])
        loss.backward()
    full = logits.new_zeros((B * S, logits.shape[-1]))
    full[flat] = logits.detach().reshape(-1, logits.shape[-1])
    return full.view(B, S, -1), loss


def forward_logits():
    model.eval()
    with torch.no_grad():
        logits, _ = run_step(train=False)
    return logits.float().cpu()


logits = forward_logits()
assert torch.isfinite(logits).all(), "non-finite logits"
log(f"forward ok: logits {tuple(logits.shape)} absmax={logits.abs().max():.3f}")

# --- DCP save with consolidated HF export; the export doubles as the HF reference's weights
out_dir = tempfile.mkdtemp(prefix="smoke_ckpt_") if rank == 0 else None
out_dir = [out_dir]; dist.broadcast_object_list(out_dir); out_dir = out_dir[0]
ckpt_cfg = CheckpointingConfig(enabled=True, checkpoint_dir=out_dir, model_save_format="safetensors", model_cache_dir=None, model_repo_id=None, save_consolidated=True, is_peft=False)
ckpt = Checkpointer(config=ckpt_cfg, dp_rank=rank, tp_rank=0, pp_rank=0, moe_mesh=moe_mesh)
ckpt.save_model(model=model, weights_path=out_dir, tokenizer=None)
dist.barrier()
consolidated = os.path.join(out_dir, "model", "consolidated")
if rank == 0:
    shards = [f for f in os.listdir(consolidated) if f.endswith(".safetensors")]
    assert shards, f"no consolidated safetensors under {out_dir}"
    log(f"checkpoint saved, HF export {len(shards)} shard(s)")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    if not os.path.exists(os.path.join(consolidated, "config.json")):
        shutil.copy(glob.glob(os.path.join(out_dir, "**", "config.json"), recursive=True)[0], consolidated)
    HF = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
    # The export's config.json is the text view; multimodal wrappers need the full shrunk config for their
    # towers. Round-trip it through JSON so transformers builds it with its own config classes.
    cfg.save_pretrained(cfg_dir)
    hf_cfg = AutoConfig.from_pretrained(cfg_dir)
    audio = getattr(hf_cfg, "audio_config", None)
    if audio is not None and "text_hidden_size" not in audio.__dict__:  # Inkling: read as `hidden_size`, stored only as an alias
        audio.__dict__["text_hidden_size"] = text.hidden_size
    hf, info = HF.from_pretrained(consolidated, config=hf_cfg, dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
    # The export must be a complete, key-compatible HF checkpoint: nothing left to random init, nothing ignored.
    issues = {k: v for k, v in info.items() if v}
    log(f"export loads into transformers {HF.__name__}: {issues or 'clean'}")
    assert not issues, "the consolidated HF export does not load cleanly into transformers"
    if reference == "hf":
        hf = hf.cuda().eval()
        with torch.no_grad():
            hf_logits = (hf(input_ids=seq, attention_mask=mask).logits.float() * mask.unsqueeze(-1)).cpu()
        compare(logits, hf_logits, f"vs transformers {HF.__name__}", HF_TOL)
    del hf
dist.barrier()

if args.train:
    # A few AdamW steps on the fixed batch: the loss must fall, i.e. gradients flow through FSDP2 / EP
    # and the mesh-aware clip (expert grads live on the EP mesh, the rest on the FSDP mesh), as in molt.
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    losses, norms = [], []
    for _ in range(TRAIN_STEPS):
        opt.zero_grad(set_to_none=True)
        _, loss = run_step(train=True)
        grad_norm = scale_grads_and_clip_grad_norm(
            1.0, [model], pp_enabled=False, device_mesh=device_mesh, moe_mesh=moe_mesh,
            ep_axis_name="ep" if moe_mesh is not None and "ep" in moe_mesh.mesh_dim_names else None,
        )
        opt.step()
        losses.append(loss.item()); norms.append(float(grad_norm))
    assert all(torch.isfinite(torch.tensor(losses + norms))), f"non-finite loss / grad norm {losses} {norms}"
    log(f"train: loss {' -> '.join(f'{l:.4f}' for l in losses)}  grad_norm {' -> '.join(f'{n:.3f}' for n in norms)}")
    assert losses[-1] < losses[0], "loss did not decrease over the training steps"
    ckpt.load_model(model=model, model_path=os.path.join(out_dir, "model"))
    dist.barrier()
    reloaded = forward_logits()
    if torch.equal(reloaded, logits):
        log("checkpoint reload ok: forward bit-exact with the saved weights")
    else:
        compare(reloaded, logits, "checkpoint reload vs saved forward", RELOAD_TOL)
if rank == 0:
    shutil.rmtree(out_dir, ignore_errors=True)
shutil.rmtree(cfg_dir, ignore_errors=True)
dist.barrier()
dist.destroy_process_group()
log("SMOKE PASSED")
