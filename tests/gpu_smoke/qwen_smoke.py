# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-GPU smoke of the molt backend path on Qwen architectures, a few minutes end to end.

Each architecture is the real config cut to a few layers / experts with seeded random weights, built the
way molt builds its actor (FSDP2 + EP / CP mesh, TE attention, fp32 masters under a bf16
MixedPrecisionPolicy) and driven through molt's forward path:

  moe      Qwen3-MoE      EP2, THD packing (cu_seqlens)
  dense    Qwen3          FSDP2, THD packing
  qwen3_6  Qwen3.5-MoE    EP2, padded forward (molt drives this family through its CP hook, which
                          needs cp >= 2 on top of EP and does not fit two GPUs)

``--out`` dumps the forward logits; ``--ref`` compares them bit for bit with a previous dump (run the base
branch first, then the PR). ``--train`` additionally runs backward + optimizer step and a DCP save /
reload / consolidated-HF-export round trip.

    torchrun --nproc_per_node=2 tests/gpu_smoke/qwen_smoke.py --arch moe --train --out /tmp/moe.pt
"""
import argparse
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
ARCHS = {  # config, layers, experts, ep, padded
    "moe": ("qwen3_30b_a3b.json", 4, 16, 2, False),
    "dense": ("qwen3_4b.json", 4, 0, 1, False),
    "qwen3_6": ("qwen3_6_35b_a3b.json", 4, 16, 2, True),
}
ap = argparse.ArgumentParser()
ap.add_argument("--arch", choices=ARCHS, required=True)
ap.add_argument("--seqlen", type=int, default=256)
ap.add_argument("--out"); ap.add_argument("--ref"); ap.add_argument("--train", action="store_true")
args = ap.parse_args()
config_file, n_layers, n_experts, ep, padded = ARCHS[args.arch]
cp = 1

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


# --- config: real architecture, few layers / experts, no MTP head
cfg_dir = tempfile.mkdtemp(prefix="smoke_cfg_")
shutil.copy(os.path.join(HERE, "configs", config_file), os.path.join(cfg_dir, "config.json"))
cfg = get_hf_config(cfg_dir, "sdpa", trust_remote_code=True)
text = getattr(cfg, "text_config", None) or cfg
n_orig = text.num_hidden_layers
for c in {id(cfg): cfg, id(text): text}.values():
    for k, v in list(vars(c).items()):
        if isinstance(v, list) and len(v) == n_orig:
            setattr(c, k, v[:n_layers])
text.num_hidden_layers = n_layers
if n_experts and getattr(text, "num_experts", None):
    text.num_experts = n_experts
for k in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
    if getattr(text, k, None):
        setattr(text, k, 0)
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
log(f"{type(model).__name__}: layers={n_layers} experts={n_experts or '-'} ep={ep} padded={padded} world={world}")

# --- fixed batch: two sequences, the second one shorter
vocab = int(text.vocab_size)
B, S = 2, args.seqlen
g = torch.Generator().manual_seed(1234)
seq = torch.randint(5, min(vocab, 30000), (B, S), generator=g).cuda()
mask = torch.ones(B, S, dtype=torch.long, device="cuda")
mask[1, S // 2 :] = 0
labels_full = torch.roll(seq, -1, 1)
model.train() if args.train else model.eval()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5) if args.train else None


def run_step():
    """One forward (+ backward with --train) through molt's path; returns dense [B, S, V] logits."""
    if padded:
        out = model(input_ids=seq, attention_mask=mask)
        logits = out if isinstance(out, torch.Tensor) else out["logits"] if isinstance(out, dict) else out.logits
        loss = None
        if args.train:
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
    if args.train:
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels_full.reshape(-1)[flat])
        loss.backward()
    full = logits.new_zeros((B * S, logits.shape[-1]))
    full[flat] = logits.detach().reshape(-1, logits.shape[-1])
    return full.view(B, S, -1), loss


with torch.enable_grad() if args.train else torch.no_grad():
    logits, loss = run_step()
logits = logits.float().cpu()
assert torch.isfinite(logits).all(), "non-finite logits"
log(f"forward ok: logits {tuple(logits.shape)} absmax={logits.abs().max():.3f}")
if rank == 0 and args.out:
    torch.save(logits, args.out)
if args.ref:
    ref = torch.load(args.ref)
    same = torch.equal(ref, logits)
    if rank == 0:
        d = (ref - logits).abs()
        log(f"parity vs {args.ref}: bit_exact={same} max_abs_diff={d.max().item():.3e}")
    assert same, "logits differ from the reference dump"

if args.train:
    # molt's clip: mesh-aware (expert grads live on the EP mesh, the rest on the FSDP mesh)
    grad_norm = scale_grads_and_clip_grad_norm(
        1.0, [model], pp_enabled=False, device_mesh=device_mesh, moe_mesh=moe_mesh,
        ep_axis_name="ep" if moe_mesh is not None and "ep" in moe_mesh.mesh_dim_names else None,
    )
    opt.step()
    assert torch.isfinite(loss).item(), f"non-finite loss {loss.item()}"
    log(f"train step ok: loss={loss.item():.4f} grad_norm={float(grad_norm):.4f}")
    out_dir = tempfile.mkdtemp(prefix="smoke_ckpt_") if rank == 0 else None
    out_dir = [out_dir]; dist.broadcast_object_list(out_dir); out_dir = out_dir[0]
    ckpt_cfg = CheckpointingConfig(enabled=True, checkpoint_dir=out_dir, model_save_format="safetensors", model_cache_dir=None, model_repo_id=None, save_consolidated=True, is_peft=False)
    ckpt = Checkpointer(config=ckpt_cfg, dp_rank=rank, tp_rank=0, pp_rank=0, moe_mesh=moe_mesh)
    ckpt.save_model(model=model, weights_path=out_dir, tokenizer=None)
    dist.barrier()
    ckpt.load_model(model=model, model_path=os.path.join(out_dir, "model"))
    dist.barrier()
    if rank == 0:
        shards = [f for f in os.listdir(os.path.join(out_dir, "model", "consolidated")) if f.endswith(".safetensors")]
        assert shards, f"no consolidated safetensors under {out_dir}"
        log(f"checkpoint save / reload / export ok: {len(shards)} shard(s)")
        shutil.rmtree(out_dir, ignore_errors=True)
shutil.rmtree(cfg_dir, ignore_errors=True)
dist.barrier()
dist.destroy_process_group()
log("SMOKE PASSED")
