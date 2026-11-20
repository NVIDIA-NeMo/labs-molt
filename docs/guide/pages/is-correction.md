# IS correction

In asynchronous training the trainer recomputes a token's probability with newer weights and
different kernels than the engine that sampled it. Molt corrects the resulting off-policy update
with the per-token importance ratio `pi_train / pi_rollout`. Three flags control it.

> Source: [`molt/models/loss.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/models/loss.py) · [`molt/trainer/algorithm/advantage.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/trainer/algorithm/advantage.py)

## The three knobs

| Flag | Values | Meaning |
|---|---|---|
| `--algo.advantage.is_correction_level` | `off`, `token`, `seq`, `geo` | What the gate looks at. `token` gates each token's own ratio. `seq` and `geo` aggregate a sequence's ratios, as `exp(sum)` and `exp(mean)`, into a filter that rejects whole sequences. |
| `--algo.advantage.is_correction_mode` | `mask`, `clip`, `trunc` | What happens outside the band. `mask` drops the unit, `clip` clamps its weight into the band, `trunc` clamps only the upper tail. |
| `--algo.advantage.is_correction_threshold` | `LOW HIGH` or `HIGH` | The band on the ratio. A single value is an upper bound only. The recipes use a tight `0.99 1.01`. |

`seq` and `geo` require `mode mask`. Sequences that pass still carry their per-token weight.

## Named schemes

| Flags | Scheme |
|---|---|
| `level token`, `mode trunc` | Truncated importance sampling (TIS) |
| `level token`, `mode mask` | Token masking (IcePop) |
| `level token`, `mode clip` | Per-token weight clamp |
| `level geo`, `mode mask` | Sequence masking by geometric mean, the recipe default |
| `level seq`, `mode mask` | Sequence masking by the product of ratios |
| `level token`, threshold `inf` | Unclipped token-level importance sampling, the REINFORCE Zero recipe |

## Score centering

Score centering is an alternative that needs no ratio.

```bash
--algo.advantage.score_centering_top_k 32 --algo.advantage.is_correction_level off
```

**The problem.** The engine and the trainer are two implementations of one model, so they
disagree slightly on every token's probability. Plain REINFORCE then nudges the trainer toward
the engine's preferences at every token, even when the reward is zero. That nudge is the drift.

**Where it comes from.** At one position the engine picked token `y` from its distribution `q`,
and the trainer assigns its own distribution `p`. The update is `advantage × ∇log p_y`. Averaged
over the engine's sampling:

```text
E_{y~q}[∇log p_y]  =  Σ_v q_v ∇log p_v
```

If `q = p` this is zero. When they differ it is not, and every update carries a push that does
not depend on the reward.

**The fix.** Subtract the average push before the advantage is applied:

```text
train on   log p_y  −  Σ_v q_v log p_v      instead of   log p_y
```

The expected update at every position becomes zero and no sample is thrown away. The engine
reports its top-k log-probs per token, so the sum runs over those `k` tokens. `k = 32` recovers
about 95% of the mass.

## Combine them

| You want | Flags |
|---|---|
| Score centering alone | `--algo.advantage.score_centering_top_k 32 --algo.advantage.is_correction_level off` |
| Score centering with TIS | add `--algo.advantage.is_correction_level token --algo.advantage.is_correction_mode trunc` |
| Score centering with IcePop | add `--algo.advantage.is_correction_level token --algo.advantage.is_correction_mode mask` |
| Score centering with reverse KL to the rollout policy | add `--algo.kl.use_loss --algo.kl.estimator k2 --algo.kl.init_coef 0.001` |
| The FlashREINFORCE gate | keep `level seq` or `geo` and do not add score centering |

Score centering cannot be stacked on a sequence-level gate. With the exact ratio the centering
term is identically zero, so `PolicyLoss` rejects the pair.

## FlashREINFORCE

FlashREINFORCE applies the exact ratio to every surviving token and drops whole responses whose
mismatch is too large. Importance sampling already removes the drift, and the gate only bounds
the variance.

```bash
--train.force_on_policy \
--algo.advantage.estimator flash_reinforce \
--algo.advantage.is_correction_level seq \
--algo.advantage.is_correction_gating binary_kl \
--algo.advantage.is_correction_threshold 0.05 \
--actor.loss_agg_mode seq-mean-token-mean
```

Start from `0.05`, which works well in most settings. `examples/scripts/quick_start/rl_flash_reinforce_r1d_1p5b.sh`
runs it on one node with the tighter `5e-3` that the FlashREINFORCE paper uses for that model.

## REINFORCE Zero

[REINFORCE Zero](https://github.com/hijkzzz/REINFORCE-Zero) is the asynchronous recipe built on
Molt. Rollout and trainer disagree for two reasons: older weights and different GPU kernels. The
recipe aligns the kernels and corrects the weight lag with unclipped token-level importance
sampling.

```bash
--algo.advantage.estimator dr_grpo \
--algo.advantage.is_correction_level token \
--algo.advantage.is_correction_threshold inf \
--actor.loss_agg_mode seq-mean-token-mean \
--train.force_on_policy
```

A threshold of `inf` leaves the band open: no token is masked or clipped, and each one carries its
exact ratio.

The full recipe also turns on `--train.batch_invariant`, which runs the engine and the trainer on
the same batch-invariant kernels; the REINFORCE Zero runs use it. Those kernels are still being
upstreamed and the flag arrives with them, so `main` does not accept it yet and the command above
corrects only the weight lag. `main` already ships the metrics, rollout dump and replay, and the
kernel-alignment agent skill,
[`.agents/skills/train-inference-consistency`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/.agents/skills/train-inference-consistency/SKILL.md).

## References

- **TIS**: truncated importance sampling of the train and rollout ratio.
- **IcePop**: token-level masking of ratios outside the band.
- **MIS**: masked importance sampling at the sequence level (Yingru Li), which motivates the
  `seq` and `geo` filters.
- **Score Centering**: [arXiv 2609.20807](https://arxiv.org/abs/2609.20807).
