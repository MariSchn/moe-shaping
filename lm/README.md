# `lm/` — LM-scale MoE routing experiments

This directory scales the MoE routing procedures from the toy `src/` experiments
(bilevel / alternating optimization, and bandit-trained routing) up to a real
(small) Mixture-of-Experts **language model**,
trained on the [CSCS Alps](https://docs.cscs.ch/) cluster (GH200 nodes) on top of
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

It is adapted from the [`swiss-ai/lsaie-ss26-gipfelsturm`](https://github.com/swiss-ai/lsaie-ss26-gipfelsturm)
training harness: a thin SLURM launcher around Megatron-LM. Megatron already has
native MoE (router, top-k, experts, aux-loss / loss-free balancing), so the
custom additions are a small **MoE model preset**, a **bilevel-alternation
patch** that alternates optimizer updates between router and expert parameters,
and a **bandit-router patch** that trains the router from the per-token loss as a
reward instead of from the task gradient.

## What "bilevel alternation" means here

The optimizer alternates between two phases over a fixed cycle:

- **router phase** (`--bilevel-router-steps` steps): only the MoE router
  parameters (`*.router.*`) are updated; expert parameters are frozen.
- **expert phase** (`--bilevel-expert-steps` steps): only the MoE expert
  parameters (`*.experts.*`) are updated; router parameters are frozen.

The transformer backbone (attention, embeddings, norms, lm-head) trains every
step. Freezing is implemented by **masking the inactive param group's learning
rate to 0** (router/expert params are placed in their own optimizer param groups).
Setting `--bilevel-router-steps 0` disables alternation entirely (joint training
baseline). See `patches/` for details.

## Setup

**1. Configure your paths:**

```bash
cp config.sh.example config.sh
# Edit: SBATCH_ACCOUNT, WANDB_API_KEY (optional)
```

`config.sh` is git-ignored; the example is committed as a template. `WORKDIR` is
auto-derived from the launcher's location, so you only set it to run Megatron
from a different checkout (e.g. a scratch copy).

**2. Initialize the Megatron-LM submodule** (pinned to `core_v0.16.1`):

```bash
git submodule update --init lm/Megatron-LM   # run from the repo root
```

**3. Set up the EDF container environment** (copy `alps3.toml` to `~/.edf/`):

```bash
mkdir -p ~/.edf
sed "s|workdir = .*|workdir = \"$HOME\"|" alps3.toml > ~/.edf/alps3.toml
```

## Running

```bash
./launch.sh [steps] [nodes]          # default: 1000 steps, 1 node (4 GH200)
```

The launcher generates a self-contained SLURM script in `logs/` and submits it.
Model dimensions and bilevel knobs are env-overridable (see the header of
`launch.sh`). Examples:

```bash
./launch.sh 50                                # quick smoke run
BILEVEL=0 ./launch.sh 1000                    # joint-training baseline (no alternation)
ROUTER_LR=1e-3 EXPERT_LR=3e-4 ./launch.sh 2000
NUM_EXPERTS=16 HIDDEN=768 NUM_LAYERS=12 ./launch.sh 2000
```

`BILEVEL=0` runs plain Megatron MoE training (no `--bilevel-*` flags, no
router/expert param-group split). Bilevel and baseline runs get distinct
job/log/W&B names (tag `bl-r<R>e<E>` vs `joint`) so they don't clobber each other.

The default model is a ~tiny top-2 MoE (8 layers, hidden 512, 8 experts) sized to
iterate quickly on a single node with pure data parallelism (TP=PP=1).

## Bandit-trained routing

`BANDIT=1` blocks the task gradient into the router and trains it from bandit
feedback instead: the per-token LM loss, negated, is the reward for every routing
decision that token passed through. See `patches/0006-bandit-router.patch` for
the mechanism and `megatron/core/transformer/moe/bandit.py` inside a snapshot for
the code.

```bash
BANDIT=1 ./launch.sh 3000                                  # pure bandit routing
BANDIT=1 BANDIT_BASELINE=batch_mean ./launch.sh 3000        # cheaper baseline
BANDIT=1 BANDIT_KEEP_TASK_GRAD=1 BANDIT_GRAD_ALIGN_INTERVAL=25 ./launch.sh 3000
```

Knobs: `BANDIT_BASELINE` (`critic` | `batch_mean` | `none`), `BANDIT_EXPLORE`
(`gumbel` | `epsilon` | `none`) with `BANDIT_TAU`/`BANDIT_TAU_FINAL` or
`BANDIT_EPSILON`/`BANDIT_EPSILON_FINAL`, `BANDIT_COEFF`, `BANDIT_ADV_NORM`,
`BANDIT_ENTROPY`, `BANDIT_CRITIC_COEFF`, `BANDIT_KEEP_TASK_GRAD`,
`BANDIT_GRAD_ALIGN_INTERVAL`.

Note that Adam is invariant to the scale of a parameter's gradient, and in a pure
bandit run the REINFORCE term is the router's *only* gradient — so `BANDIT_COEFF`
barely changes the router's step size there, and `ROUTER_LR` is the real
step-size knob. Holding `LR` fixed with no `ROUTER_LR` gives every arm the same
per-step router movement and differs only in its direction, which is what makes
the arms comparable.

`ROUTER_LR=0` freezes the router at its initialization — the control arm for "how
much does learning the routing matter at all".

## Experiment grids

`sweeps/` holds the grids used for the bandit study, and `analyze_runs.py`
collects their metrics (per-iteration series and validation losses from the SLURM
stdout logs, plus the signed `bandit/*` diagnostics from the TensorBoard event
files, which the stdout line drops).

```bash
./sweeps/stage1.sh              # 800 steps, one seed: pick a bandit config
./sweeps/stage2_baselines.sh    # 3000 steps x 3 seeds: noLB / aux / auxfree / frozen
BANDIT_BASELINE=critic ./sweeps/stage2_bandit.sh
python3.11 analyze_runs.py 's2-*' --out results/s2.json
```

## Container image

**alps3** extended image (NGC PyTorch 26.01-py3): includes a patched NCCL,
libfabric, OpenMPI, nvshmem. A working EDF env is in `alps3.toml`.

## Dataset

[Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix)
`climbmix_small` subset, pre-tokenized with the GPT-2 BPE tokenizer
(`data/gpt2-vocab.json`, `data/gpt2-merges.txt`). Already converted to Megatron's
binary format on capstor:

```
/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small.{bin,idx}
```

To re-download / re-convert, see `data/download_climbmix.sh` and
`data/convert_data.sbatch`.

> Checkpointing is currently disabled due to a [known SIGSEGV bug](https://github.com/NVIDIA/Megatron-LM/issues/1861)
> on GH200/ARM64; rely on in-run eval (`--eval-interval`) rather than resume.

## Megatron-LM patches

Megatron-LM is a git submodule pinned to a release. Local modifications live as
patch files in `patches/`. Keep each patch isolated to one concern, with a comment
header documenting intent and how to relocate the code if line numbers shift on a
future Megatron version.

`launch.sh` applies the whole set **once, on the login node**, into an immutable
snapshot at `.megatron-snapshots/<hash>` (hash = submodule commit + patch
contents) and points the job at that, so a submitted run keeps exactly the source
it was launched with and concurrent jobs cannot clobber each other. The builder
`git init`s the staging tree before applying — `git apply` run from a subdirectory
of a git work tree silently ignores patched paths outside that subdirectory, which
otherwise yields a snapshot of *unpatched* Megatron with a zero exit code — and
then refuses to publish the snapshot unless every file the patches name actually
changed.

Since patches build on each other, check them as a set rather than one at a time:

```bash
cd Megatron-LM && git checkout -- . && git apply ../patches/*.patch && git checkout -- .
```

### Current patches

| Patch | Description |
|-------|-------------|
| `0001-log-tokens-per-sec-to-wandb.patch` | Logs tokens/sec/GPU to stdout, TensorBoard, W&B |
| `0002-bilevel-alternation.patch` | Router/expert param-group split + per-phase LR masking; adds `--bilevel-*` / `--router-lr` / `--expert-lr` args; logs `bilevel/phase` |
| `0003-log-expert-imbalance.patch` | Always logs `expert_imbalance` (raw load-balance loss, monitor-only) even when the aux-loss coefficient is 0 |
| `0004-log-expert-load.patch` | Logs the per-expert load distribution as `load/min`, `load/max` and `load/p10..p90` |
| `0005-log-num-parameters-to-wandb.patch` | Logs the total model parameter count to W&B at startup |
| `0006-bandit-router.patch` | Trains the router from bandit feedback (REINFORCE) instead of the task gradient; adds `--bandit-*` args and `router/entropy` |

## Dependencies

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) `core_v0.16.1` (git submodule)
