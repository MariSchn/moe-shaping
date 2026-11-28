# `lm/` — LM-scale MoE training

This directory scales the MoE training from the
toy `src/` experiments up to a real (small) Mixture-of-Experts **language model**,
trained on the [CSCS Alps](https://docs.cscs.ch/) cluster (GH200 nodes) on top of
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

It is adapted from the [`swiss-ai/lsaie-ss26-gipfelsturm`](https://github.com/swiss-ai/lsaie-ss26-gipfelsturm)
training harness: a thin SLURM launcher around Megatron-LM. Megatron already has
native MoE (router, top-k, experts, aux-loss / loss-free balancing), so the only
custom addition is a small **MoE model preset**.

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
Model dimensions are env-overridable (see the header of
`launch.sh`). Examples:

```bash
./launch.sh 50                                # quick smoke run
NUM_EXPERTS=16 HIDDEN=768 NUM_LAYERS=12 ./launch.sh 2000
```

The default model is a ~tiny top-2 MoE (8 layers, hidden 512, 8 experts) sized to
iterate quickly on a single node with pure data parallelism (TP=PP=1).

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
patch files in `patches/`, applied automatically by `launch.sh` before each run
(`git checkout -- . && git apply ../patches/*.patch`). Keep each patch isolated to
one concern, with a comment header documenting intent and how to relocate the code
if line numbers shift on a future Megatron version.

Verify a patch applies cleanly:

```bash
cd Megatron-LM && git apply --check ../patches/0001-log-tokens-per-sec-to-wandb.patch
```

### Current patches

| Patch | Description |
|-------|-------------|
| `0001-log-tokens-per-sec-to-wandb.patch` | Logs tokens/sec/GPU to stdout, TensorBoard, W&B |

## Dependencies

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) `core_v0.16.1` (git submodule)
