#!/bin/bash
#SBATCH --job-name=build-domain-mix
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

OUT=/iopsstor/scratch/cscs/$USER/moe-shaping/domain_mix
export HF_HOME=/iopsstor/scratch/cscs/$USER/moe-shaping/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
cd $SLURM_SUBMIT_DIR
uv run --no-project --python 3.11 --with datasets --with tiktoken --with numpy \
    python data/build_domain_mix.py --out $OUT \
    --eval-manifest /iopsstor/scratch/cscs/$USER/moe-shaping/domain_eval/manifest.json "$@"
