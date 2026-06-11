#!/bin/bash
#SBATCH --job-name=parquet-to-megatron
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/convert_data_%j.log
#SBATCH --error=logs/convert_data_%j.log

WORKDIR=${WORKDIR:-$SLURM_SUBMIT_DIR}
INPUT=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small
OUTPUT=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small

mkdir -p logs

srun --mpi=pmix --network=disable_rdzv_get --environment=alps3 \
    bash -c "
    echo '=== Node: \$(hostname) ==='
    echo '=== Starting Parquet → Megatron conversion ==='
    python3 ${WORKDIR}/data/parquet_to_megatron.py \
        --input ${INPUT} \
        --output-prefix ${OUTPUT}
    echo '=== Conversion complete ==='
"
