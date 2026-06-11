#!/bin/bash
#
# Train the small bilevel-alternation MoE on CSCS Alps (default: 1 GH200 node).
#
# Usage: ./launch.sh [steps] [nodes]
#   steps   training iterations            (default 1000)
#   nodes   number of nodes, 4 GPUs each   (default 1)
#
# Everything else is env-overridable (set inline or in config.sh):
#   Model:   NUM_LAYERS HIDDEN FFN HEADS KV_HEADS NUM_EXPERTS MBS GBS SEQ_LEN
#            (defaults below define a ~tiny top-2 MoE; per-expert FFN = FFN)
#   Bilevel: BILEVEL=1|0            master switch (default 1). BILEVEL=0 = joint
#                                   baseline: no --bilevel-* flags, plain MoE.
#            BILEVEL_ROUTER_STEPS  router-only update steps per cycle (default 1)
#            BILEVEL_EXPERT_STEPS  expert-only update steps per cycle (default 1)
#            ROUTER_LR / EXPERT_LR per-phase LRs (default: --lr for both)
#   Schedule/eval: LR EVAL_INTERVAL EVAL_ITERS LR_WARMUP_ITERS TIME
#   Cluster: SBATCH_ACCOUNT PARTITION (empty = cluster default, e.g. PARTITION=debug)
#
# Baseline vs bilevel runs get distinct job/log/W&B names (RUN_TAG: joint vs
# bl-r<R>e<E>), so their TensorBoard dirs and runs do not clobber each other.
#
# Examples:
#   ./launch.sh                                  # 1000 steps, 1 node, alternating
#   ./launch.sh 50                               # quick 50-step smoke run
#   BILEVEL=0 ./launch.sh 1000                   # joint-training baseline
#   NUM_EXPERTS=16 HIDDEN=768 ./launch.sh 2000   # bigger model

set -euo pipefail

WORKDIR=${WORKDIR:-/users/smarian/projects/moe-shaping/lm}
[ -f "$WORKDIR/config.sh" ] && source "$WORKDIR/config.sh"

# ---- Cluster + W&B config (defaults; override via env.sh or env) ----
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-aa004}
PARTITION=${PARTITION:-normal}
export WANDB_PROJECT=${WANDB_PROJECT:-moe-shaping-clariden}
export WANDB_ENTITY=${WANDB_ENTITY:-mari-schn}

STEPS=${1:-1000}
NODES=${2:-1}

# ---- Model ----
NUM_LAYERS=${NUM_LAYERS:-8}
HIDDEN=${HIDDEN:-512}
FFN=${FFN:-1024}
HEADS=${HEADS:-8}
KV_HEADS=${KV_HEADS:-2}
NUM_EXPERTS=${NUM_EXPERTS:-8}
MBS=${MBS:-8}
GBS=${GBS:-256}
SEQ_LEN=${SEQ_LEN:-4096}

# ---- Bilevel alternation ----
# BILEVEL=1 -> alternate router/expert phases (default). BILEVEL=0 -> joint
# training baseline: no --bilevel-* flags are emitted at all, so it's plain
# Megatron MoE training (no router/expert param-group split, no lr masking).
BILEVEL=${BILEVEL:-0}
BILEVEL_ROUTER_STEPS=${BILEVEL_ROUTER_STEPS:-50}
BILEVEL_EXPERT_STEPS=${BILEVEL_EXPERT_STEPS:-10}
ROUTER_LR=${ROUTER_LR:-}
EXPERT_LR=${EXPERT_LR:-}

# ---- Schedule / eval ----
LR=${LR:-3e-4}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
EVAL_ITERS=${EVAL_ITERS:-10}
# Warmup scales with run length; the scheduler requires warmup < train_iters,
# so clamp it for short/smoke runs (e.g. STEPS=50 -> 5).
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-$(( STEPS / 10 ))}
[ "$LR_WARMUP_ITERS" -ge "$STEPS" ] && LR_WARMUP_ITERS=$(( STEPS / 2 ))
TIME=${TIME:-01:00:00}

# ---- Bilevel arg block + run tag (keeps baseline vs bilevel logs separate) ----
if [ "$BILEVEL" = 1 ]; then
    RUN_TAG="bl-r${BILEVEL_ROUTER_STEPS}e${BILEVEL_EXPERT_STEPS}"
    BILEVEL_LINES="    --bilevel-router-steps ${BILEVEL_ROUTER_STEPS}
    --bilevel-expert-steps ${BILEVEL_EXPERT_STEPS}"
    [ -n "$ROUTER_LR" ] && BILEVEL_LINES="${BILEVEL_LINES}
    --router-lr ${ROUTER_LR}"
    [ -n "$EXPERT_LR" ] && BILEVEL_LINES="${BILEVEL_LINES}
    --expert-lr ${EXPERT_LR}"
    BILEVEL_ARGS_BLOCK="BILEVEL_ARGS=(
${BILEVEL_LINES}
)"
else
    RUN_TAG="joint"
    BILEVEL_ARGS_BLOCK="BILEVEL_ARGS=()"
fi

JOB_NAME="moe-${RUN_TAG}-${STEPS}s-${NODES}n"

################ W&B block ################
WANDB_BLOCK='
# WANDB
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

PARTITION_DIRECTIVE=""
[ -n "$PARTITION" ] && PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=${SBATCH_ACCOUNT}
${PARTITION_DIRECTIVE}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
SBATCH_DIRECTIVES

cat >> "$SCRIPT" << 'BODY_HEAD'

echo "START TIME: \$(date)"

################ Configs ################
BODY_HEAD

cat >> "$SCRIPT" << BODY_WORKDIR
WORKDIR=${WORKDIR}
MEGATRON_LM_DIR=\$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/\$USER/moe-shaping/cache
BODY_WORKDIR

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${STEPS}

# Logging
PROJECT_NAME=${WANDB_PROJECT}
EXP_NAME=moe-${RUN_TAG}-\${SLURM_NNODES}n
LOG_DIR=/iopsstor/scratch/cscs/\$USER/moe-shaping/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/moe-shaping/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/moe-shaping/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

SETUP

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)

# Top-2 routing (Megatron default moe_router_topk=2); per-expert FFN = --ffn-hidden-size.
MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.01
)

# Bilevel router/expert alternation (consumed by the patched training.py).
# Empty when BILEVEL=0 (joint-training baseline).
${BILEVEL_ARGS_BLOCK}
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr ${LR}
    --lr-decay-style constant
    --lr-warmup-iters ${LR_WARMUP_ITERS}
)
TRAINING

cat >> "$SCRIPT" << 'REST'

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --tensorboard-dir $TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
)

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${BILEVEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

REST

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

# Replace placeholder with actual W&B block
sed -i '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << 'FOOTER'

echo "CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 $TRAINING_CMD"

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
sbatch "$SCRIPT"
