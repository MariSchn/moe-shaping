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
#   Bilevel: BILEVEL=1|0            master switch (default 0). BILEVEL=1 splits the
#                                   MoE router/expert params into their own optimizer
#                                   param groups; BILEVEL=0 = joint baseline (no split,
#                                   single --lr). With BILEVEL=1, the sub-mode depends
#                                   on what you set:
#            BILEVEL_ROUTER_STEPS  } not both 0 -> ALTERNATING router-only / expert-only
#            BILEVEL_EXPERT_STEPS  }   phases (per-cycle step counts; default 50 / 10)
#            ROUTER_LR / EXPERT_LR both steps 0 -> DIFFERENT-LR ONLY: no alternation,
#                                   router & experts train every step at their own LR.
#                                   In alternating mode these are the optional per-phase
#                                   LRs (default --lr for both).
#   Schedule/eval: LR EVAL_INTERVAL EVAL_ITERS LR_WARMUP_ITERS TIME
#   Cluster: SBATCH_ACCOUNT PARTITION (empty = cluster default, e.g. PARTITION=debug)
#   W&B: RUN_NAME (run/TB name)  |  MoE LB: LB_TYPE (aux_loss|...|none) AUX_LOSS_COEFF
#   Aux-loss-free LB: EXPERT_BIAS=1 (DeepSeek-V3 per-expert bias; forces sigmoid
#                     router) [BIAS_UPDATE_RATE] | SCORE_FUNCTION (softmax|sigmoid)
#
# Baseline vs bilevel runs get distinct job/log/W&B names (RUN_TAG: joint vs
# bl-r<R>e<E>), so their TensorBoard dirs and runs do not clobber each other.
#
# Examples:
#   ./launch.sh 1000                             # joint baseline (BILEVEL=0)
#   BILEVEL=1 ./launch.sh 1000                   # alternating (default 50/10 steps)
#   BILEVEL=1 BILEVEL_ROUTER_STEPS=0 BILEVEL_EXPERT_STEPS=0 \
#       ROUTER_LR=1e-3 EXPERT_LR=3e-4 ./launch.sh 1000   # different-LR only, no alternation
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

# ---- Bilevel / router-expert param-group split ----
# BILEVEL=1 -> split MoE router (*.router.*) and expert (*.experts.*) params into
# their own optimizer param groups. The sub-mode is picked by what you set below:
#   * BILEVEL_ROUTER_STEPS/BILEVEL_EXPERT_STEPS not both 0 -> alternating phases.
#   * both step counts 0 (+ ROUTER_LR/EXPERT_LR) -> different-LR only, no alternation.
# BILEVEL=0 -> joint baseline: no --bilevel-* flags, no split, plain Megatron MoE.
BILEVEL=${BILEVEL:-0}
BILEVEL_ROUTER_STEPS=${BILEVEL_ROUTER_STEPS:-50}
BILEVEL_EXPERT_STEPS=${BILEVEL_EXPERT_STEPS:-10}
ROUTER_LR=${ROUTER_LR:-}
EXPERT_LR=${EXPERT_LR:-}

# ---- Schedule / eval ----
LR=${LR:-1e-3}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
EVAL_ITERS=${EVAL_ITERS:-10}
# Warmup scales with run length; the scheduler requires warmup < train_iters,
# so clamp it for short/smoke runs (e.g. STEPS=50 -> 5).
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-$(( STEPS / 10 ))}
[ "$LR_WARMUP_ITERS" -ge "$STEPS" ] && LR_WARMUP_ITERS=$(( STEPS / 2 ))
TIME=${TIME:-01:00:00}

# ---- Bilevel arg block + run tag (keeps baseline vs split-mode logs separate) ----
if [ "$BILEVEL" = 1 ]; then
    if [ "$BILEVEL_ROUTER_STEPS" = 0 ] && [ "$BILEVEL_EXPERT_STEPS" = 0 ]; then
        # Different-LR only: split router/expert groups, no alternation. Needs at
        # least one of ROUTER_LR/EXPERT_LR (else it's identical to the joint baseline).
        if [ -z "$ROUTER_LR" ] && [ -z "$EXPERT_LR" ]; then
            echo "ERROR: BILEVEL=1 with BILEVEL_ROUTER_STEPS=BILEVEL_EXPERT_STEPS=0 needs ROUTER_LR and/or EXPERT_LR." >&2
            exit 1
        fi
        RUN_TAG="splitlr-r${ROUTER_LR:-lr}-e${EXPERT_LR:-lr}"
        BILEVEL_LINES=""
        [ -n "$ROUTER_LR" ] && BILEVEL_LINES="    --router-lr ${ROUTER_LR}"
        [ -n "$EXPERT_LR" ] && BILEVEL_LINES="${BILEVEL_LINES:+${BILEVEL_LINES}
}    --expert-lr ${EXPERT_LR}"
    else
        # Alternating: router-only / expert-only phases (ROUTER_LR/EXPERT_LR optional).
        RUN_TAG="bl-r${BILEVEL_ROUTER_STEPS}e${BILEVEL_EXPERT_STEPS}"
        BILEVEL_LINES="    --bilevel-router-steps ${BILEVEL_ROUTER_STEPS}
    --bilevel-expert-steps ${BILEVEL_EXPERT_STEPS}"
        [ -n "$ROUTER_LR" ] && BILEVEL_LINES="${BILEVEL_LINES}
    --router-lr ${ROUTER_LR}"
        [ -n "$EXPERT_LR" ] && BILEVEL_LINES="${BILEVEL_LINES}
    --expert-lr ${EXPERT_LR}"
    fi
    BILEVEL_ARGS_BLOCK="BILEVEL_ARGS=(
${BILEVEL_LINES}
)"
else
    RUN_TAG="joint"
    BILEVEL_ARGS_BLOCK="BILEVEL_ARGS=()"
fi

# JOB_NAME names the generated sbatch script (logs/<JOB_NAME>.sbatch) and the
# per-run log files (logs/<JOB_NAME>-<jobid>.log). Override it when sweeping
# params that don't change RUN_TAG (e.g. layer ablations) so runs don't clobber
# each other's script/logs.
JOB_NAME=${JOB_NAME:-moe-${RUN_TAG}-${STEPS}s-${NODES}n}

# W&B run name + TensorBoard subdir (override with RUN_NAME=...).
RUN_NAME=${RUN_NAME:-moe-${RUN_TAG}-${NODES}n}

# ---- MoE load balancing ----
# LB_TYPE: aux_loss (default) | seq_aux_loss | global_aux_loss | sinkhorn | none.
# AUX_LOSS_COEFF: weight of the aux load-balancing loss (ignored when LB_TYPE=none).
# Monitoring: 'expert_imbalance' is always logged for aux-loss routing (patch 0003),
#   so AUX_LOSS_COEFF=0 trains with NO load balancing yet still tracks imbalance;
#   LB_TYPE=none turns load balancing AND imbalance monitoring fully off.
LB_TYPE=${LB_TYPE:-aux_loss}
AUX_LOSS_COEFF=${AUX_LOSS_COEFF:-0.01}
MOE_LB_LINES="    --moe-router-load-balancing-type ${LB_TYPE}"
[ "$LB_TYPE" != none ] && MOE_LB_LINES="${MOE_LB_LINES}
    --moe-aux-loss-coeff ${AUX_LOSS_COEFF}"

# ---- Aux-loss-free load balancing (DeepSeek-V3 expert bias) ----
# EXPERT_BIAS=1 adds a dynamic per-expert routing bias that balances expert load
# WITHOUT an aux loss; Megatron requires the sigmoid score function for it, so we
# force SCORE_FUNCTION=sigmoid below. For pure aux-loss-free balancing, pair it
# with LB_TYPE=none AUX_LOSS_COEFF=0 (no aux loss at all). BIAS_UPDATE_RATE is the
# bias step size (Megatron default 1e-3). SCORE_FUNCTION can also be set on its own
# (softmax default | sigmoid) for a sigmoid router without the bias.
# Set a distinct RUN_NAME/JOB_NAME to keep these runs' logs separate.
EXPERT_BIAS=${EXPERT_BIAS:-0}
BIAS_UPDATE_RATE=${BIAS_UPDATE_RATE:-1e-3}
SCORE_FUNCTION=${SCORE_FUNCTION:-softmax}
if [ "$EXPERT_BIAS" = 1 ]; then
    SCORE_FUNCTION=sigmoid   # required by Megatron for expert-bias routing
    MOE_LB_LINES="${MOE_LB_LINES}
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate ${BIAS_UPDATE_RATE}"
fi
[ "$SCORE_FUNCTION" != softmax ] && MOE_LB_LINES="${MOE_LB_LINES}
    --moe-router-score-function ${SCORE_FUNCTION}"

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
EXP_NAME=${RUN_NAME}
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
${MOE_LB_LINES}
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
