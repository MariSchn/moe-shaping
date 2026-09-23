#!/bin/bash
#
# ~1.3B-parameter MoE (64 experts, top-8, dropless): baseline vs load balancing vs
# bilevel vs bandit vs LOLA. Every arm shares the model, data, schedule, seeds and --lr;
# only the assignments passed to run() differ.
#
# Usage (from lm/):  ARMS="noLB aux" SEEDS="42" ./sweeps/compare.sh
#   ARMS   subset of the arms below (default: all but lola)
#   SEEDS  default "42 43"; the frozen control runs on the first seed only

set -euo pipefail
cd "$(dirname "$0")/.."

export STEPS=${STEPS:-1500}
export LR=1e-3 LR_DECAY_STYLE=cosine MIN_LR=1e-4
export EVAL_INTERVAL=${EVAL_INTERVAL:-250}
export EVAL_ITERS=${EVAL_ITERS:-10}
export TIME=${TIME:-04:00:00}
export PARTITION=${PARTITION:-normal}
export CKPT_INTERVAL=${CKPT_INTERVAL:-250}
export PRECISION_AWARE_OPTIMIZER=${PRECISION_AWARE_OPTIMIZER:-0}
export WANDB_PROJECT=${WANDB_PROJECT:-moe-method-comparison}

export NUM_LAYERS=12 HIDDEN=1024 FFN=512 HEADS=16 KV_HEADS=4
export NUM_EXPERTS=64 TOPK=8
export MBS=${MBS:-8} GBS=256 SEQ_LEN=4096

ARM_LOLA_ALPHA=${LOLA_ALPHA:-}
unset LOLA_ALPHA
ARMS=${ARMS:-"noLB aux auxfree noLB-sigmoid bilevel bandit bandit-hybrid frozen"}
SEEDS=${SEEDS:-"42 43"}
PREFIX=${PREFIX:-cmp}

BANDIT_KNOBS="BANDIT=1 BANDIT_BASELINE=critic BANDIT_EXPLORE=epsilon BANDIT_EPSILON=0.1
              BANDIT_EPSILON_FINAL=0.01 BANDIT_COEFF=0.001 BANDIT_ENTROPY=0"

arm_assignments() {
    case "$1" in
        noLB)          echo "AUX_LOSS_COEFF=0" ;;
        aux)           echo "AUX_LOSS_COEFF=0.01" ;;
        auxfree)       echo "AUX_LOSS_COEFF=0 EXPERT_BIAS=1" ;;
        noLB-sigmoid)  echo "AUX_LOSS_COEFF=0 SCORE_FUNCTION=sigmoid" ;;
        bilevel)       echo "AUX_LOSS_COEFF=0 BILEVEL=1 BILEVEL_ROUTER_STEPS=10 BILEVEL_EXPERT_STEPS=40" ;;
        bandit)        echo "AUX_LOSS_COEFF=0 $BANDIT_KNOBS" ;;
        bandit-hybrid) echo "AUX_LOSS_COEFF=0 $BANDIT_KNOBS BANDIT_KEEP_TASK_GRAD=1" ;;
        lola)          echo "AUX_LOSS_COEFF=0 LOLA_ALPHA=${ARM_LOLA_ALPHA:?set LOLA_ALPHA for the lola arm}" ;;
        aux-*)         echo "AUX_LOSS_COEFF=${1#aux-}" ;;
        auxfree-*)     echo "AUX_LOSS_COEFF=0 EXPERT_BIAS=1 BIAS_UPDATE_RATE=${1#auxfree-}" ;;
        bilevel-*x*)   spec=${1#bilevel-}
                       echo "AUX_LOSS_COEFF=0 BILEVEL=1 BILEVEL_ROUTER_STEPS=${spec%x*} BILEVEL_EXPERT_STEPS=${spec#*x}" ;;
        bandit-hybrid-noexplore) echo "AUX_LOSS_COEFF=0 $BANDIT_KNOBS BANDIT_KEEP_TASK_GRAD=1 BANDIT_EXPLORE=none" ;;
        bandit-hybrid-coeff0) echo "AUX_LOSS_COEFF=0 $BANDIT_KNOBS BANDIT_KEEP_TASK_GRAD=1 BANDIT_COEFF=0" ;;
        bandit-hybrid-align) echo "AUX_LOSS_COEFF=0 $BANDIT_KNOBS BANDIT_KEEP_TASK_GRAD=1 BANDIT_GRAD_ALIGN_INTERVAL=25" ;;
        lola-*-cap*)   spec=${1#lola-}
                       echo "AUX_LOSS_COEFF=0 LOLA_ALPHA=${spec%-cap*} LOLA_MAX_RATIO=${spec#*-cap}" ;;
        lola-*)        echo "AUX_LOSS_COEFF=0 LOLA_ALPHA=${1#lola-}" ;;
        frozen)        echo "AUX_LOSS_COEFF=0 ROUTER_LR=0" ;;
        *)             echo "unknown arm $1" >&2; exit 1 ;;
    esac
}

first_seed=${SEEDS%% *}
for arm in $ARMS; do
    assignments=$(arm_assignments "$arm")
    for seed in $SEEDS; do
        [ "$arm" = frozen ] && [ "$seed" != "$first_seed" ] && continue
        name="${PREFIX}-${arm}-s${seed}"
        echo "=== $name: $assignments ==="
        (
            export JOB_NAME="$name" RUN_NAME="$name" SEED="$seed"
            for assignment in $assignments; do export "$assignment"; done
            ./launch.sh "$STEPS"
        )
    done
done
