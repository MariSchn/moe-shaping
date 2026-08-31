#!/bin/bash
#
# Stage 2, baseline arms: the headline comparison, 3000 steps x 3 seeds.
#
# The four arms here differ ONLY in how expert load is balanced and whether the
# router learns. Model, batch, sequence length, schedule, --lr, warmup, weight
# decay, clipping and seeds are identical across all of them (and across the
# bandit arm added by stage2_bandit.sh), so the loss difference between arms is
# attributable to the routing mechanism.
#
#   noLB     router trained by the task gradient, no load balancing at all
#   aux      router trained by the task gradient + the Switch aux loss (coeff 0.01)
#   auxfree  router trained by the task gradient + the DeepSeek-V3 expert bias
#   frozen   router never updated (--router-lr 0): the "routing does not need to
#            be learned" control
#
# Every arm keeps --moe-router-load-balancing-type aux_loss so that patch 0003's
# expert_imbalance monitor and patch 0004's load/* quantiles are recorded for all
# of them; an arm that should apply no aux loss sets its coefficient to 0. (The
# prior attempt used --moe-router-load-balancing-type none for auxfree and lost
# every load statistic for that arm.)
#
# Two unavoidable asymmetries, called out in the report:
#   * auxfree needs the sigmoid score function -- Megatron requires it for the
#     expert bias -- so that arm's router scoring differs from the others'.
#   * frozen sets --router-lr 0, which puts the router in its own optimizer param
#     group (patch 0002). The group split itself has no effect at lr 0.
#
# Usage: ./sweeps/stage2_baselines.sh   (from lm/)

set -euo pipefail
cd "$(dirname "$0")/.."

export STEPS=3000
export LR=2e-3
export EVAL_INTERVAL=100
export TIME=11:00:00

run() {
    local arm="$1"; shift
    for seed in 42 43 44; do
        local name="s2-${arm}-s${seed}"
        echo "=== $name ==="
        env "$@" JOB_NAME="$name" RUN_NAME="$name" SEED=$seed LR=$LR \
            EVAL_INTERVAL=$EVAL_INTERVAL TIME=$TIME ./launch.sh $STEPS
    done
}

run noLB    AUX_LOSS_COEFF=0
run aux     AUX_LOSS_COEFF=0.01
run auxfree AUX_LOSS_COEFF=0 EXPERT_BIAS=1
run frozen  AUX_LOSS_COEFF=0 ROUTER_LR=0

echo
squeue -u "$USER"
