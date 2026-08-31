#!/bin/bash
#
# Stage 1: pick a bandit configuration, at 800 steps and a single seed.
#
# Every run below shares the same model, batch, schedule, seed and --lr. Only the
# --bandit-* knob under test changes, so a difference in final loss is a
# difference in the bandit setup and nothing else. Two reference arms (a normal
# aux-loss-balanced router and a frozen one) are included so the bandit numbers
# can be read against something.
#
# Note on the learning rate: Adam is invariant to the scale of a parameter's
# gradient, and in the pure-bandit arms the REINFORCE term is the router's *only*
# gradient. So --bandit-coeff barely moves the router's step size there and the
# real step-size knob is --router-lr. Holding --lr fixed with no --router-lr
# therefore gives every arm the same per-step router movement and differs only in
# its direction, which is the comparison we want. The two rlr runs deliberately
# break that parity to check the bandit arm is not simply mis-scaled; they are
# labelled as tuning checks, not as comparable arms.
#
# Usage: ./sweeps/stage1.sh   (from lm/)

set -euo pipefail
cd "$(dirname "$0")/.."

export STEPS=800
export SEED=42
export LR=2e-3
export EVAL_INTERVAL=100
export TIME=03:00:00

run() {
    local name="$1"; shift
    echo "=== $name ==="
    env "$@" JOB_NAME="$name" RUN_NAME="$name" SEED=$SEED LR=$LR \
        EVAL_INTERVAL=$EVAL_INTERVAL TIME=$TIME ./launch.sh $STEPS
}

# --- reference arms (no bandit) ---
run s1-base-aux    AUX_LOSS_COEFF=0.01
run s1-base-frozen AUX_LOSS_COEFF=0 ROUTER_LR=0

# --- which baseline reduces the advantage's variance best ---
run s1-bd-bm      BANDIT=1 AUX_LOSS_COEFF=0 BANDIT_BASELINE=batch_mean
run s1-bd-critic  BANDIT=1 AUX_LOSS_COEFF=0 BANDIT_BASELINE=critic
run s1-bd-nobase  BANDIT=1 AUX_LOSS_COEFF=0 BANDIT_BASELINE=none

# --- how much exploration, and of which kind ---
run s1-bd-greedy  BANDIT=1 AUX_LOSS_COEFF=0 BANDIT_BASELINE=critic BANDIT_EXPLORE=none
run s1-bd-eps     BANDIT=1 AUX_LOSS_COEFF=0 BANDIT_BASELINE=critic BANDIT_EXPLORE=epsilon \
                  BANDIT_EPSILON=0.1 BANDIT_EPSILON_FINAL=0.01
run s1-bd-tau3    BANDIT=1 AUX_LOSS_COEFF=0 BANDIT_BASELINE=critic BANDIT_TAU=3.0 BANDIT_TAU_FINAL=0.3

# --- does anything rescue it: load balancing, the task gradient, entropy ---
run s1-bd-aux     BANDIT=1 AUX_LOSS_COEFF=0.01 BANDIT_BASELINE=critic
run s1-bd-entropy BANDIT=1 AUX_LOSS_COEFF=0 BANDIT_BASELINE=critic BANDIT_ENTROPY=0.01
run s1-bd-hybrid  BANDIT=1 AUX_LOSS_COEFF=0 BANDIT_BASELINE=critic \
                  BANDIT_KEEP_TASK_GRAD=1 BANDIT_GRAD_ALIGN_INTERVAL=25
run s1-bd-hyb-c01 BANDIT=1 AUX_LOSS_COEFF=0 BANDIT_BASELINE=critic \
                  BANDIT_KEEP_TASK_GRAD=1 BANDIT_COEFF=0.1 BANDIT_GRAD_ALIGN_INTERVAL=25

# --- tuning checks: NOT HP-matched to the arms above (see the header) ---
run s1-bd-rlr5e-4 BANDIT=1 AUX_LOSS_COEFF=0 BANDIT_BASELINE=critic ROUTER_LR=5e-4
run s1-bd-rlr5e-3 BANDIT=1 AUX_LOSS_COEFF=0 BANDIT_BASELINE=critic ROUTER_LR=5e-3

echo
squeue -u "$USER"
