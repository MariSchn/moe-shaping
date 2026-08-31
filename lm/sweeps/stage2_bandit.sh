#!/bin/bash
#
# Stage 2, bandit arm: 3000 steps x 3 seeds, with the configuration stage 1 chose.
#
# Shares every hyper-parameter with sweeps/stage2_baselines.sh -- same model,
# batch, schedule, --lr, warmup and seeds, and no --router-lr -- so the only
# difference from the noLB arm is where the router's gradient comes from.
#
# Pass the stage-1 winner's knobs as environment variables, e.g.
#   BANDIT_BASELINE=critic BANDIT_EXPLORE=gumbel ./sweeps/stage2_bandit.sh
# A second arm can be added with ARM=bandit-aux AUX_LOSS_COEFF=0.01 ...

set -euo pipefail
cd "$(dirname "$0")/.."

export STEPS=3000
export LR=2e-3
export EVAL_INTERVAL=100
export TIME=11:00:00
ARM=${ARM:-bandit}

# NOTE: this uses an exporting subshell rather than `env VAR=x ./launch.sh`. A
# shim earlier in PATH than GNU env (~/.local/bin/env) swallows that form
# silently -- it exits 0 having run nothing.
for seed in 42 43 44; do
    name="s2-${ARM}-s${seed}"
    echo "=== $name ==="
    (
        export BANDIT=1 \
               AUX_LOSS_COEFF="${AUX_LOSS_COEFF:-0}" \
               BANDIT_BASELINE="${BANDIT_BASELINE:-critic}" \
               BANDIT_EXPLORE="${BANDIT_EXPLORE:-gumbel}" \
               BANDIT_TAU="${BANDIT_TAU:-1.0}" \
               BANDIT_TAU_FINAL="${BANDIT_TAU_FINAL:-0.1}" \
               BANDIT_ENTROPY="${BANDIT_ENTROPY:-0.0}" \
               BANDIT_CRITIC_COEFF="${BANDIT_CRITIC_COEFF:-0.01}" \
               BANDIT_KEEP_TASK_GRAD="${BANDIT_KEEP_TASK_GRAD:-0}" \
               JOB_NAME="$name" RUN_NAME="$name" SEED="$seed" LR="$LR" \
               EVAL_INTERVAL="$EVAL_INTERVAL" TIME="$TIME"
        ./launch.sh "$STEPS"
    )
done

echo
squeue -u "$USER"
