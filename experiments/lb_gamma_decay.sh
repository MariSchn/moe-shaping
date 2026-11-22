# Aux-free load balancing with a cosine decay of the bias update speed gamma (Beam) on both
# fixed targets, 3 seeds. gamma is constant until step router.decay_start and decays to 0
# at step router.decay_end. Compare with discounted UCB, discount 0.5, in ucb_discount_sweep.sh.
#
# The best schedule, gamma=2e-3 decaying from step 0 to 0 at step 1500, fits both targets
# for 8/10 seeds 1-10 and 18/20 seeds 11-30, UCB for 9/10 and 19/20. UCB fits the balanced
# target more closely (target_mse 1.9 vs 5.2 on seeds 11-30). Around the best, each bias
# moves by less in total, gamma * (decay_end + decay_start) / 2, and the balanced target
# stays collapsed, or by more and the imbalanced target breaks. Seeds 1-3 fitting both
# targets (loss on fresh inputs below 0.2 and 0.02) in CPU runs:
#
#   gamma  decay_start  decay_end  bias movement  fit both
#   2e-3   0            1500       1.5            2, 3      the best
#   1e-3   38           5000       2.5            2         Beam's, constant during its LR warm-up (0.75%)
#   1e-3   0            1500       0.75           none
#   2e-3   0            1000       1              1
#   2e-3   500          1500       2              2
#   2e-3   0            2500       2.5            none
#   4e-3   0            1500       3              none
#
# The 3 seeds of a schedule run in parallel, the schedules and targets one after another.
for target in fixed_balanced fixed_imbalanced_pair
do
    if [ $target = fixed_balanced ]; then prefix="Balanced Target"; else prefix="Imbalanced Target"; fi
    for schedule in "2e-3 0 1500" "1e-3 38 5000" "1e-3 0 1500" "2e-3 0 1000" "2e-3 500 1500" "2e-3 0 2500" "4e-3 0 1500"; do
        read -r gamma start end <<< "$schedule"
        for seed in {1..3}; do
            python src/main.py target=$target router=load_balancing router.gamma=$gamma router.decay_start=$start router.decay_end=$end seed=$seed wandb.run_name="'$prefix - Aux-free Load Balancing (gamma=$gamma, decay steps $start-$end)'" hydra.run.dir="./outputs/lb-gamma-decay/$target/gamma$gamma-start$start-end$end-seed$seed" &
            sleep 5
        done
        wait
    done
done
