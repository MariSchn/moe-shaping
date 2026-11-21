# LinUCB's ridge lambda against its weight and discount on both fixed targets, 3 seeds. The
# bonus is at most weight * sqrt((1 + x^2) / lambda), so lambda = 1 caps it at sqrt(2) for
# weight 1. Each config is "weight discount lambda"; discount 1.0 never forgets old pulls.
# The 3 seeds of a config run in parallel, the configs and targets one after another.
for target in fixed_balanced fixed_imbalanced_pair
do
    if [ $target = fixed_balanced ]; then prefix="Balanced Target"; else prefix="Imbalanced Target"; fi
    for config in "1.0 1.0 1.0" "1.0 1.0 0.1" "1.0 0.5 1.0" "1.0 0.5 0.1" "10.0 0.5 1.0" "10.0 0.5 0.1"; do
        read -r weight discount ridge <<< "$config"
        for seed in {1..3}; do
            python src/main.py target=$target router=linucb router.linucb_weight=$weight router.linucb_ridge=$ridge router.per_step=true router.discount=$discount seed=$seed wandb.run_name="'$prefix - Per-step LinUCB Exploration (w=$weight, discount=$discount, lambda=$ridge)'" hydra.run.dir="./outputs/linucb-ridge-sweep/$target/w$weight-discount$discount-ridge$ridge-seed$seed" &
            sleep 5
        done
        wait
    done
done
