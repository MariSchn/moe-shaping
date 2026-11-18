for target in fixed_balanced fixed_imbalanced_pair
do
    if [ $target = fixed_balanced ]; then prefix="Balanced Target"; else prefix="Imbalanced Target"; fi
    for discount in 0.5 0.75 0.9 0.99 0.999 0.9999; do
        python src/main.py target=$target router=ucb router.ucb_weight=1.0 router.per_step=true router.discount=$discount wandb.run_name="'$prefix - Per-step Discounted UCB Exploration (discount=$discount)'" hydra.run.dir="./outputs/ucb-discount-sweep/$target/discount$discount" &
        sleep 5
    done
    wait
done
