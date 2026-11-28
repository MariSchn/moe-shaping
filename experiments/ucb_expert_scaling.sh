# Per-step discounted UCB with 4, 8 and 16 experts at active fractions top_k / num_experts
# of 0.25, 0.5 and 0.75, with discount 1 - active fraction, on both fixed targets, 3 seeds.
# The targets stay 4-expert top-2 models, only the trained model scales.
# The 3 seeds of a configuration run in parallel, the configurations and targets one after another.
for target in fixed_balanced fixed_imbalanced_pair
do
    if [ $target = fixed_balanced ]; then prefix="Balanced Target"; else prefix="Imbalanced Target"; fi
    for num_experts in 4 8 16; do
        for split in "0.25 0.75" "0.5 0.5" "0.75 0.25"; do
            read -r active discount <<< "$split"
            top_k=$(awk "BEGIN {print $num_experts * $active}")
            for seed in {1..3}; do
                python src/main.py target=$target router=ucb router.per_step=true router.discount=$discount model.num_experts=$num_experts model.top_k=$top_k seed=$seed wandb.run_name="'$prefix - Per-step Discounted UCB Exploration (experts=$num_experts, top_k=$top_k, discount=$discount)'" hydra.run.dir="./outputs/ucb-expert-scaling/$target/experts$num_experts-topk$top_k-seed$seed" &
                sleep 5
            done
            wait
        done
    done
done
