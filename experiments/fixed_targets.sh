# Baseline vs. aux-free load balancing vs. UCB exploration bonus on both fixed targets.
# The 3 seeds of a method run in parallel, the methods one after another.
for target in fixed_balanced fixed_imbalanced_pair
do
    if [ $target = fixed_balanced ]; then prefix="Balanced Target"; else prefix="Imbalanced Target"; fi

    # for seed in {1..3}; do
    #     python src/main.py target=$target router=mlp seed=$seed wandb.run_name="$prefix - Baseline" hydra.run.dir="./outputs/fixed-targets/$target/baseline-seed$seed" &
    #     sleep 5
    # done

    # for seed in {1..3}; do
    #     python src/main.py target=$target router=load_balancing router.gamma=1e-3 seed=$seed wandb.run_name="$prefix - Aux-free Load Balancing" hydra.run.dir="./outputs/fixed-targets/$target/aux-free-lb-seed$seed" &
    #     sleep 5
    # done

    # for seed in {1..3}; do
    #     python src/main.py target=$target router=ucb router.ucb_weight=1.0 seed=$seed wandb.run_name="$prefix - UCB Exploration" hydra.run.dir="./outputs/fixed-targets/$target/ucb-seed$seed" &
    #     sleep 5
    # done

    for seed in {1..3}; do
        python src/main.py target=$target router=ucb router.ucb_weight=1.0 router.per_step=true seed=$seed wandb.run_name="$prefix - Per-step UCB Exploration" hydra.run.dir="./outputs/fixed-targets/$target/ucb-per-step-seed$seed" &
        sleep 5
    done

    for seed in {1..3}; do
        python src/main.py target=$target router=ucb router.ucb_weight=1.0 router.per_step=true router.discount=0.99 seed=$seed wandb.run_name="$prefix - Per-step Discounted Per-step UCB Exploration" hydra.run.dir="./outputs/fixed-targets/$target/ucb-discounted-seed$seed" &
        sleep 5
    done
done
