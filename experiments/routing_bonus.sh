# for seed in {1..3}
# do
#     python src/main.py target=fixed wandb.run_name="Baseline" hydra.run.dir="./outputs/routing-bonus/fixed-target/baseline-seed$seed" seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py target=fixed wandb.run_name="Aux-free Load Balancing" hydra.run.dir="./outputs/routing-bonus/fixed-target/aux-free-lb-seed$seed" router=load_balancing router.gamma=1e-3 seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py target=fixed wandb.run_name="UCB Exploration" hydra.run.dir="./outputs/routing-bonus/fixed-target/ucb-seed$seed" router=ucb router.ucb_weight=1.0 seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py target=fixed wandb.run_name="Rising Bandit Exploration" hydra.run.dir="./outputs/routing-bonus/fixed-target/rising-bandit-seed$seed" router=rising_bandit_bonus router.epsilon=0.25 router.sigma=0.1 router.alpha=3.0 seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py target=fixed wandb.run_name="Random Router" hydra.run.dir="./outputs/routing-bonus/fixed-target/random-seed$seed" router=random seed=$seed &
#     sleep 5
# done

for seed in {1..3}
do
    python src/main.py target=fixed wandb.run_name="Frozen Router" hydra.run.dir="./outputs/routing-bonus/fixed-target/frozen-seed$seed" training.freeze_router=true seed=$seed &
    sleep 5
done

# for seed in {1..3}
# do
#     python src/main.py target=fixed_imbalanced wandb.run_name="Imbalanced Target - Baseline" hydra.run.dir="./outputs/routing-bonus/fixed-imbalanced-target/baseline-seed$seed" seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py target=fixed_imbalanced wandb.run_name="Imbalanced Target - Aux-free Load Balancing" hydra.run.dir="./outputs/routing-bonus/fixed-imbalanced-target/aux-free-lb-seed$seed" router=load_balancing router.gamma=1e-3 seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py target=fixed_imbalanced wandb.run_name="Imbalanced Target - UCB Exploration" hydra.run.dir="./outputs/routing-bonus/fixed-imbalanced-target/ucb-seed$seed" router=ucb router.ucb_weight=1.0 seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py target=fixed_imbalanced wandb.run_name="Imbalanced Target - Rising Bandit Exploration" hydra.run.dir="./outputs/routing-bonus/fixed-imbalanced-target/rising-bandit-seed$seed" router=rising_bandit_bonus router.epsilon=0.25 router.sigma=0.1 router.alpha=3.0 seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py target=fixed_imbalanced wandb.run_name="Imbalanced Target - Random Router" hydra.run.dir="./outputs/routing-bonus/fixed-imbalanced-target/random-seed$seed" router=random seed=$seed &
#     sleep 5
# done

for seed in {1..3}
do
    python src/main.py target=fixed_imbalanced wandb.run_name="Imbalanced Target - Frozen Router" hydra.run.dir="./outputs/routing-bonus/fixed-imbalanced-target/frozen-seed$seed" training.freeze_router=true seed=$seed &
    sleep 5
done
