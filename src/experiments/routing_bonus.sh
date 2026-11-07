# for seed in {1..3}
# do
#     python src/main.py --config ./configs/fixed_target.yaml wandb.run_name="Baseline" output_dir="./outputs/routing-bonus/fixed-target/baseline-seed$seed" seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py --config ./configs/fixed_target.yaml wandb.run_name="Aux-free Load Balancing" output_dir="./outputs/routing-bonus/fixed-target/aux-free-lb-seed$seed" model.router_type=load_balancing model.router_kwargs.gamma=1e-3 seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py --config ./configs/fixed_target.yaml wandb.run_name="UCB Exploration" output_dir="./outputs/routing-bonus/fixed-target/ucb-seed$seed" model.router_type=ucb model.router_kwargs.ucb_weight=1.0 seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py --config ./configs/fixed_target.yaml wandb.run_name="Rising Bandit Exploration" output_dir="./outputs/routing-bonus/fixed-target/rising-bandit-seed$seed" model.router_type=rising_bandit_bonus model.router_kwargs.epsilon=0.25 model.router_kwargs.sigma=0.1 model.router_kwargs.alpha=3.0 seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py --config ./configs/fixed_target.yaml wandb.run_name="Random Router" output_dir="./outputs/routing-bonus/fixed-target/random-seed$seed" model.router_type=random seed=$seed &
#     sleep 5
# done

for seed in {1..3}
do
    python src/main.py --config ./configs/fixed_target.yaml wandb.run_name="Frozen Router" output_dir="./outputs/routing-bonus/fixed-target/frozen-seed$seed" model.freeze_router=true seed=$seed &
    sleep 5
done

# for seed in {1..3}
# do
#     python src/main.py --config ./configs/fixed_imbalanced_target.yaml wandb.run_name="Imbalanced Target - Baseline" output_dir="./outputs/routing-bonus/fixed-imbalanced-target/baseline-seed$seed" seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py --config ./configs/fixed_imbalanced_target.yaml wandb.run_name="Imbalanced Target - Aux-free Load Balancing" output_dir="./outputs/routing-bonus/fixed-imbalanced-target/aux-free-lb-seed$seed" model.router_type=load_balancing model.router_kwargs.gamma=1e-3 seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py --config ./configs/fixed_imbalanced_target.yaml wandb.run_name="Imbalanced Target - UCB Exploration" output_dir="./outputs/routing-bonus/fixed-imbalanced-target/ucb-seed$seed" model.router_type=ucb model.router_kwargs.ucb_weight=1.0 seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py --config ./configs/fixed_imbalanced_target.yaml wandb.run_name="Imbalanced Target - Rising Bandit Exploration" output_dir="./outputs/routing-bonus/fixed-imbalanced-target/rising-bandit-seed$seed" model.router_type=rising_bandit_bonus model.router_kwargs.epsilon=0.25 model.router_kwargs.sigma=0.1 model.router_kwargs.alpha=3.0 seed=$seed &
#     sleep 5
# done

# for seed in {1..3}
# do
#     python src/main.py --config ./configs/fixed_imbalanced_target.yaml wandb.run_name="Imbalanced Target - Random Router" output_dir="./outputs/routing-bonus/fixed-imbalanced-target/random-seed$seed" model.router_type=random seed=$seed &
#     sleep 5
# done

for seed in {1..3}
do
    python src/main.py --config ./configs/fixed_imbalanced_target.yaml wandb.run_name="Imbalanced Target - Frozen Router" output_dir="./outputs/routing-bonus/fixed-imbalanced-target/frozen-seed$seed" model.freeze_router=true seed=$seed &
    sleep 5
done
