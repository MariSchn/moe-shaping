# Baseline vs. aux-free load balancing vs. discounted UCB vs. the separated router, on a
# target that only the separated router can represent.
# The 3 seeds of a method run in parallel, the methods one after another.
prefix="Separation Counterexample"
for seed in {1..3}; do
    python src/main.py target=separation_counterexample model.num_experts=3 training.num_steps=60000 logging.frame_every=600 router=mlp seed=$seed wandb.run_name="$prefix - Baseline" hydra.run.dir="./outputs/separation-counterexample/baseline-seed$seed" &
    sleep 5
done
wait

for seed in {1..3}; do
    python src/main.py target=separation_counterexample model.num_experts=3 training.num_steps=60000 logging.frame_every=600 router=load_balancing router.gamma=1e-4 seed=$seed wandb.run_name="'$prefix - Aux-free Load Balancing (gamma=1e-4)'" hydra.run.dir="./outputs/separation-counterexample/aux-free-lb-seed$seed" &
    sleep 5
done
wait

for seed in {1..3}; do
    python src/main.py target=separation_counterexample model.num_experts=3 training.num_steps=60000 logging.frame_every=600 router=ucb router.discount=0.5 seed=$seed wandb.run_name="'$prefix - Discounted UCB Exploration (discount=0.5)'" hydra.run.dir="./outputs/separation-counterexample/ucb-discounted-seed$seed" &
    sleep 5
done
wait

for seed in {1..3}; do
    python src/main.py target=separation_counterexample model.num_experts=3 training.num_steps=60000 logging.frame_every=600 router=separated seed=$seed wandb.run_name="$prefix - Separated Router" hydra.run.dir="./outputs/separation-counterexample/separated-seed$seed" &
    sleep 5
done
wait
