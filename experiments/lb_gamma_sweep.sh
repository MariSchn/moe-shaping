# Sweep over the bias update speed gamma of aux-free load balancing on both fixed targets, 3 seeds.
# The 3 seeds of a gamma run in parallel, the gammas and targets one after another.
for target in fixed_balanced fixed_imbalanced_pair
do
    if [ $target = fixed_balanced ]; then prefix="Balanced Target"; else prefix="Imbalanced Target"; fi
    for gamma in 1e-5 1e-4 1e-3 1e-2 1e-1; do
        for seed in {1..3}; do
            python src/main.py target=$target router=load_balancing router.gamma=$gamma seed=$seed wandb.run_name="'$prefix - Aux-free Load Balancing (gamma=$gamma)'" hydra.run.dir="./outputs/lb-gamma-sweep/$target/gamma$gamma-seed$seed" &
            sleep 5
        done
        wait
    done
done
