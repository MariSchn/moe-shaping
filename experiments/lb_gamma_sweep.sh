# Sweep over the bias update speed gamma of aux-free load balancing on both fixed targets, one seed.
# The gammas of a target run in parallel, the targets one after another.
for target in fixed_balanced fixed_imbalanced_pair
do
    if [ $target = fixed_balanced ]; then prefix="Balanced Target"; else prefix="Imbalanced Target"; fi
    for gamma in 1e-5 1e-4 1e-3 1e-2 1e-1; do
        python src/main.py target=$target router=load_balancing router.gamma=$gamma wandb.run_name="'$prefix - Aux-free Load Balancing (gamma=$gamma)'" hydra.run.dir="./outputs/lb-gamma-sweep/$target/gamma$gamma" &
        sleep 5
    done
    wait
done
