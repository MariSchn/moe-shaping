import json
import sys

import numpy as np

WANDB = "https://wandb.ai/mari-schn/moe-method-comparison/runs/"
FAMILY = {
    "noLB": "none", "noLB-sigmoid": "none", "frozen": "none",
    "aux": "balance", "auxfree": "balance",
    "bilevel": "router", "bandit": "router", "bandit-hybrid": "router",
}
LABEL = {
    "noLB": "Baseline (no LB)", "noLB-sigmoid": "Sigmoid router, no LB", "frozen": "Frozen random router",
    "aux": "Aux-loss LB", "auxfree": "Aux-loss-free LB", "bilevel": "Bilevel (10 router / 40 expert)",
    "bandit": "Bandit (REINFORCE only)", "bandit-hybrid": "Bandit + task gradient",
}
SHOWN_LAYERS = (0, 5, 11)


def lola_label(arm):
    spec = arm[len("lola-"):]
    alpha, _, cap = spec.partition("-cap")
    return f"LOLA α={alpha}" + (f", capped {cap}×" if cap else "")


def mean_series(runs, key, stride):
    steps = sorted({int(s) for r in runs for s in r["series"].get(key, {})})
    out = []
    for step in steps:
        if step % stride and step != steps[-1]:
            continue
        values = [r["series"][key][str(step)] for r in runs if str(step) in r["series"].get(key, {})]
        if values:
            out.append([step, float(np.mean(values))])
    return out


def main(src, dst, first_seed=42):
    data = json.load(open(src))
    by_arm = {}
    for run in data["runs"]:
        by_arm.setdefault(run["arm"], []).append(run)
    arms = []
    for arm, runs in sorted(by_arm.items()):
        done = [r for r in runs if r["complete"]]
        family = FAMILY.get(arm, "router")
        entry = {
            "arm": arm,
            "label": LABEL.get(arm) or lola_label(arm),
            "family": family,
            "seeds": [],
            "diverged": [],
        }
        for r in sorted(runs, key=lambda r: r["seed"]):
            wandb_id = f'{r["name"]}-{r["job"]}' if r["name"] != "cmp-noLB-s43" else "cmp-noLB-s43-3488190"
            if not r["complete"]:
                entry["diverged"].append({"seed": r["seed"], "last_step": r["last_step"], "wandb": WANDB + wandb_id})
                continue
            s = r["spec_summary"]
            layers = r["spec_layers"]
            entry["seeds"].append({
                "seed": r["seed"],
                "wandb": WANDB + wandb_id,
                "loss": r["final_valid_loss"],
                "ppl": r["final_valid_ppl"],
                "maxvio_global": r["final_maxvio_global_valid"],
                "domain_nmi": s["domain_nmi"],
                "vocab_in": s["vocab_specialization_input"],
                "vocab_out": s["vocab_specialization_output"],
                "coact": s["coactivation_max_mean"],
                "eff_experts": s["effective_experts"],
                "dead": s["dead_experts_total"],
                "domain_ppl": s["domain_ppl"],
                "saturation": [[p["fraction"], p["mean"]] for p in s["router_saturation"]],
                "layer_nmi": [l["domain_nmi"] for l in layers],
                "layer_dead": [l["dead_experts"] for l in layers],
                "layer_eff": [l["effective_experts"] for l in layers],
                "layer_maxvio": [l["maxvio_global"] for l in layers],
            })
            if r["seed"] == first_seed:
                entry["domains"] = r["domains"]
                entry["domain_spec"] = {
                    str(i): layers[i]["domain_specialization"] for i in SHOWN_LAYERS
                }
        if done:
            entry["maxvio_batch"] = mean_series(done, "maxvio/batch", 10)
            entry["train_loss"] = mean_series(done, "lm loss", 10)
            entry["valid"] = sorted(
                [int(step), float(np.mean([r["valid"][step] for r in done if step in r["valid"]]))]
                for step in done[0]["valid"]
            )
        arms.append(entry)
    json.dump({"arms": arms, "topk": 8, "num_experts": 64, "num_layers": 12}, open(dst, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
