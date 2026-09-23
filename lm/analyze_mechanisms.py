import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

SCRATCH = os.path.expandvars("/iopsstor/scratch/cscs/$USER/moe-shaping/moe-method-comparison")
NUM_EXPERTS, TOPK, DEAD = 64, 8, 0.1


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def snapshot_loads(run):
    files = glob.glob(os.path.join(SCRATCH, run, "analysis", "saturation", "rank*_iter*.pt"))
    by_iter = defaultdict(list)
    for f in files:
        by_iter[int(re.search(r"iter(\d+)", f).group(1))].append(f)
    loads = {}
    for it, paths in sorted(by_iter.items()):
        counts = None
        for p in paths:
            snap = torch.load(p)
            layer_counts = np.stack([
                np.bincount(snap[layer].long().flatten().numpy(), minlength=NUM_EXPERTS)
                for layer in sorted(snap)
            ])
            counts = layer_counts if counts is None else counts + layer_counts
        loads[it] = counts / counts.sum(axis=1, keepdims=True)
    return loads


def death_timeline(run):
    loads = snapshot_loads(run)
    iters = sorted(loads)
    dead = {it: loads[it] < DEAD / NUM_EXPERTS for it in iters}
    final = dead[iters[-1]]
    first = dead[iters[0]]
    ever_alive_after_first = np.zeros_like(final)
    for it in iters[1:]:
        ever_alive_after_first |= ~dead[it]
    return {
        "iters": iters,
        "dead_count": [int(dead[it].sum()) for it in iters],
        "final_dead": int(final.sum()),
        "final_dead_already_dead_at_first": int((final & first).sum()),
        "revived_after_first": int((first & ~final).sum()),
        "dead_per_layer_final": final.sum(axis=1).tolist(),
        "dead_per_layer_first": first.sum(axis=1).tolist(),
        "max_load_share_first": loads[iters[0]].max(axis=1).round(3).tolist(),
    }


def main(compare_json, out):
    data = json.load(open(compare_json))
    runs = {r["name"]: r for r in data["runs"] if r["complete"]}
    arms = defaultdict(list)
    for r in runs.values():
        arms[r["arm"]].append(r)
    result = {"per_layer": {}, "domain_ppl_delta": {}, "layer_corr": {}, "deaths": {}}

    base = arms["noLB"]
    base_dom = {d: np.mean([r["spec_summary"]["domain_loss"][d] for r in base]) for d in base[0]["domains"]}
    for arm, members in sorted(arms.items()):
        layers = [r["spec_layers"] for r in members]
        per_layer = {
            k: np.mean([[l[k] for l in ls] for ls in layers], axis=0).round(3).tolist()
            for k in ("dead_experts", "effective_experts", "domain_nmi", "maxvio_global")
        }
        result["per_layer"][arm] = per_layer
        dom = {d: np.mean([r["spec_summary"]["domain_loss"][d] for r in members]) for d in base_dom}
        result["domain_ppl_delta"][arm] = {d: round(float(dom[d] - base_dom[d]), 4) for d in dom}
        effs = np.concatenate([[l["effective_experts"] for l in ls] for ls in layers])
        nmis = np.concatenate([[l["domain_nmi"] for l in ls] for ls in layers])
        result["layer_corr"][arm] = round(spearman(effs, nmis), 3)

    for arm in ("noLB", "noLB-sigmoid", "aux", "auxfree", "bilevel", "bandit-hybrid", "frozen"):
        for r in arms.get(arm, []):
            result["deaths"][r["name"]] = death_timeline(r["name"])

    json.dump(result, open(out, "w"), indent=1)
    print("dead experts per layer (final, seed-mean):")
    for arm in ("noLB", "noLB-sigmoid", "bandit-hybrid", "bilevel", "frozen", "aux", "auxfree"):
        print(f"  {arm:14s}", result["per_layer"][arm]["dead_experts"])
    print("domain NMI per layer:")
    for arm in ("noLB", "aux", "auxfree"):
        print(f"  {arm:14s}", result["per_layer"][arm]["domain_nmi"])
    print("layer-level Spearman(effective experts, domain NMI) within arm:")
    print("  ", result["layer_corr"])
    print("domain loss delta vs noLB (nats):")
    for arm in ("aux", "auxfree", "noLB-sigmoid", "frozen", "bilevel", "bandit-hybrid"):
        print(f"  {arm:14s}", result["domain_ppl_delta"][arm])
    print("expert deaths over training (probe set):")
    for name, d in result["deaths"].items():
        print(f"  {name:24s} dead@{d['iters'][0]}={d['dead_count'][0]:4d} final={d['final_dead']:4d} "
              f"final-dead-already-at-first={d['final_dead_already_dead_at_first']:4d} revived={d['revived_after_first']:3d} "
              f"series={d['dead_count']}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
