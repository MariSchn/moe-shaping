import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

SCRATCH = os.path.expandvars("/iopsstor/scratch/cscs/$USER/moe-shaping")
ITER_LINE = re.compile(r"iteration\s+(\d+)/\s*(\d+) \|")
METRIC = re.compile(r"\| ([\w/ ]+?): ([-0-9.E+]+)")
VALID = re.compile(r"validation loss at iteration (\d+)( on validation set)? \| lm loss value: ([0-9.E+-]+)")
MAXVIO_GLOBAL = re.compile(r"MaxVio_global \(valid\) mean over layers ([0-9.]+) \| worst layer ([0-9.]+)")
RUN = re.compile(r"^(?P<prefix>[^-]+)-(?P<arm>.+)-s(?P<seed>\d+)-(?P<job>\d+)\.log$")
SERIES = ("lm loss", "maxvio/batch", "maxvio/batch_worst_layer", "lola/correction_to_task_grad")


def parse_log(path):
    series = defaultdict(dict)
    valid, maxvio_valid = {}, {}
    final = None
    last_valid_iter = None
    total = None
    with open(path, errors="replace") as f:
        for line in f:
            m = ITER_LINE.search(line)
            if m:
                step, total = int(m.group(1)), int(m.group(2))
                for name, value in METRIC.findall(line):
                    name = name.strip()
                    if name in SERIES:
                        series[name][step] = float(value)
                continue
            m = VALID.search(line)
            if m:
                step, loss = int(m.group(1)), float(m.group(3))
                if m.group(2):
                    final = loss
                else:
                    valid[step] = loss
                last_valid_iter = step
                continue
            m = MAXVIO_GLOBAL.search(line)
            if m:
                maxvio_valid[last_valid_iter if last_valid_iter is not None else -1] = float(m.group(1))
    return {
        "total_steps": total,
        "last_step": max(series["lm loss"]) if series["lm loss"] else 0,
        "final_valid_loss": final,
        "valid": valid,
        "maxvio_global_valid": maxvio_valid,
        "series": {k: v for k, v in series.items()},
    }


def run_logs(prefix, excluded_jobs, min_job):
    runs = defaultdict(list)
    for path in glob.glob(f"logs/{prefix}-*-s*-*.log"):
        m = RUN.match(os.path.basename(path))
        if not m or m["prefix"] != prefix or int(m["job"]) in excluded_jobs or int(m["job"]) < min_job:
            continue
        runs[(m["arm"], int(m["seed"]))].append((int(m["job"]), path))
    return {key: sorted(paths) for key, paths in runs.items()}


def parse_run(paths):
    merged = None
    for _, path in paths:
        parsed = parse_log(path)
        if merged is None:
            merged = parsed
            continue
        for key in ("valid", "maxvio_global_valid"):
            merged[key].update(parsed[key])
        for name, points in parsed["series"].items():
            merged["series"].setdefault(name, {}).update(points)
        merged["total_steps"] = parsed["total_steps"] or merged["total_steps"]
        merged["last_step"] = max(merged["last_step"], parsed["last_step"])
        if parsed["final_valid_loss"] is not None:
            merged["final_valid_loss"] = parsed["final_valid_loss"]
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="cmp")
    parser.add_argument("--project", default="moe-method-comparison")
    parser.add_argument("--out", default="results/compare.json")
    parser.add_argument("--exclude-jobs", default="")
    parser.add_argument("--min-job", type=int, default=0)
    args = parser.parse_args()
    excluded = {int(j) for j in args.exclude_jobs.split(",") if j}

    runs = []
    for (arm, seed), paths in sorted(run_logs(args.prefix, excluded, args.min_job).items()):
        parsed = parse_run(paths)
        job = paths[-1][0]
        name = f"{args.prefix}-{arm}-s{seed}"
        spec_path = os.path.join(SCRATCH, args.project, name, "analysis", "specialization.json")
        spec = json.load(open(spec_path)) if os.path.exists(spec_path) else None
        final = parsed["final_valid_loss"]
        runs.append(
            {
                "name": name,
                "arm": arm,
                "seed": seed,
                "job": job,
                "complete": final is not None and spec is not None,
                "final_valid_loss": final,
                "final_valid_ppl": math.exp(final) if final is not None else None,
                "final_maxvio_global_valid": parsed["maxvio_global_valid"].get(parsed["total_steps"]),
                "spec_summary": spec["summary"] if spec else None,
                "spec_layers": [
                    {k: layer[k] for k in ("domain_nmi", "maxvio_global", "effective_experts",
                                           "dead_experts", "coactivation_max_mean",
                                           "vocab_specialization_input", "vocab_specialization_output",
                                           "domain_specialization", "load_share")}
                    for layer in spec["layers"]
                ] if spec else None,
                "domains": spec["domains"] if spec else None,
                **{k: parsed[k] for k in ("total_steps", "last_step", "valid", "maxvio_global_valid", "series")},
            }
        )

    by_arm = defaultdict(list)
    for run in runs:
        if run["complete"]:
            by_arm[run["arm"]].append(run)
    arms = {}
    for arm, members in by_arm.items():
        def stat(get):
            values = [get(r) for r in members]
            values = [v for v in values if v is not None]
            if not values:
                return None
            return {"mean": float(np.mean(values)), "min": float(min(values)),
                    "max": float(max(values)), "n": len(values)}
        summary_keys = ("domain_nmi", "domain_specialization_max_over_domains",
                        "vocab_specialization_input", "vocab_specialization_output",
                        "coactivation_max_mean", "maxvio_global_domains", "effective_experts",
                        "dead_experts_total")
        arms[arm] = {
            "seeds": [r["seed"] for r in members],
            "final_valid_loss": stat(lambda r: r["final_valid_loss"]),
            "final_valid_ppl": stat(lambda r: r["final_valid_ppl"]),
            "maxvio_global_valid": stat(lambda r: r["final_maxvio_global_valid"]),
            "maxvio_batch_last100": stat(lambda r: float(np.mean(
                [v for s, v in r["series"].get("maxvio/batch", {}).items() if s > r["total_steps"] - 100]
            )) if r["series"].get("maxvio/batch") else None),
            **{key: stat(lambda r, key=key: r["spec_summary"][key]) for key in summary_keys},
            "saturation_by_fraction": [
                {"fraction": point["fraction"],
                 "mean": float(np.mean([r["spec_summary"]["router_saturation"][i]["mean"] for r in members]))}
                for i, point in enumerate(members[0]["spec_summary"]["router_saturation"])
            ],
            "domain_ppl": {
                d: float(np.mean([r["spec_summary"]["domain_ppl"][d] for r in members]))
                for d in members[0]["spec_summary"]["domain_ppl"]
            },
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"runs": runs, "arms": arms}, open(args.out, "w"))
    header = f"{'arm':16s} {'n':>2s} {'val ppl':>9s} {'MaxVio_g':>9s} {'dom NMI':>8s} {'vocab in':>8s} {'coact':>6s} {'eff E':>6s} {'dead':>5s}"
    print(header)
    for arm, a in sorted(arms.items(), key=lambda kv: kv[1]["final_valid_ppl"]["mean"]):
        fmt = lambda s, f="{:.3f}": f.format(s["mean"]) if s else "-"
        print(f"{arm:16s} {a['final_valid_ppl']['n']:>2d} {fmt(a['final_valid_ppl'], '{:.2f}'):>9s} "
              f"{fmt(a['maxvio_global_valid']):>9s} {fmt(a['domain_nmi']):>8s} "
              f"{fmt(a['vocab_specialization_input']):>8s} {fmt(a['coactivation_max_mean']):>6s} "
              f"{fmt(a['effective_experts'], '{:.1f}'):>6s} {fmt(a['dead_experts_total'], '{:.0f}'):>5s}")
    incomplete = [r["name"] for r in runs if not r["complete"]]
    if incomplete:
        print("incomplete:", ", ".join(incomplete))


if __name__ == "__main__":
    main()
