import json
import sys

INVALIDATED = {"bandit", "bandit-hybrid", "frozen", "lola-1e-4", "lola-3e-4", "lola-1e-4-cap0.5"}


def main(main_json, fix_json, out):
    main_runs = json.load(open(main_json))["runs"]
    fix_runs = json.load(open(fix_json))["runs"]
    kept = [r for r in main_runs if r["arm"] not in INVALIDATED]
    dropped = sorted({r["name"] for r in main_runs if r["arm"] in INVALIDATED})
    json.dump({"runs": kept + fix_runs, "invalidated": dropped}, open(out, "w"))
    print(f"kept {len(kept)} main-sweep runs, added {len(fix_runs)} fixed runs, dropped {len(dropped)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
