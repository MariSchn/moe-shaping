#!/usr/bin/env python3.11
"""Collect metrics from the lm/ sweep runs into a single JSON summary.

Two sources, because neither is complete on its own:

  * the SLURM stdout log (logs/<job>-<id>.log) carries the per-iteration training
    line and the periodic validation losses, but Megatron drops any metric whose
    window average is <= 0 from that line, so signed diagnostics never survive it;
  * the TensorBoard event files under the run's scratch log dir carry everything
    that reached the writer, including the signed "bandit/*" diagnostics.

Event files are read with a small self-contained TFRecord/protobuf scanner so this
script needs nothing beyond the standard library (the login node has no
tensorboard install).

Usage:
    python3.11 analyze_runs.py 's2-*' --out results/s2.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import struct
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

SCRATCH = f"/iopsstor/scratch/cscs/{os.environ.get('USER', '')}/moe-shaping"
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

ITER_RE = re.compile(r" iteration\s+(\d+)/\s*(\d+) \|(.*)")
VAL_RE = re.compile(r"validation loss at iteration (\d+) \| lm loss value: ([0-9.E+-]+)")
FIELD_RE = re.compile(r"([A-Za-z0-9_/ ()-]+?):\s*([0-9.E+-]+)")

# Fields worth carrying through from the per-iteration training line.
KEEP = {
    "lm loss": "lm_loss",
    "expert_imbalance": "expert_imbalance",
    "load_balancing_loss": "load_balancing_loss",
    "router/entropy": "router_entropy",
    "load/max": "load_max",
    "load/min": "load_min",
    "load/p10": "load_p10",
    "load/p50": "load_p50",
    "load/p90": "load_p90",
    "grad norm": "grad_norm",
    "tokens/sec/GPU": "tokens_per_sec_per_gpu",
    "elapsed time per iteration (ms)": "ms_per_iter",
    "learning rate": "learning_rate",
}


# --------------------------------------------------------------------------
# stdout log
# --------------------------------------------------------------------------


def parse_stdout(path: str) -> Dict[str, Any]:
    """Pull the per-iteration series and the validation losses out of a job log."""
    series: Dict[str, List[Tuple[int, float]]] = {}
    validation: List[Tuple[int, float]] = []
    total_iters = None
    finished = False

    with open(path, errors="replace") as handle:
        for line in handle:
            match = VAL_RE.search(line)
            if match:
                validation.append((int(match.group(1)), float(match.group(2))))
                continue
            if "END TIME:" in line:
                finished = True
            match = ITER_RE.search(line)
            if not match:
                continue
            iteration = int(match.group(1))
            total_iters = int(match.group(2))
            for raw_name, value in FIELD_RE.findall(match.group(3)):
                name = raw_name.strip()
                if name in KEEP:
                    series.setdefault(KEEP[name], []).append((iteration, float(value)))

    return {
        "series": series,
        "validation": validation,
        "total_iters": total_iters,
        "finished": finished,
    }


# --------------------------------------------------------------------------
# TensorBoard event files
# --------------------------------------------------------------------------


def _varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _fields(buf: bytes) -> Iterator[Tuple[int, int, Any]]:
    """Yield (field_number, wire_type, payload) for one protobuf message."""
    pos = 0
    end = len(buf)
    while pos < end:
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value, pos = buf[pos : pos + length], pos + length
        elif wire == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            return
        yield field, wire, value


def _tfrecords(path: str) -> Iterator[bytes]:
    """Yield the payload of each TFRecord in the file, stopping at a truncated tail."""
    with open(path, "rb") as handle:
        while True:
            header = handle.read(8)
            if len(header) < 8:
                return
            (length,) = struct.unpack("<Q", header)
            handle.read(4)  # length crc
            payload = handle.read(length)
            if len(payload) < length:
                return
            handle.read(4)  # payload crc
            yield payload


def _scalar_from_tensor(buf: bytes) -> Optional[float]:
    """Read a single float out of a TensorProto (float_val or tensor_content)."""
    for field, _wire, value in _fields(buf):
        if field == 5 and len(value) >= 4:  # packed float_val
            return struct.unpack("<f", value[:4])[0]
        if field == 4 and len(value) >= 4:  # tensor_content
            return struct.unpack("<f", value[:4])[0]
    return None


def parse_tensorboard(directory: str) -> Dict[str, List[Tuple[int, float]]]:
    """Read every scalar series from the event files under a TensorBoard dir."""
    scalars: Dict[str, List[Tuple[int, float]]] = {}
    files = sorted(glob.glob(os.path.join(directory, "**", "events.out.tfevents.*"), recursive=True))
    for path in files:
        for payload in _tfrecords(path):
            step = 0
            summary = None
            for field, _wire, value in _fields(payload):
                if field == 2:
                    step = value
                elif field == 5:
                    summary = value
            if summary is None:
                continue
            for field, _wire, value in _fields(summary):
                if field != 1:
                    continue
                tag = None
                scalar = None
                for vfield, _vwire, vvalue in _fields(value):
                    if vfield == 1:
                        tag = vvalue.decode("utf-8", "replace")
                    elif vfield == 2:
                        scalar = struct.unpack("<f", vvalue)[0]
                    elif vfield == 8:
                        scalar = _scalar_from_tensor(vvalue)
                if tag is not None and scalar is not None:
                    scalars.setdefault(tag, []).append((step, scalar))
    for series in scalars.values():
        series.sort()
    return scalars


# --------------------------------------------------------------------------
# run assembly
# --------------------------------------------------------------------------


ARG_RE = re.compile(r"^\s+(--[a-z0-9-]+)(?:\s+(\S+))?\s*$")


def parse_sbatch(path: str) -> Dict[str, str]:
    """Recover the training flags a run was launched with from its sbatch script."""
    args: Dict[str, str] = {}
    with open(path, errors="replace") as handle:
        for line in handle:
            match = ARG_RE.match(line.rstrip("\n"))
            if match:
                args[match.group(1)] = match.group(2) if match.group(2) else "true"
    return args


def tail_mean(series: List[Tuple[int, float]], fraction: float = 0.1) -> Optional[float]:
    """Mean over the last `fraction` of a series, to smooth step-to-step noise."""
    if not series:
        return None
    count = max(1, int(len(series) * fraction))
    return sum(value for _step, value in series[-count:]) / count


def collect(pattern: str) -> List[Dict[str, Any]]:
    """Assemble one record per run whose job name matches `pattern`."""
    runs = []
    for script in sorted(glob.glob(os.path.join(LOG_DIR, f"{pattern}.sbatch"))):
        name = os.path.basename(script)[: -len(".sbatch")]
        logs = sorted(glob.glob(os.path.join(LOG_DIR, f"{name}-*.log")))
        if not logs:
            continue
        stdout = parse_stdout(logs[-1])
        args = parse_sbatch(script)
        exp_name = None
        with open(script, errors="replace") as handle:
            for line in handle:
                if line.startswith("EXP_NAME="):
                    exp_name = line.split("=", 1)[1].strip()
                    break
        tensorboard = {}
        if exp_name:
            tb_dir = os.path.join(SCRATCH, "moe-shaping-clariden", exp_name, "tensorboard")
            if os.path.isdir(tb_dir):
                tensorboard = parse_tensorboard(tb_dir)

        validation = stdout["validation"]
        runs.append(
            {
                "name": name,
                "log": os.path.basename(logs[-1]),
                "finished": stdout["finished"],
                "total_iters": stdout["total_iters"],
                "last_iter": (
                    stdout["series"].get("lm_loss", [(0, 0)])[-1][0]
                    if stdout["series"].get("lm_loss")
                    else 0
                ),
                "args": args,
                "final_val_loss": validation[-1][1] if validation else None,
                "best_val_loss": min((v for _i, v in validation), default=None),
                "validation": validation,
                "train_loss_tail": tail_mean(stdout["series"].get("lm_loss", [])),
                "summary": {
                    key: tail_mean(series)
                    for key, series in stdout["series"].items()
                    if key != "lm_loss"
                },
                "bandit": {
                    tag: tail_mean(series)
                    for tag, series in tensorboard.items()
                    if tag.startswith("bandit/")
                },
                "series": stdout["series"],
                "tensorboard": {
                    tag: series
                    for tag, series in tensorboard.items()
                    if tag.startswith("bandit/") or tag in ("router/entropy",)
                },
            }
        )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", help="job-name glob, e.g. 's2-*'")
    parser.add_argument("--out", default=None, help="write the full JSON here")
    parser.add_argument("--quiet", action="store_true", help="suppress the table")
    options = parser.parse_args()

    runs = collect(options.pattern)
    if not runs:
        print(f"no runs matching {options.pattern!r}", file=sys.stderr)
        sys.exit(1)

    if not options.quiet:
        header = f"{'run':<26} {'iters':>12} {'val':>8} {'train':>8} {'imbal':>7} {'ent':>6} {'ldmax':>7}"
        print(header)
        print("-" * len(header))
        for run in runs:
            summary = run["summary"]
            print(
                f"{run['name']:<26} "
                f"{run['last_iter']:>5}/{run['total_iters'] or 0:<6} "
                f"{_fmt(run['final_val_loss']):>8} "
                f"{_fmt(run['train_loss_tail']):>8} "
                f"{_fmt(summary.get('expert_imbalance')):>7} "
                f"{_fmt(summary.get('router_entropy')):>6} "
                f"{_fmt(summary.get('load_max')):>7}"
            )
            if run["bandit"]:
                extras = "  ".join(f"{k.split('/')[-1]}={_fmt(v)}" for k, v in sorted(run["bandit"].items()))
                print(f"{'':<26}   {extras}")

    if options.out:
        os.makedirs(os.path.dirname(os.path.abspath(options.out)), exist_ok=True)
        with open(options.out, "w") as handle:
            json.dump(runs, handle)
        print(f"\nwrote {options.out} ({len(runs)} runs)")


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
