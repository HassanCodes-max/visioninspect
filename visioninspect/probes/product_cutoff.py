"""Would a few good photos of a new product set its cut-off? Uses scores saved by finetune.py, with no retraining.

This was chosen after seeing the Colab results, so it is a lead to test and not a measured claim.
Usage: product_cutoff.py <results folder>
"""
import json
import sys
from pathlib import Path

import numpy as np

SHOTS = (3, 5, 10, 20)
REPEATS = 50


def load_rows(folder):
    rows = []
    for path in sorted(Path(folder).glob("new_*.json")):
        rows += json.loads(path.read_text())["rows"]
    return rows


def to_logit(values):
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def one_product(scores, labels, shots, rng):
    """Use `shots` good photos for the cut-off, and score only the other photos."""
    good = np.flatnonzero(labels == 0)
    bad = np.flatnonzero(labels == 1)
    picked = rng.choice(good, shots, replace=False)
    rest = np.setdiff1d(good, picked)
    values = to_logit(scores)
    cutoff = values[picked].mean() + 3 * max(values[picked].std(ddof=1), 0.5)
    return (values[bad] >= cutoff).mean(), (values[rest] >= cutoff).mean()


def evaluate(rows, shots, repeats=REPEATS, seed=0):
    rng = np.random.default_rng(seed)
    table = {}
    for name in sorted({r["product"] for r in rows}):
        part = [r for r in rows if r["product"] == name]
        scores = np.array([r["score"] for r in part])
        labels = np.array([r["label"] for r in part])
        results = [one_product(scores, labels, shots, rng) for _ in range(repeats)]
        caught, alarm = np.mean(results, axis=0)
        table[name] = {"dataset": part[0]["dataset"], "caught": float(caught), "false_alarm": float(alarm),
                       "balanced": float((caught + 1 - alarm) / 2)}
    return table


def overall(table, dataset=None):
    part = [t for t in table.values() if dataset is None or t["dataset"] == dataset]
    return {k: float(np.mean([t[k] for t in part])) for k in ("balanced", "caught", "false_alarm")}


if __name__ == "__main__":
    rows = load_rows(sys.argv[1])
    for shots in SHOTS:
        table = evaluate(rows, shots)
        print(shots, "good photos", {k: round(v, 3) for k, v in overall(table).items()},
              "visa", round(overall(table, "visa")["balanced"], 3), "mvtec", round(overall(table, "mvtec")["balanced"], 3),
              "types at 85%+", sum(t["balanced"] >= 0.85 for t in table.values()))
    for name, row in sorted(evaluate(rows, 5).items(), key=lambda item: -item[1]["balanced"]):
        print(f"  {name:22} {row['balanced']:.3f} caught {row['caught']:.3f} false alarm {row['false_alarm']:.3f}")
