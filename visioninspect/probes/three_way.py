"""Score a three-way decision on product types the cut-offs never saw. See PLAN_ONE_PHOTO.md."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

TARGET = 0.95
LIMITS = {"correct": 0.95, "coverage": 0.60, "missed": 0.05}


def choose_cutoffs(scores, labels, target=TARGET):
    """Widest automatic ranges that stay at least `target` correct. Empty ranges mean always REVIEW."""
    order = np.argsort(scores)
    ranked, truth = np.asarray(scores)[order], np.asarray(labels)[order]
    good_share = np.cumsum(truth == 0) / np.arange(1, len(truth) + 1)
    bad_share = np.cumsum((truth == 1)[::-1])[::-1] / np.arange(len(truth), 0, -1)
    good_ok = np.flatnonzero((good_share >= target) & (truth == 0))
    bad_ok = np.flatnonzero((bad_share >= target) & (truth == 1))
    low = ranked[good_ok[-1]] if len(good_ok) else -np.inf
    high = ranked[bad_ok[0]] if len(bad_ok) else np.inf
    if low >= high:
        return -np.inf, np.inf
    return float(low), float(high)


def decide(scores, low, high):
    scores = np.asarray(scores)
    return np.where(scores <= low, 0, np.where(scores >= high, 1, -1))


def leave_one_type_out(rows, key="prob"):
    scores = np.array([r[key] for r in rows], dtype=float)
    labels = np.array([r["label"] for r in rows])
    types = np.array([r["product"] for r in rows])
    calls = np.full(len(rows), -1)
    for name in sorted(set(types)):
        low, high = choose_cutoffs(scores[types != name], labels[types != name])
        calls[types == name] = decide(scores[types == name], low, high)
    return calls, labels


def summarise(rows, key="prob"):
    labels = np.array([r["label"] for r in rows])
    scores = np.array([r[key] for r in rows], dtype=float)
    types = sorted({r["product"] for r in rows})
    per_type = {}
    for name in types:
        mask = np.array([r["product"] == name for r in rows])
        per_type[name] = float(roc_auc_score(labels[mask], scores[mask]))
    calls, truth = leave_one_type_out(rows, key)
    made = calls >= 0
    result = {
        "images": len(rows),
        "per_type_auroc": float(np.mean(list(per_type.values()))),
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "coverage": float(made.mean()),
        "correct": float((calls[made] == truth[made]).mean()) if made.any() else 0.0,
        "missed": float(((calls == 0) & (truth == 1)).sum() / max((truth == 1).sum(), 1)),
        "wrongly_rejected": float(((calls == 1) & (truth == 0)).sum() / max((truth == 0).sum(), 1)),
        "review": float((calls == -1).mean()),
        "by_type": per_type,
    }
    result["passed"] = bool(result["correct"] >= LIMITS["correct"] and result["coverage"] >= LIMITS["coverage"]
                            and result["missed"] <= LIMITS["missed"])
    return result


if __name__ == "__main__":
    data = json.loads(Path(sys.argv[1]).read_text())
    for name in ("prob", "map_max"):
        print(name, json.dumps(summarise(data, name), indent=2))
