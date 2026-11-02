"""Work out the decision cut-offs from Colab's saved scores, and write models/bands.json.

Usage: make_bands.py <known.json> <output json>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "probes"))
import three_way  # noqa: E402

LEFT_OFF = ("pcb3", "mvtec_screw")


def share(values):
    return float(np.mean(values)) if len(values) else 0.0


def measure(rows, target=0.95):
    rows = [r for r in rows if r["product"] not in LEFT_OFF]
    scores = np.array([r["score"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    types = np.array([r["product"] for r in rows])
    calls = np.full(len(rows), -1)
    for name in sorted(set(types)):
        low, high = three_way.choose_cutoffs(scores[types != name], labels[types != name], target)
        calls[types == name] = three_way.decide(scores[types == name], low, high)
    made = calls >= 0
    tested = {"images": len(rows), "automatic": float(made.mean()), "correct": float((calls[made] == labels[made]).mean()),
              "missed": float(((calls == 0) & (labels == 1)).sum() / (labels == 1).sum()),
              "wrongly_rejected": float(((calls == 1) & (labels == 0)).sum() / (labels == 0).sum()), "review": float((~made).mean()),
              "pass_good_share": share(labels[calls == 0] == 0), "reject_defect_share": share(labels[calls == 1] == 1),
              "review_defect_share": share(labels[calls == -1] == 1)}
    low, high = three_way.choose_cutoffs(scores, labels, target)
    bands = {"low": low, "high": high, "pass_good_share": tested["pass_good_share"],
             "reject_defect_share": tested["reject_defect_share"], "review_defect_share": tested["review_defect_share"]}
    return {"bands": bands, "tested": tested}


if __name__ == "__main__":
    result = measure(json.loads(Path(sys.argv[1]).read_text())["rows"])
    target = Path(sys.argv[2])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
