"""Sets the backup AI's cut-offs on the 32 earlier photos, then checks them once on the 80 fresh photos.
Rules fixed before looking at the fresh photos:
- photo size: the one with the higher area under the ROC curve on the 32 (ties go to the smaller, faster size);
- one cut: the score that gives the best balanced accuracy on the 32;
- REVIEW band: the cut moved down and up to the nearest scores that keep REVIEW at or under 25% of the 32.
Usage: python -I backup_score_pick.py <folder with score jobs>"""
import json
import math
import sys
from pathlib import Path

rows = [r for p in sorted(Path(sys.argv[1]).glob("*/scores.json")) for r in json.loads(p.read_text())["rows"]]
tune = [r for r in rows if r["kind"] == "blind"]
test = [r for r in rows if r["kind"] == "photo"]
print("photos", len(tune), "to tune,", len(test), "to test")
MAX_REVIEW = 0.25


def auc(items, key):
    bad = [r[key] for r in items if r["label"] == "bad"]
    good = [r[key] for r in items if r["label"] == "good"]
    return sum((b > g) + 0.5 * (b == g) for b in bad for g in good) / (len(bad) * len(good))


def balanced(items, key, cut):
    bad = [r for r in items if r["label"] == "bad"]
    good = [r for r in items if r["label"] == "good"]
    return 0.5 * (sum(r[key] >= cut for r in bad) / len(bad) + sum(r[key] < cut for r in good) / len(good))


def wilson(hits, total):
    if not total:
        return 0.0, 0.0
    z = 1.96
    share = hits / total
    scale = 1 + z * z / total
    middle = (share + z * z / (2 * total)) / scale
    spread = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / scale
    return round(100 * (middle - spread), 1), round(100 * (middle + spread), 1)


for side in (448, 640):
    key = f"score_{side}"
    secs = sorted(r[f"seconds_{side}"] for r in rows)
    print(side, "px: AUC tune", round(auc(tune, key), 3), "test", round(auc(test, key), 3), "| median s", secs[len(secs) // 2])
sides = {side: auc(tune, f"score_{side}") for side in (448, 640)}
side = 448 if sides[448] >= sides[640] else 640
key = f"score_{side}"
values = sorted({r[key] for r in tune})
cuts = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
cut = max(cuts, key=lambda c: (balanced(tune, key, c), -abs(c - 0.5)))
print("chosen size", side, "| cut", round(cut, 4), "| balanced on tune", round(100 * balanced(tune, key, cut), 1))
below = sorted((v for v in values if v < cut), reverse=True)
above = sorted(v for v in values if v >= cut)
low, high = cut, cut
for i in range(1, len(tune)):
    new_low = below[i - 1] if i - 1 < len(below) else low
    new_high = above[i - 1] if i - 1 < len(above) else high
    share = sum(new_low < r[key] < new_high for r in tune) / len(tune)
    if share > MAX_REVIEW:
        break
    low, high = new_low, new_high
print("band low", round(low, 4), "high", round(high, 4))


def decide(score):
    return "REJECT" if score >= high else "PASS" if score <= low else "REVIEW"


for name, items in (("tune", tune), ("test", test)):
    out = {d: [r for r in items if decide(r[key]) == d] for d in ("PASS", "REVIEW", "REJECT")}
    bad = [r for r in items if r["label"] == "bad"]
    good = [r for r in items if r["label"] == "good"]
    auto = out["PASS"] + out["REJECT"]
    right = sum(r["label"] == "good" for r in out["PASS"]) + sum(r["label"] == "bad" for r in out["REJECT"])
    print(f"\n{name}: PASS {len(out['PASS'])}, REVIEW {len(out['REVIEW'])}, REJECT {len(out['REJECT'])} of {len(items)}")
    print("  automatic decisions right", right, "/", len(auto), wilson(right, len(auto)))
    print("  PASS right", sum(r["label"] == "good" for r in out["PASS"]), "/", len(out["PASS"]), wilson(sum(r["label"] == "good" for r in out["PASS"]), len(out["PASS"])))
    print("  REJECT right", sum(r["label"] == "bad" for r in out["REJECT"]), "/", len(out["REJECT"]), wilson(sum(r["label"] == "bad" for r in out["REJECT"]), len(out["REJECT"])))
    print("  defects caught (REJECT)", sum(decide(r[key]) == "REJECT" for r in bad), "/", len(bad), "| defects passed", sum(decide(r[key]) == "PASS" for r in bad))
    print("  good passed", sum(decide(r[key]) == "PASS" for r in good), "/", len(good), "| good rejected", sum(decide(r[key]) == "REJECT" for r in good))
    print("  balanced at the cut", round(100 * balanced(items, key, cut), 1))
    by = {}
    for r in items:
        by.setdefault(r["product"], []).append(r)
    for product, group in sorted(by.items()):
        print("   ", product, "bad->", [decide(r[key])[0] for r in group if r["label"] == "bad"], "good->", [decide(r[key])[0] for r in group if r["label"] == "good"])
