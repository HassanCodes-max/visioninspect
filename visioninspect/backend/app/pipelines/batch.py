"""Batch statistics for industrial use. All numbers come from this code. The AI only suggests causes and actions."""
import json
import math
import re
from collections import Counter

Z_95 = 1.96
DEFAULT_LIMIT = 5.0
PROMPT = """You are a quality engineer writing for a factory or shop. These are the results of checking photos of one batch:
{facts}
Reply with JSON only, with these keys:
"causes": up to 3 short likely causes of these defects in production, handling or storage
"actions": up to 3 short practical next steps
Do not write any numbers. Do not claim anything is certain."""


def wilson(hits, total):
    """95% range for a share measured on a small sample, as percentages."""
    if not total:
        return 0.0, 0.0
    share = hits / total
    scale = 1 + Z_95 ** 2 / total
    middle = (share + Z_95 ** 2 / (2 * total)) / scale
    spread = Z_95 * math.sqrt(share * (1 - share) / total + Z_95 ** 2 / (4 * total ** 2)) / scale
    return round(100 * max(0.0, middle - spread), 1), round(100 * min(1.0, middle + spread), 1)


def lot_advice(rejected, review, total, limit):
    """Compare the batch with the buyer's limit, allowing for chance and for photos still waiting on a person."""
    if not total:
        return "no_data"
    low, high = wilson(rejected, total)
    if low > limit:
        return "hold"
    worst = wilson(rejected + review, total)[1]
    return "accept" if worst <= limit else "check_more"


def stock_estimate(rejected, total, stock):
    if not stock or not total:
        return None
    low, high = wilson(rejected, total)
    return {"stock": stock, "likely": round(stock * rejected / total), "low": round(stock * low / 100), "high": round(stock * high / 100)}


def analyse(results, limit=DEFAULT_LIMIT, stock=None):
    done = [r for r in results if "decision" in r]
    total = len(done)
    rejected = sum(r["decision"] == "REJECT" for r in done)
    review = sum(r["decision"] == "REVIEW" for r in done)
    found = [r["explanation"] for r in done if r["decision"] == "REJECT" and r.get("explanation") and r["explanation"].get("defect")]
    counts = Counter(item["defect"].lower() for item in found)
    named = sum(counts.values())
    running = 0
    pareto = []
    for name, count in counts.most_common():
        running += count
        pareto.append({"defect": name, "count": count, "share": round(100 * count / named), "running_share": round(100 * running / named)})
    areas = [r["affected_percent"] for r in done if r["decision"] == "REJECT" and r.get("affected_percent")]
    low, high = wilson(rejected, total)
    return {"photos": total, "rejected": rejected, "review": review, "passed": total - rejected - review,
            "defect_rate": round(100 * rejected / total, 1) if total else 0.0, "range_low": low, "range_high": high,
            "limit": limit, "advice": lot_advice(rejected, review, total, limit), "pareto": pareto,
            "severity": dict(Counter(item["severity"] for item in found if item.get("severity"))),
            "mean_area": round(sum(areas) / len(areas), 2) if areas else 0.0, "stock": stock_estimate(rejected, total, stock),
            "causes": [], "actions": []}


def facts_for_ai(stats):
    lines = [f"{stats['photos']} photos checked, {stats['rejected']} rejected, {stats['review']} need a person to look."]
    if stats["pareto"]:
        lines.append("Defects found: " + ", ".join(f"{p['defect']} ({p['count']})" for p in stats["pareto"]))
    if stats["severity"]:
        lines.append("Severity: " + ", ".join(f"{name} ({count})" for name, count in stats["severity"].items()))
    return "\n".join(lines)


def parse_advice(text):
    """Keep only short plain suggestions. Anything with a digit is dropped, so the AI cannot add its own numbers."""
    found = re.search(r"\{.*\}", text, re.S)
    try:
        reply = json.loads(found.group(0)) if found else {}
    except json.JSONDecodeError:
        reply = {}

    def tidy(key):
        items = reply.get(key)
        if not isinstance(items, list):
            return []
        kept = [item.strip()[:160] for item in items if isinstance(item, str) and item.strip() and not re.search(r"\d", item)]
        return kept[:3]

    return {"causes": tidy("causes"), "actions": tidy("actions")}
