"""Turn model outputs into a decision and plain measurements. No model code lives here."""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

GRID = 32
PATCHES = GRID * GRID
PASS, REVIEW, REJECT = "PASS", "REVIEW", "REJECT"


@dataclass(frozen=True)
class Bands:
    """Cut-offs and what happened to test photos in each band."""
    low: float
    high: float
    pass_good_share: float
    reject_defect_share: float
    review_defect_share: float

    def evidence(self, decision):
        return {PASS: self.pass_good_share, REJECT: self.reject_defect_share}.get(decision, self.review_defect_share)


def decide(score, supported, bands):
    if not supported:
        return REVIEW, "not_supported"
    if score >= bands.high:
        return REJECT, "defect_found"
    if score <= bands.low:
        return PASS, "no_defect_found"
    return REVIEW, "uncertain"


# Share of the AI's answers that were right on 80 fresh photos (results/RESULTS_FRESH.md):
# 23 of 24 "defect" answers and 39 of 56 "good" answers.
AI_EVIDENCE = {REJECT: round(23 / 24, 3), PASS: round(39 / 56, 3)}
# The smaller backup AI on the CPU decides by its defect score (results/RESULTS_BACKUP_AI.md).
# The cut-offs were set on the 32 earlier photos, and the shares were measured on the 80 fresh ones.
BACKUP_BANDS = (0.005912, 0.01243)
BACKUP_EVIDENCE = {PASS: round(35 / 40, 3), REJECT: round(28 / 29, 3)}


def decide_by_ai(explanation):
    """For items without a trained model. A "good" answer passes, and the site shows how often that was right."""
    if explanation is None:
        return REVIEW, "not_supported"
    if explanation["verdict"] == "defect":
        return REJECT, "ai_defect"
    if explanation["verdict"] == "good":
        return PASS, "ai_good"
    return REVIEW, "ai_no_defect"


def decide_by_score(score, bands=None):
    """The backup AI's decision from its defect score, with REVIEW only between the two cut-offs."""
    low, high = bands or BACKUP_BANDS
    if score is None:
        return REVIEW, "not_supported"
    if score >= high:
        return REJECT, "ai_defect"
    if score <= low:
        return PASS, "ai_good"
    return REVIEW, "ai_no_defect"


def find_regions(heat, threshold):
    """Separate marked areas on the 32×32 map, as fractions of the photo."""
    grid = np.asarray(heat, dtype=np.float32).reshape(GRID, GRID) >= threshold
    labelled, count = ndimage.label(grid)
    regions = []
    for index, slices in enumerate(ndimage.find_objects(labelled), start=1):
        rows, columns = slices
        patches = int((labelled[slices] == index).sum())
        regions.append({"x": columns.start / GRID, "y": rows.start / GRID, "width": (columns.stop - columns.start) / GRID,
                        "height": (rows.stop - rows.start) / GRID, "area_percent": round(100 * patches / PATCHES, 2)})
    return sorted(regions, key=lambda region: -region["area_percent"])
