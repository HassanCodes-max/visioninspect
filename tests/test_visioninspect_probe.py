import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).parent.parent / "visioninspect" / "probes"))
import one_photo_probe  # noqa: E402
import three_way  # noqa: E402


def rows(groups):
    out = []
    for product, good, bad in groups:
        out += [{"product": product, "label": 0, "prob": s} for s in good]
        out += [{"product": product, "label": 1, "prob": s} for s in bad]
    return out


def test_clean_split_decides_everything():
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    low, high = three_way.choose_cutoffs(scores, np.array([0, 0, 0, 1, 1, 1]))
    assert low == 0.3 and high == 0.7
    assert list(three_way.decide(scores, low, high)) == [0, 0, 0, 1, 1, 1]


def test_scores_in_the_overlap_go_to_review():
    scores = np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9])
    labels = np.array([0, 0, 1, 0, 1, 1, 1])
    low, high = three_way.choose_cutoffs(scores, labels)
    calls = three_way.decide(scores, low, high)
    assert list(calls[:2]) == [0, 0] and list(calls[-2:]) == [1, 1]
    assert (calls[2:4] == -1).all()


def test_no_separation_means_always_review():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    low, high = three_way.choose_cutoffs(scores, np.array([0, 1, 0, 1]))
    assert (three_way.decide(scores, low, high) == -1).all()


def test_cutoffs_never_use_the_type_they_are_applied_to():
    data = rows([("a", [0.1, 0.2], [0.8, 0.9]), ("b", [0.1, 0.2], [0.8, 0.9]), ("shifted", [0.85, 0.95], [0.3, 0.4])])
    calls, truth = three_way.leave_one_type_out(data)
    shifted = [i for i, row in enumerate(data) if row["product"] == "shifted"]
    assert (calls[shifted] != truth[shifted]).any()


def test_good_scale_on_every_type_passes():
    groups = [(name, list(np.linspace(0.05, 0.3, 20)), list(np.linspace(0.7, 0.95, 20))) for name in "abcd"]
    result = three_way.summarise(rows(groups))
    assert result["passed"] and result["coverage"] == 1.0 and result["missed"] == 0


def test_per_type_auroc_can_hide_a_shifted_scale():
    groups = [("a", [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]), ("b", [0.7, 0.75, 0.8], [0.85, 0.9, 0.95])]
    result = three_way.summarise(rows(groups))
    assert result["per_type_auroc"] == 1.0
    assert result["pooled_auroc"] < 1.0
    assert not result["passed"]


def test_probe_reads_the_visa_layout(tmp_path):
    for folder, name in (("Normal", "0001.JPG"), ("Anomaly", "000.JPG")):
        path = tmp_path / "candle" / "Data" / "Images" / folder / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
    good, bad = one_photo_probe.find_products(tmp_path)["candle"]
    assert [p.name for p in good] == ["0001.JPG"] and [p.name for p in bad] == ["000.JPG"]


def test_sampling_is_repeatable():
    paths = list(range(100))
    assert one_photo_probe.sample(paths, 30, "x") == one_photo_probe.sample(paths, 30, "x")
    assert one_photo_probe.sample(paths, 30, "x") != one_photo_probe.sample(paths, 30, "y")
