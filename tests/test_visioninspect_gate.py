import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL.Image")
PROBES = Path(__file__).parent.parent / "visioninspect" / "probes"
sys.path.insert(0, str(PROBES))
import gate_probe  # noqa: E402
import product_cutoff  # noqa: E402

spec = importlib.util.spec_from_file_location("vi_build_gate", PROBES / "build_gate.py")
build_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_gate)


def clustered(types=6, per_type=40, size=32, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1, (types, size))
    rows = np.concatenate([centres[i] + rng.normal(0, spread, (per_type, size)) for i in range(types)])
    names = np.repeat([f"t{i}" for i in range(types)], per_type)
    labels = np.tile([0, 1], len(rows) // 2)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True), names, labels


def unit(rows):
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_closeness_rewards_a_near_match():
    gallery = np.eye(4)
    assert gate_probe.closeness(np.eye(4)[:1], gallery, 1)[0] == 1
    assert gate_probe.closeness(unit(np.array([[1.0, 1, 0, 0]])), gallery, 1)[0] < 1


def test_split_is_even_within_product_and_label():
    names = np.repeat(["a", "b"], 8)
    labels = np.tile([0, 1], 8)
    half = gate_probe.split_half(names, labels)
    assert half.sum() == 8
    assert half[(names == "a") & (labels == 1)].sum() == 2


def test_separated_products_pass():
    features, names, labels = clustered(per_type=300)
    outside = {"far": unit(np.random.default_rng(1).normal(0, 1, (30, 32)))}
    result = gate_probe.evaluate(features, names, labels, outside)
    assert result["passed"]
    assert result["right_product"] == 1
    assert result["left_out_review"] > 0.9


def test_identical_products_cannot_pass():
    features, names, labels = clustered(types=1, per_type=120)
    names = np.array([f"t{i % 3}" for i in range(120)])
    result = gate_probe.evaluate(features, names, labels, {"same": features[:20]})
    assert not result["passed"]


def test_good_photo_cutoff_never_scores_its_own_photos():
    labels = np.array([0] * 6 + [1] * 2)
    scores = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8])
    caught, alarm = product_cutoff.one_product(scores, labels, 5, np.random.default_rng(0))
    assert 0 <= caught <= 1
    assert alarm in (0.0, 1.0)


def test_kernel_builds_into_one_file():
    code = build_gate.build()
    assert "from vit_probe import" not in code
    assert code.count('if __name__ == "__main__":') == 1
    assert "def run_gate()" in code and "def visa_types(root)" in code
