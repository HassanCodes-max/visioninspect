import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("PIL.Image")
sys.path.insert(0, str(Path(__file__).parent.parent / "visioninspect" / "probes"))
import vit_probe  # noqa: E402

PATCHES = vit_probe.PATCHES


def fake_store(count=80, size=24):
    """Defective photos have a block of patches pointing a new way, seen only on those patches."""
    rng = np.random.default_rng(0)
    tokens = rng.normal(0, 0.05, (count, PATCHES, size)).astype(np.float32)
    tokens[..., 0] += 1
    labels = np.array([0, 1] * (count // 2))
    grids = np.zeros((count, PATCHES), dtype=bool)
    for index in np.flatnonzero(labels):
        grids[index, 40:60] = True
        tokens[index, 40:60, 1] += 2
    unit = torch.nn.functional.normalize(torch.from_numpy(tokens), dim=-1)
    names = np.array(["a", "a", "b", "b"] * (count // 4))
    return vit_probe.Store(unit.half(), unit.mean(1), vit_probe.self_contrast(unit).half(), grids, labels, names, np.array(["visa"] * count))


def test_self_contrast_marks_the_odd_patch():
    tokens = torch.ones(1, 64, 4) + 0.01 * torch.randn(1, 64, 4)
    tokens[0, 7] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    contrast = vit_probe.self_contrast(torch.nn.functional.normalize(tokens, dim=-1), neighbours=4)
    assert contrast[0].argmax() == 7


def test_every_type_is_left_out_exactly_once():
    names = [f"visa{i}" for i in range(12)] + [f"mvtec_{i}" for i in range(15)]
    folds = vit_probe.group_folds(names)
    assert sorted(sum(folds, [])) == sorted(names)
    assert len(folds) == vit_probe.FOLDS
    assert max(len(f) for f in folds) - min(len(f) for f in folds) <= 1


def test_folds_spread_each_dataset():
    names = [f"visa{i}" for i in range(12)] + [f"mvtec_{i}" for i in range(15)]
    for fold in vit_probe.group_folds(names):
        assert sum(n.startswith("mvtec_") for n in fold) >= 1
        assert sum(not n.startswith("mvtec_") for n in fold) >= 1


def test_cutoff_balances_both_classes():
    scores = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]
    cutoff = vit_probe.best_cutoff(scores, [0, 0, 0, 1, 1, 1])
    assert 0.3 < cutoff <= 0.6
    assert vit_probe.measure(scores, [0, 0, 0, 1, 1, 1], cutoff)["balanced"] == 1


def test_balanced_accuracy_is_not_fooled_by_rare_defects():
    labels = [0] * 98 + [1] * 2
    result = vit_probe.measure([0.1] * 100, labels, 0.5)
    assert result["balanced"] == 0.5
    assert result["caught"] == 0


def test_top_mean_keeps_a_small_hot_spot():
    values = torch.zeros(2, 1000)
    values[1, :10] = 1
    assert vit_probe.top_mean(values)[1] == 1
    assert vit_probe.top_mean(values)[0] == 0


def test_patch_draw_keeps_defects_and_a_fixed_size():
    store = fake_store(8)
    images, patches, targets = vit_probe.draw_patches(store, np.arange(4), np.random.default_rng(0))
    assert len(targets) == 4 * vit_probe.PATCHES_PER_IMAGE
    assert targets[images == 1].sum() == 20
    assert targets[images == 0].sum() == 0
    assert store.grids[images[targets == 1], patches[targets == 1]].all()


def test_head_learns_a_planted_defect():
    store = fake_store()
    head = vit_probe.train_head(store, np.arange(60), True, "cpu", 0)
    scores = vit_probe.score_images(store, head, np.arange(60, 80), True, "cpu")
    labels = store.labels[60:80]
    assert scores[labels == 1].min() > scores[labels == 0].max()


def test_mvtec_layout_is_read(tmp_path):
    base = tmp_path / "bottle"
    for path in ("train/good/000.png", "test/good/000.png", "test/broken/000.png", "ground_truth/broken/000_mask.png"):
        target = base / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    kind = vit_probe.mvtec_types(tmp_path)[0]
    assert kind["name"] == "mvtec_bottle" and len(kind["good"]) == 2
    assert kind["bad"][0][1].exists()


def test_missing_mask_gives_an_empty_grid(tmp_path):
    assert not vit_probe.mask_grid(tmp_path / "missing.png").any()
    assert not vit_probe.mask_grid(None).any()
