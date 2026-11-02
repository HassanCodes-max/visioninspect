import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
Image = pytest.importorskip("PIL.Image")
sys.path.insert(0, str(Path(__file__).parent.parent / "visioninspect" / "probes"))
import finetune  # noqa: E402

SIZE = finetune.SIZE
PATCHES = finetune.PATCHES


def tiny_model():
    config = transformers.Dinov2Config(hidden_size=24, num_hidden_layers=6, num_attention_heads=2, mlp_ratio=2,
                                       image_size=SIZE, patch_size=14)
    return finetune.Inspector(transformers.Dinov2Model(config), 24)


def photo():
    pixels = np.random.default_rng(0).integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def records(names):
    return [{"product": name, "label": label, "dataset": "mvtec" if name.startswith("mvtec_") else "visa", "path": f"{name}{index}"}
            for name in names for label in (0, 1) for index in range(4)]


def test_cut_paste_marks_what_it_changed():
    original = photo()
    pasted, mask = finetune.cut_paste(original, np.random.default_rng(1))
    changed = (np.asarray(pasted) != np.asarray(original)).any(-1)
    assert mask.any()
    assert not changed[~mask].any()


def test_augment_moves_photo_and_mask_together():
    pixels = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    pixels[100:160, 100:160] = 255
    mask = pixels[..., 0] > 0
    rng = np.random.default_rng(3)
    for _ in range(10):
        image, area = finetune.augment(Image.fromarray(pixels), mask, rng)
        bright = np.asarray(image).mean(-1) > 90
        assert area.shape == (SIZE, SIZE)
        assert (bright == area).mean() > 0.97


def test_grid_needs_a_real_share_of_the_patch():
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[0, 0] = True
    assert finetune.to_grid(mask).sum() == 0
    mask[:14, :14] = True
    grid = finetune.to_grid(mask)
    assert grid[0] == 1 and grid.sum() == 1 and len(grid) == PATCHES


def test_only_the_last_blocks_and_head_train():
    model = tiny_model()
    blocks = model.backbone.encoder.layer
    assert not any(p.requires_grad for p in blocks[0].parameters())
    assert all(p.requires_grad for p in blocks[-1].parameters())
    assert not model.backbone.embeddings.patch_embeddings.projection.weight.requires_grad


def test_earlier_blocks_learn_more_slowly():
    groups = finetune.learning_groups(tiny_model())
    rates = [g["lr"] for g in groups[1:]]
    assert groups[0]["lr"] > rates[0]
    assert rates[1:] == sorted(rates[1:], reverse=True)
    assert len(groups) == 2 + finetune.UNFROZEN


def test_model_gives_one_logit_per_patch_and_trains():
    model = tiny_model()
    images = torch.randn(2, 3, SIZE, SIZE)
    logits = model(images)
    assert logits.shape == (2, PATCHES)
    finetune.total_loss(logits, torch.zeros(2, PATCHES), torch.tensor([0.0, 1.0])).backward()
    assert model.backbone.encoder.layer[-1].mlp.fc1.weight.grad is not None
    assert model.head[0].weight.grad is not None


def test_losses_reward_the_right_answer():
    targets = torch.zeros(1, PATCHES)
    targets[0, :20] = 1
    right, wrong = targets * 20 - 10, (1 - targets) * 20 - 10
    assert finetune.focal_loss(right, targets) < finetune.focal_loss(wrong, targets)
    assert finetune.dice_loss(right, targets) < finetune.dice_loss(wrong, targets)


def test_new_product_splits_keep_types_apart():
    names = [f"visa{i}" for i in range(12)] + [f"mvtec_{i}" for i in range(15)]
    splits = finetune.split_new(records(names))
    assert len(splits) == finetune.FOLDS
    seen = []
    for _, fit, cal, test in splits:
        fit_names, cal_names, test_names = ({r["product"] for r in part} for part in (fit, cal, test))
        assert not fit_names & test_names and not cal_names & test_names and not fit_names & cal_names
        seen += sorted(test_names)
    assert sorted(seen) == sorted(names)


def test_known_split_shares_types_but_not_photos():
    _, fit, cal, test = finetune.split_known(records(["visa0", "mvtec_1"]))
    assert {r["product"] for r in fit} == {r["product"] for r in test}
    used = {(r["product"], r["label"], r["path"]) for r in fit + cal}
    assert not used & {(r["product"], r["label"], r["path"]) for r in test}


def test_phone_folders_are_labelled_by_name(tmp_path):
    for folder, name in (("Image_brokenphones", "a.PNG"), ("Image_whole", "b.png")):
        path = tmp_path / folder / name
        path.parent.mkdir()
        path.write_bytes(b"x")
    good, bad = finetune.phone_photos(tmp_path)
    assert [p.name for p in good] == ["b.png"] and [p.name for p in bad] == ["a.PNG"]


def test_merged_uses_each_rows_own_cutoff():
    rows = [{"label": 1, "score": 0.6, "cutoff": 0.5}, {"label": 1, "score": 0.6, "cutoff": 0.7},
            {"label": 0, "score": 0.6, "cutoff": 0.7}, {"label": 0, "score": 0.6, "cutoff": 0.5}]
    result = finetune.merged(rows)
    assert result["caught"] == 0.5 and result["false_alarm"] == 0.5
