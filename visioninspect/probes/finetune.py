"""Stage 2 of PLAN_VIT_85.md: fine-tune the last DINOv2 blocks on a GPU. Saves after every split, so it can resume."""
import io
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vit_probe import (CAL_SHARE, GRID, MEAN, PATCHES, SEED, SIZE, STD, best_cutoff, group_folds, mask_grid, measure,
                       mvtec_types, pick, self_contrast, top_mean, visa_types)

DATA = Path(os.environ.get("TUNE_DATA", "/content/data"))
OUT = Path(os.environ.get("TUNE_OUT", "/content/out"))
CACHE = Path(os.environ.get("TUNE_CACHE", "/content/cache"))
MODEL = os.environ.get("TUNE_MODEL", "facebook/dinov2-base")
EPOCHS = int(os.environ.get("TUNE_EPOCHS", "5"))
BATCH = int(os.environ.get("TUNE_BATCH", "8"))
FOLDS = 3
UNFROZEN = 4
HEAD_LR = 1e-3
TOP_LR = 5e-5
BLOCK_DECAY = 0.75
PHONE_PER_CLASS = 150
CUT_PASTE = 0.5
PATCH_SHARE = 0.15


def prepare(kinds):
    """Resize once, so training reads small files. Returns one record per photo."""
    records = []
    for kind in kinds:
        good, bad = pick(kind)
        folder = CACHE / kind["name"]
        folder.mkdir(parents=True, exist_ok=True)
        for index, (path, mask) in enumerate([(p, None) for p in good] + bad):
            target = folder / f"{index:04d}.jpg"
            grid = folder / f"{index:04d}.npy"
            if not target.exists() or not grid.exists():
                Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR).save(target, quality=95)
                np.save(grid, np.packbits(mask_full(mask)))
            records.append({"path": str(target), "grid": str(grid), "label": int(mask is not None), "product": kind["name"],
                            "dataset": kind["dataset"]})
    return records


def mask_full(path):
    if path is None or not Path(path).exists():
        return np.zeros((SIZE, SIZE), dtype=bool)
    return np.asarray(Image.open(path).convert("L").point(lambda value: 255 if value else 0).resize((SIZE, SIZE), Image.NEAREST)) > 0


def load_mask(path):
    return np.unpackbits(np.load(path))[:SIZE * SIZE].reshape(SIZE, SIZE).astype(bool)


def cut_paste(image, rng):
    """Paste a turned scrap of the photo elsewhere on it, as a synthetic defect."""
    width = int(rng.integers(SIZE // 20, SIZE // 5))
    height = int(rng.integers(SIZE // 40, SIZE // 8))
    left, top = (int(rng.integers(0, SIZE - side)) for side in (width, height))
    scrap = image.crop((left, top, left + width, top + height)).convert("RGBA")
    scrap = ImageEnhance.Brightness(scrap).enhance(float(rng.uniform(0.7, 1.3))).rotate(float(rng.uniform(0, 360)), expand=True)
    x, y = (int(rng.integers(SIZE // 6, 5 * SIZE // 6 - side)) for side in scrap.size)
    pasted = image.copy()
    pasted.paste(scrap, (x, y), scrap)
    area = Image.new("L", (SIZE, SIZE))
    area.paste(scrap.getchannel("A"), (x, y))
    return pasted, np.asarray(area) > 0


def augment(image, mask, rng):
    """Same geometry for photo and mask. Lighting, blur and JPEG imitate other cameras."""
    area = Image.fromarray(mask.astype(np.uint8) * 255)
    if rng.random() < 0.5:
        image, area = image.transpose(Image.FLIP_LEFT_RIGHT), area.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        image, area = image.transpose(Image.FLIP_TOP_BOTTOM), area.transpose(Image.FLIP_TOP_BOTTOM)
    turns = int(rng.integers(0, 4))
    if turns:
        turn = (Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270)[turns - 1]
        image, area = image.transpose(turn), area.transpose(turn)
    keep = float(rng.uniform(0.8, 1.0))
    side = int(SIZE * keep)
    left, top = (int(rng.integers(0, SIZE - side + 1)) for _ in range(2))
    box = (left, top, left + side, top + side)
    image = image.resize((SIZE, SIZE), Image.BILINEAR, box=box)
    area = area.resize((SIZE, SIZE), Image.NEAREST, box=box)
    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.75, 1.25)))
    image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.8, 1.2)))
    image = ImageEnhance.Color(image).enhance(float(rng.uniform(0.8, 1.2)))
    if rng.random() < 0.2:
        image = image.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.2))))
    if rng.random() < 0.3:
        stream = io.BytesIO()
        image.save(stream, "JPEG", quality=int(rng.integers(50, 90)))
        image = Image.open(io.BytesIO(stream.getvalue())).convert("RGB")
    return image, np.asarray(area) > 0


def to_tensor(image):
    pixels = (np.asarray(image, dtype=np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(pixels.transpose(2, 0, 1))


def to_grid(mask):
    share = torch.from_numpy(mask.astype(np.float32))[None, None]
    return (nn.functional.avg_pool2d(share, 14)[0, 0] >= PATCH_SHARE).float().reshape(PATCHES)


class Photos(Dataset):
    def __init__(self, records, training):
        self.records, self.training = records, training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        if not self.training:
            return to_tensor(image), torch.zeros(PATCHES), float(record["label"])
        rng = np.random.default_rng()
        mask, label = load_mask(record["grid"]), record["label"]
        if not label and rng.random() < CUT_PASTE:
            image, mask = cut_paste(image, rng)
            label = 1
        image, mask = augment(image, mask, rng)
        return to_tensor(image), to_grid(mask), float(label)


class Inspector(nn.Module):
    """Patch tokens, self-comparison, then one defect logit per patch."""

    def __init__(self, backbone, size):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(2 * size + 1, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 1))
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        for block in backbone.encoder.layer[-UNFROZEN:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in backbone.layernorm.parameters():
            parameter.requires_grad = True

    def forward(self, images):
        tokens = self.backbone(pixel_values=images).last_hidden_state[:, 1:]
        unit = nn.functional.normalize(tokens, dim=-1)
        with torch.no_grad():
            contrast = self_contrast(unit.detach().float())
        features = torch.cat([unit, unit - unit.mean(1, keepdim=True), contrast.unsqueeze(-1).to(unit.dtype)], -1)
        return self.head(features).squeeze(-1)


def learning_groups(model):
    groups = [{"params": list(model.head.parameters()), "lr": HEAD_LR},
              {"params": list(model.backbone.layernorm.parameters()), "lr": TOP_LR}]
    for depth, block in enumerate(reversed(list(model.backbone.encoder.layer[-UNFROZEN:]))):
        groups.append({"params": list(block.parameters()), "lr": TOP_LR * BLOCK_DECAY ** depth})
    return groups


def focal_loss(logits, targets, gamma=2.0, alpha=0.75):
    plain = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    chance = torch.exp(-plain)
    weight = alpha * targets + (1 - alpha) * (1 - targets)
    return (weight * (1 - chance) ** gamma * plain).mean()


def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    shared = (probs * targets).sum(-1)
    return (1 - (2 * shared + smooth) / (probs.sum(-1) + targets.sum(-1) + smooth)).mean()


def total_loss(logits, targets, labels):
    image = nn.functional.binary_cross_entropy_with_logits(top_mean(logits), labels)
    return focal_loss(logits, targets) + dice_loss(logits, targets) + image


def build_model(device):
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(MODEL)
    return Inspector(backbone, backbone.config.hidden_size).to(device)


def train(records, device, seed):
    torch.manual_seed(seed)
    model = build_model(device)
    loader = DataLoader(Photos(records, True), batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
    optimiser = torch.optim.AdamW(learning_groups(model), weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, EPOCHS * len(loader))
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    for epoch in range(EPOCHS):
        model.train()
        model.backbone.embeddings.eval()
        total, started = 0.0, time.time()
        for images, grids, labels in loader:
            images, grids, labels = images.to(device), grids.to(device), labels.to(device, dtype=torch.float32)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                loss = total_loss(model(images), grids, labels)
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            total += float(loss)
        print(f"epoch {epoch + 1}/{EPOCHS} loss {total / max(len(loader), 1):.3f} seconds {time.time() - started:.0f}", flush=True)
    return model.eval()


def score(model, records, device):
    loader = DataLoader(Photos(records, False), batch_size=BATCH * 2, num_workers=2)
    out = []
    with torch.no_grad():
        for images, _, _ in loader:
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                logits = model(images.to(device))
            out.append(top_mean(torch.sigmoid(logits.float())).cpu().numpy())
    return np.concatenate(out)


def report(rows):
    scores = np.array([r["score"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    return measure(scores, labels, rows[0]["cutoff"])


def phone_photos(root):
    """Cracked and intact phones in everyday photos. Never used for training or cut-offs."""
    good, bad = [], []
    for path in sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")):
        name = path.parent.name.lower()
        (bad if "broken" in name or "crack" in name else good).append(path)
    rng = random.Random(SEED)
    return rng.sample(good, min(PHONE_PER_CLASS, len(good))), rng.sample(bad, min(PHONE_PER_CLASS, len(bad)))


def prepare_phones(root):
    good, bad = phone_photos(root)
    if not good or not bad:
        raise ValueError(f"expected good and broken phone folders under {root}")
    folder = CACHE / "phones"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (path, label) in enumerate([(p, 0) for p in good] + [(p, 1) for p in bad]):
        target = folder / f"{index:04d}.jpg"
        if not target.exists():
            Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR).save(target, quality=95)
        records.append({"path": str(target), "label": label, "product": "phones", "dataset": "phones"})
    return records


def run_split(tag, fit, cal, test, phones, device, seed):
    target = OUT / f"{tag}.json"
    if target.exists():
        print("already done", tag, flush=True)
        return json.loads(target.read_text())
    print(tag, "fit", len(fit), "cal", len(cal), "test", len(test), flush=True)
    model = train(fit, device, seed)
    cutoff = best_cutoff(score(model, cal, device), [r["label"] for r in cal])
    rows = [{**{k: r[k] for k in ("label", "product", "dataset")}, "score": float(v), "cutoff": cutoff}
            for r, v in zip(test, score(model, test, device))]
    phone = []
    if phones:
        phone = [{"label": r["label"], "score": float(v), "cutoff": cutoff} for r, v in zip(phones, score(model, phones, device))]
    result = {"tag": tag, "rows": rows, "phones": phone}
    target.write_text(json.dumps(result))
    for dataset in ("visa", "mvtec"):
        part = [r for r in rows if r["dataset"] == dataset]
        if part:
            print(tag, dataset, report(part), flush=True)
    if phone:
        print(tag, "phones", report(phone), flush=True)
    if tag == "known":
        torch.save({"state": model.state_dict(), "cutoff": cutoff, "model": MODEL}, OUT / "known_model.pt")
    del model
    torch.cuda.empty_cache()
    return result


def split_new(records):
    names = sorted({r["product"] for r in records})
    splits = []
    for fold, left_out in enumerate(group_folds(names, FOLDS)):
        rest = sorted(set(names) - set(left_out))
        held = set(random.Random(f"cal-{fold}").sample(rest, max(1, round(len(rest) * CAL_SHARE))))
        splits.append((f"new_{fold}", [r for r in records if r["product"] in rest and r["product"] not in held],
                       [r for r in records if r["product"] in held], [r for r in records if r["product"] in left_out]))
    return splits


def split_known(records):
    rng = random.Random(SEED)
    train_half, test = [], []
    for name in sorted({r["product"] for r in records}):
        for label in (0, 1):
            part = [r for r in records if r["product"] == name and r["label"] == label]
            rng.shuffle(part)
            train_half += part[:len(part) // 2]
            test += part[len(part) // 2:]
    rng.shuffle(train_half)
    cut = int(len(train_half) * CAL_SHARE)
    return "known", train_half[cut:], train_half[:cut], test


def summarise(results):
    summary = {}
    for level, tags in (("new_products", [r for r in results if r["tag"].startswith("new")]),
                        ("known_products", [r for r in results if r["tag"] == "known"])):
        rows = [row for r in tags for row in r["rows"]]
        summary[level] = {"all": merged(rows)}
        for dataset in ("visa", "mvtec"):
            summary[level][dataset] = merged([r for r in rows if r["dataset"] == dataset])
        phones = [row for r in tags for row in r["phones"]]
        if phones:
            summary[level]["phones"] = merged(phones)
    return summary


def merged(rows):
    labels = np.array([r["label"] for r in rows])
    hits = np.array([r["score"] >= r["cutoff"] for r in rows])
    caught, passed = float(hits[labels == 1].mean()), float((~hits[labels == 0]).mean())
    return {"balanced": (caught + passed) / 2, "caught": caught, "false_alarm": 1 - passed, "images": len(rows)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("No GPU found. In Colab: Runtime, Change runtime type, T4 GPU, then run all cells again.")
    print("gpu", torch.cuda.get_device_name(0), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = visa_types(DATA / "visa") + mvtec_types(DATA / "mvtec")
    if len(kinds) != 27:
        raise ValueError(f"expected 27 product types, found {len(kinds)}")
    records = prepare(kinds)
    phones = prepare_phones(DATA / "phones")
    print("photos", len(records), "phones", len(phones), flush=True)
    results = [run_split(*split, phones, device, seed) for seed, split in enumerate(split_new(records))]
    results.append(run_split(*split_known(records), phones, device, SEED))
    summary = summarise(results)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
