"""Stage 1 of PLAN_VIT_85.md: frozen DINOv2 with trained heads. Uses a GPU when there is one."""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(os.environ.get("PROBE_OUT", "/kaggle/working"))
DATA = Path(os.environ.get("PROBE_DATA", "/kaggle/input"))
SIZE = 448
GRID = SIZE // 14
PATCHES = GRID * GRID
PER_TYPE = 100
TOP_SHARE = 0.01
NEIGHBOURS = 16
FOLDS = 9
CAL_SHARE = 0.2
PATCHES_PER_IMAGE = 100
EPOCHS = 10
SEED = 0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VARIANTS = ("cls", "patch", "patch_self", "blend")


def visa_types(root):
    found = []
    for normal in sorted(Path(root).rglob("Data/Images/Normal")):
        masks = normal.parents[1] / "Masks" / "Anomaly"
        found.append({"dataset": "visa", "name": normal.parents[2].name, "good": sorted(normal.glob("*.JPG")),
                      "bad": [(p, masks / f"{p.stem}.png") for p in sorted((normal.parent / "Anomaly").glob("*.JPG"))]})
    return found


def mvtec_types(root):
    found = []
    for train in sorted(Path(root).rglob("train/good")):
        base = train.parents[1]
        bad = []
        for folder in sorted((base / "test").iterdir()):
            if folder.is_dir() and folder.name != "good":
                bad += [(p, base / "ground_truth" / folder.name / f"{p.stem}_mask.png") for p in sorted(folder.glob("*.png"))]
        found.append({"dataset": "mvtec", "name": f"mvtec_{base.name}", "bad": bad,
                      "good": sorted(train.glob("*.png")) + sorted((base / "test" / "good").glob("*.png"))})
    return found


def pick(kind):
    rng = random.Random(f"{kind['name']}-{SEED}")
    bad = rng.sample(kind["bad"], min(PER_TYPE, len(kind["bad"])))
    return rng.sample(kind["good"], min(PER_TYPE, len(kind["good"]))), bad


def to_array(path):
    image = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    return ((np.asarray(image, dtype=np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


def mask_grid(path):
    if path is None or not Path(path).exists():
        return np.zeros((GRID, GRID), dtype=bool)
    mask = Image.open(path).convert("L").point(lambda value: 255 if value else 0)
    return np.asarray(mask.resize((GRID, GRID), Image.BILINEAR)) > 40


def self_contrast(tokens, neighbours=NEIGHBOURS):
    """How unlike a patch is to its most similar patches in the same photo. Tokens must be unit length."""
    import torch
    similar = tokens @ tokens.transpose(1, 2)
    similar.diagonal(dim1=1, dim2=2).fill_(-2)
    return 1 - similar.topk(neighbours, dim=-1).values.mean(-1)


def group_folds(names, folds=FOLDS, seed=SEED):
    """Every type lands in exactly one fold, and each dataset spreads evenly across folds."""
    groups = [[] for _ in range(folds)]
    ordered = random.Random(seed).sample(sorted(names), len(names))
    ordered.sort(key=lambda name: name.startswith("mvtec_"))
    for index, name in enumerate(ordered):
        groups[index % folds].append(name)
    return groups


def best_cutoff(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    best, best_score = float(scores.max()), -1.0
    for value in np.unique(scores):
        score = (scores[labels == 1] >= value).mean() / 2 + (scores[labels == 0] < value).mean() / 2
        if score > best_score:
            best, best_score = float(value), score
    return best


def measure(scores, labels, cutoff):
    scores, labels = np.asarray(scores), np.asarray(labels)
    caught = float((scores[labels == 1] >= cutoff).mean())
    passed = float((scores[labels == 0] < cutoff).mean())
    return {"balanced": (caught + passed) / 2, "caught": caught, "false_alarm": 1 - passed}


def top_mean(values, share=TOP_SHARE):
    import torch
    count = max(1, int(values.shape[1] * share))
    return values.topk(count, dim=1).values.mean(1)


class Store:
    """Patch tokens for every photo, kept small in memory."""

    def __init__(self, tokens, cls, contrast, grids, labels, names, datasets):
        import torch
        self.tokens, self.cls, self.contrast = tokens, cls, contrast
        self.grids, self.labels, self.names, self.datasets = grids, labels, names, datasets
        self.means = torch.cat([part.float().mean(1) for part in tokens.split(64)])

    def features(self, images, patches, with_self):
        import torch
        tokens = self.tokens[images, patches].float()
        if not with_self:
            return tokens
        return torch.cat([tokens, tokens - self.means[images], self.contrast[images, patches].float().unsqueeze(-1)], -1)


def draw_patches(store, images, rng):
    """All-defect patches first, then ordinary ones, up to the per-photo limit."""
    chosen_images, chosen_patches, targets = [], [], []
    for image in images:
        hot = np.flatnonzero(store.grids[image])[:PATCHES_PER_IMAGE // 2]
        cold = rng.choice(np.flatnonzero(~store.grids[image]), PATCHES_PER_IMAGE - len(hot), replace=False)
        for group, target in ((hot, 1.0), (cold, 0.0)):
            chosen_images.append(np.full(len(group), image))
            chosen_patches.append(group)
            targets.append(np.full(len(group), target, dtype=np.float32))
    return np.concatenate(chosen_images), np.concatenate(chosen_patches), np.concatenate(targets)


def train_head(store, images, with_self, device, seed):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    image_ids, patch_ids, targets = draw_patches(store, images, rng)
    size = store.features(image_ids[:1], patch_ids[:1], with_self).shape[-1]
    head = torch.nn.Sequential(torch.nn.Linear(size, 256), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, 1)).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    weight = torch.tensor(float((targets == 0).sum() / max((targets == 1).sum(), 1)), device=device)
    for _ in range(EPOCHS):
        for start in range(0, len(targets), 4096):
            order = rng.integers(0, len(targets), min(4096, len(targets) - start))
            batch = store.features(image_ids[order], patch_ids[order], with_self).to(device)
            truth = torch.from_numpy(targets[order]).to(device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(batch).squeeze(-1), truth, pos_weight=weight)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return head.eval()


def score_images(store, head, images, with_self, device):
    import torch
    scores = []
    every = np.arange(PATCHES)
    with torch.no_grad():
        for start in range(0, len(images), 32):
            part = images[start:start + 32]
            image_ids = np.repeat(part, PATCHES)
            patch_ids = np.tile(every, len(part))
            batch = store.features(image_ids, patch_ids, with_self).to(device)
            probs = torch.sigmoid(head(batch).squeeze(-1)).reshape(len(part), PATCHES)
            scores.append(top_mean(probs).cpu().numpy())
    return np.concatenate(scores)


def run_variants(store, fit, others, device, seed):
    """Fit on `fit`, then return a score per variant for each image in `others`."""
    import torch
    from sklearn.linear_model import LogisticRegression
    linear = LogisticRegression(C=0.1, max_iter=1000).fit(store.cls[fit].numpy(), store.labels[fit])
    results = {"cls": linear.predict_proba(store.cls[others].numpy())[:, 1]}
    results["patch"] = score_images(store, train_head(store, fit, False, device, seed), others, False, device)
    results["patch_self"] = score_images(store, train_head(store, fit, True, device, seed), others, True, device)
    results["blend"] = (results["cls"] + results["patch_self"]) / 2
    return results


def summarise(rows, level):
    summary = {}
    for variant in VARIANTS:
        summary[variant] = {}
        for dataset in ("visa", "mvtec", "all"):
            part = [r for r in rows if dataset == "all" or r["dataset"] == dataset]
            hits = [r[f"{variant}_hit"] for r in part]
            labels = np.array([r["label"] for r in part])
            hits = np.array(hits, dtype=bool)
            caught, passed = hits[labels == 1].mean(), (~hits[labels == 0]).mean()
            summary[variant][dataset] = {"balanced": float((caught + passed) / 2), "caught": float(caught), "false_alarm": float(1 - passed)}
    return {level: summary}


def extract(kinds, device):
    import torch
    from transformers import AutoModel
    name = "facebook/dinov2-base" if device == "cuda" else "facebook/dinov2-small"
    model = AutoModel.from_pretrained(name).eval().to(device)
    print("model", name, device, flush=True)
    tokens, cls, grids, labels, names, datasets, contrast = [], [], [], [], [], [], []
    started = time.time()
    for kind in kinds:
        good, bad = pick(kind)
        items = [(path, None, 0) for path in good] + [(path, mask, 1) for path, mask in bad]
        for start in range(0, len(items), 16):
            part = items[start:start + 16]
            batch = torch.from_numpy(np.stack([to_array(path) for path, _, _ in part])).to(device)
            with torch.no_grad():
                out = model(pixel_values=batch).last_hidden_state
            unit = torch.nn.functional.normalize(out[:, 1:], dim=-1)
            tokens.append(unit.half().cpu())
            cls.append(torch.nn.functional.normalize(out[:, 0], dim=-1).cpu())
            contrast.append(self_contrast(unit).half().cpu())
        grids += [mask_grid(mask).ravel() for _, mask, _ in items]
        labels += [label for _, _, label in items]
        names += [kind["name"]] * len(items)
        datasets += [kind["dataset"]] * len(items)
        print(kind["name"], len(items), round(time.time() - started), flush=True)
    return Store(torch.cat(tokens), torch.cat(cls), torch.cat(contrast), np.stack(grids), np.array(labels), np.array(names), np.array(datasets))


def new_products(store, device):
    rows = []
    for fold, left_out in enumerate(group_folds(sorted(set(store.names)))):
        rest = sorted(set(store.names) - set(left_out))
        held = random.Random(f"cal-{fold}").sample(rest, max(1, round(len(rest) * CAL_SHARE)))
        fit = np.flatnonzero(~np.isin(store.names, left_out + held))
        cal = np.flatnonzero(np.isin(store.names, held))
        test = np.flatnonzero(np.isin(store.names, left_out))
        scores = run_variants(store, fit, np.concatenate([cal, test]), device, fold)
        record = [{"dataset": str(store.datasets[i]), "product": str(store.names[i]), "label": int(store.labels[i]), "fold": fold} for i in test]
        for variant, values in scores.items():
            cutoff = best_cutoff(values[:len(cal)], store.labels[cal])
            for row, value in zip(record, values[len(cal):]):
                row[variant] = float(value)
                row[f"{variant}_hit"] = bool(value >= cutoff)
        rows += record
        print("fold", fold, left_out, {v: round(measure(scores[v][len(cal):], store.labels[test], best_cutoff(scores[v][:len(cal)], store.labels[cal]))["balanced"], 3) for v in VARIANTS}, flush=True)
        (OUT / "vit_new_rows.json").write_text(json.dumps(rows))
    return rows


def known_products(store, device):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(store.labels))
    half = np.zeros(len(order), dtype=bool)
    for name in set(store.names):
        for label in (0, 1):
            members = order[(store.names[order] == name) & (store.labels[order] == label)]
            half[members[:len(members) // 2]] = True
    train = np.flatnonzero(half)
    test = np.flatnonzero(~half)
    shuffled = rng.permutation(train)
    cut = int(len(shuffled) * CAL_SHARE)
    cal, fit = shuffled[:cut], shuffled[cut:]
    scores = run_variants(store, fit, np.concatenate([cal, test]), device, SEED)
    rows = [{"dataset": str(store.datasets[i]), "product": str(store.names[i]), "label": int(store.labels[i])} for i in test]
    for variant, values in scores.items():
        cutoff = best_cutoff(values[:len(cal)], store.labels[cal])
        for row, value in zip(rows, values[len(cal):]):
            row[variant] = float(value)
            row[f"{variant}_hit"] = bool(value >= cutoff)
    (OUT / "vit_known_rows.json").write_text(json.dumps(rows))
    return rows


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kinds = visa_types(DATA) + mvtec_types(DATA)
    print("types", [(k["name"], len(k["good"]), len(k["bad"])) for k in kinds], flush=True)
    if len(kinds) != 27 or any(not k["good"] or not k["bad"] for k in kinds):
        raise ValueError("expected 12 VisA and 15 MVTec types, each with good and defective photos")
    store = extract(kinds, device)
    new = new_products(store, device)
    known = known_products(store, device)
    summary = {**summarise(new, "new_products"), **summarise(known, "known_products")}
    (OUT / "vit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
