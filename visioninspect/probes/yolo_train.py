"""Train one YOLO class, "defect", from VisA and MVTec masks. Six product types are held out to measure new products.
Licence note: Ultralytics is AGPL-3.0, so the code of any site using this model must stay public."""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

OUT = Path(os.environ.get("PROBE_OUT", "/kaggle/working"))
DATA = Path(os.environ.get("PROBE_DATA", "/kaggle/input"))
WORK = Path("/tmp/yolo")
SIZE = 640
PER_TYPE = 100
SEED = 0
MIN_SHARE = 0.0005
HELD_OUT = ("candle", "pcb2", "pipe_fryum", "mvtec_bottle", "mvtec_leather", "mvtec_metal_nut")
CONFIDENCE = (0.1, 0.25, 0.5)


def visa_types(root):
    found = []
    for normal in sorted(Path(root).rglob("Data/Images/Normal")):
        masks = normal.parents[1] / "Masks" / "Anomaly"
        found.append({"name": normal.parents[2].name, "good": sorted(normal.glob("*.JPG")),
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
        found.append({"name": f"mvtec_{base.name}", "bad": bad,
                      "good": sorted(train.glob("*.png")) + sorted((base / "test" / "good").glob("*.png"))})
    return found


def pick(kind):
    rng = random.Random(f"{kind['name']}-{SEED}")
    bad = rng.sample(kind["bad"], min(PER_TYPE, len(kind["bad"])))
    return rng.sample(kind["good"], min(PER_TYPE, len(kind["good"]))), bad


def boxes_from_mask(path):
    """Normalised (cx, cy, w, h) boxes, one per separate marked area. Specks are dropped."""
    mask = np.asarray(Image.open(path).convert("L")) > 0
    found, _ = ndimage.label(mask)
    height, width = mask.shape
    boxes = []
    for row, column in ndimage.find_objects(found):
        w, h = (column.stop - column.start) / width, (row.stop - row.start) / height
        if w * h >= MIN_SHARE:
            boxes.append(((column.start + column.stop) / 2 / width, (row.start + row.stop) / 2 / height, w, h))
    return boxes


def save(records, split):
    images, labels = WORK / "images" / split, WORK / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        stem = f"{record['name']}_{index:04d}"
        Image.open(record["path"]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR).save(images / f"{stem}.jpg", quality=95)
        lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in record["boxes"]]
        (labels / f"{stem}.txt").write_text("\n".join(lines))
        record["file"] = str(images / f"{stem}.jpg")


def build():
    kinds = visa_types(DATA) + mvtec_types(DATA)
    if len(kinds) != 27:
        raise ValueError(f"expected 27 product types, found {len(kinds)}")
    train, test = [], []
    for kind in kinds:
        good, bad = pick(kind)
        target = test if kind["name"] in HELD_OUT else train
        target += [{"name": kind["name"], "path": p, "boxes": [], "label": 0} for p in good]
        target += [{"name": kind["name"], "path": p, "boxes": boxes_from_mask(m), "label": 1} for p, m in bad]
    random.Random(SEED).shuffle(train)
    cut = len(train) // 10
    save(train[cut:], "train")
    save(train[:cut], "val")
    save(test, "test")
    (WORK / "data.yaml").write_text(f"path: {WORK}\ntrain: images/train\nval: images/val\nnames:\n  0: defect\n")
    return test


def overlap(a, b):
    """Intersection over union for corner boxes."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    both = w * h
    return both / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - both)


def corners(box):
    cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def measure(model, test):
    rows = []
    for record in test:
        result = model.predict(record["file"], imgsz=SIZE, conf=min(CONFIDENCE), verbose=False)[0]
        found = [{"box": [float(v) for v in box], "conf": float(conf)} for box, conf in zip(result.boxes.xyxyn, result.boxes.conf)]
        rows.append({"product": record["name"], "label": record["label"], "found": found, "truth": [list(corners(b)) for b in record["boxes"]]})
    (OUT / "yolo_rows.json").write_text(json.dumps(rows))
    summary = {}
    for limit in CONFIDENCE:
        hits = np.array([any(f["conf"] >= limit for f in r["found"]) for r in rows])
        labels = np.array([r["label"] for r in rows])
        located = [any(f["conf"] >= limit and overlap(f["box"], t) >= 0.1 for f in r["found"] for t in r["truth"]) for r in rows if r["label"] == 1]
        caught, passed = float(hits[labels == 1].mean()), float((~hits[labels == 0]).mean())
        summary[str(limit)] = {"balanced": (caught + passed) / 2, "caught": caught, "false_alarm": 1 - passed, "located": float(np.mean(located))}
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.is_available()
    print("gpu", gpu, flush=True)
    test = build()
    model = YOLO("yolo11n.pt")
    model.train(data=str(WORK / "data.yaml"), imgsz=SIZE, epochs=60 if gpu else 25, time=9, batch=16, patience=15, device=0 if gpu else "cpu",
                project=str(OUT), name="yolo", seed=SEED, plots=False, verbose=False)
    best = YOLO(str(OUT / "yolo" / "weights" / "best.pt"))
    summary = {"held_out": HELD_OUT, "images": len(test), "by_confidence": measure(best, test)}
    (OUT / "yolo_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
