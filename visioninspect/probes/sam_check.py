"""Does SAM 2 outline a defect well when given a box? Measured against the VisA defect masks.
Usage: python sam_check.py <folder of product folders with bad/ and mask/> <output json>"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from transformers import Sam2Model, Sam2Processor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.config import Settings  # noqa: E402
from app.pipelines import quality, vision  # noqa: E402

MODEL = "facebook/sam2.1-hiera-small"
LOOSE = 0.25


def overlap(a, b):
    both = (a | b).sum()
    return float((a & b).sum() / both) if both else 0.0


def largest_box(mask):
    labelled, _ = ndimage.label(mask)
    slices = ndimage.find_objects(labelled)
    sizes = [(labelled[s] > 0).sum() for s in slices]
    rows, columns = slices[int(np.argmax(sizes))]
    return columns.start, rows.start, columns.stop, rows.stop


def loosen(box, size):
    x1, y1, x2, y2 = box
    width, height = size
    grow_x, grow_y = (x2 - x1) * LOOSE, (y2 - y1) * LOOSE
    return [max(0, x1 - grow_x), max(0, y1 - grow_y), min(width, x2 + grow_x), min(height, y2 + grow_y)]


@torch.inference_mode()
def outline(model, processor, image, box):
    inputs = processor(images=image, input_boxes=[[box]], return_tensors="pt")
    output = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(output.pred_masks.cpu(), inputs["original_sizes"])[0]
    return masks.squeeze().numpy() > 0


def main():
    root, target = Path(sys.argv[1]), Path(sys.argv[2])
    processor = Sam2Processor.from_pretrained(MODEL)
    model = Sam2Model.from_pretrained(MODEL).eval()
    settings = Settings()
    finder = vision.DefectFinder.load(settings)
    rows = []
    began = time.time()
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for photo in sorted((folder / "bad").glob("*.JPG")):
            found = list((folder / "mask").glob(f"{photo.stem}.*"))
            if not found:
                continue
            image = Image.open(photo).convert("RGB")
            truth = np.asarray(Image.open(found[0]).convert("L").resize(image.size, Image.NEAREST)) > 0
            if not truth.any():
                continue
            heat = finder.inspect(image).heat
            grid = np.asarray(heat).reshape(quality.GRID, quality.GRID) >= settings.region_threshold
            heat_mask = np.asarray(Image.fromarray(grid.astype(np.uint8) * 255).resize(image.size, Image.NEAREST)) > 0
            row = {"product": folder.name, "photo": photo.name, "heat": overlap(heat_mask, truth)}
            row["sam_good_box"] = overlap(outline(model, processor, image, loosen(largest_box(truth), image.size)), truth)
            regions = quality.find_regions(heat, settings.region_threshold)
            if regions:
                r = regions[0]
                w, h = image.size
                row["sam_heat_box"] = overlap(outline(model, processor, image, loosen((r["x"] * w, r["y"] * h, (r["x"] + r["width"]) * w, (r["y"] + r["height"]) * h), image.size)), truth)
            rows.append(row)
            print(folder.name, photo.name, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}, flush=True)
    target.write_text(json.dumps(rows, indent=1))
    summary = {key: round(float(np.mean([r[key] for r in rows if key in r])), 3) for key in ("heat", "sam_good_box", "sam_heat_box")}
    print("photos", len(rows), "mean overlap", summary, "seconds", round(time.time() - began))


if __name__ == "__main__":
    main()
