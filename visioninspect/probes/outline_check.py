"""Does smoothing the trained model's map change how well it outlines a defect? Measured against the VisA defect masks.
Usage: python outline_check.py <folder of product folders with bad/ and mask/> <output json>"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.config import Settings  # noqa: E402
from app.pipelines import quality, segment, vision  # noqa: E402


def overlap(a, b):
    both = (a | b).sum()
    return float((a & b).sum() / both) if both else 0.0


def main():
    root, target = Path(sys.argv[1]), Path(sys.argv[2])
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
            blocky = np.asarray(Image.fromarray(grid.astype(np.uint8) * 255).resize(image.size, Image.NEAREST)) > 0
            outline = segment.from_heat(heat, settings.region_threshold)
            smooth = np.zeros(truth.shape, dtype=bool)
            if outline:
                smooth = np.asarray(Image.fromarray(segment.unpack_mask(outline["mask"])).resize(image.size, Image.NEAREST))
            rows.append({"product": folder.name, "photo": photo.name, "blocky": overlap(blocky, truth), "smooth": overlap(smooth, truth),
                         "true_area": round(100 * float(truth.mean()), 2), "site_area": outline["area_percent"] if outline else 0.0})
            print(folder.name, photo.name, round(rows[-1]["blocky"], 3), round(rows[-1]["smooth"], 3), rows[-1]["true_area"], rows[-1]["site_area"], flush=True)
    target.write_text(json.dumps(rows, indent=1))
    marked = [r for r in rows if r["site_area"]]
    print("photos", len(rows), "marked", len(marked), "blocky", round(float(np.mean([r["blocky"] for r in rows])), 3),
          "smooth", round(float(np.mean([r["smooth"] for r in rows])), 3),
          "median true area", round(float(np.median([r["true_area"] for r in rows])), 2),
          "median site area when marked", round(float(np.median([r["site_area"] for r in marked])), 2),
          "seconds", round(time.time() - began))


if __name__ == "__main__":
    main()
