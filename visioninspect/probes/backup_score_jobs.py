"""Kaggle CPU jobs that run the site's own explain.score() on the 32 earlier photos (to set cut-offs)
and the 80 fresh photos (to check them), at two photo sizes. Usage: backup_score_jobs.py <scratchpad> <output folder>"""
import base64
import io
import json
import sys
from pathlib import Path

SCRATCH, OUT = Path(sys.argv[1]), Path(sys.argv[2])
BACKEND = Path(r"C:\Users\ga\OneDrive\Documents\claude project 2\visioninspect\backend\app")
SIDES = (448, 640)

RUNNER = r'''
import base64, io, json, os, subprocess, sys, time
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0", "accelerate"], check=True)
import torch
from PIL import Image
PACKAGE = Path("/kaggle/working/vi")
PACKAGE.mkdir(parents=True, exist_ok=True)
(PACKAGE / "__init__.py").write_text("")
for name, text in CODE.items():
    (PACKAGE / name).write_text(text)
sys.path.insert(0, "/kaggle/working")
from vi import explain
from vi.explain import Explainer
torch.set_num_threads(os.cpu_count())
began = time.time()
explainer = Explainer.load("cpu", explain.CPU_MODEL, explain.CPU_SIDE)
print("loaded", round(time.time() - began), "s, threads", torch.get_num_threads(), flush=True)
OUT = Path("/kaggle/working/scores.json")
saved = {"job": JOB, "rows": []}


def find(product, label, source):
    folder = "Anomaly" if label == "bad" else "Normal"
    hits = list(Path("/kaggle/input").rglob(f"{product}/Data/Images/{folder}/{source}"))
    return Image.open(hits[0]).convert("RGB")


for task in TASKS:
    if task["kind"] == "photo":
        image = find(task["product"], task["label"], task["source"])
    else:
        image = Image.open(io.BytesIO(base64.b64decode(task["data"]))).convert("RGB")
    row = {k: v for k, v in task.items() if k != "data"}
    for side in SIDES:
        explainer.side = side
        start = time.time()
        row[f"score_{side}"] = explainer.score(image)
        row[f"seconds_{side}"] = round(time.time() - start, 2)
    saved["rows"].append(row)
    OUT.write_text(json.dumps(saved))
    print(task["name"], task["label"], {s: round(row[f"score_{s}"], 3) for s in SIDES}, {s: row[f"seconds_{s}"] for s in SIDES}, flush=True)
print("done", round(time.time() - began), "s", flush=True)
'''


def kernel(slug, title, tasks):
    folder = OUT / slug
    folder.mkdir(parents=True, exist_ok=True)
    code = {name: (BACKEND / "pipelines" / name).read_text(encoding="utf-8") for name in ("explain.py", "batch.py")}
    header = f"JOB = {slug!r}\nCODE = {json.dumps(code)}\nTASKS = {json.dumps(tasks)}\nSIDES = {SIDES!r}\n"
    (folder / "run.py").write_text(header + RUNNER, encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"hassanagddsbdxfg/{slug}", "title": title, "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_internet": True, "dataset_sources": ["tensura3607/amazon-visa-anomaly"],
        "competition_sources": [], "kernel_sources": []}, indent=1), encoding="utf-8")
    print(slug, len(tasks), "tasks", round((folder / "run.py").stat().st_size / 1e6, 2), "MB")


blind = json.loads((SCRATCH / "blind_key.json").read_text())
fresh = json.loads((SCRATCH / "fresh_key.json").read_text())
def small(path):
    """The model never sees more than 640 px, so the photo is sent at that size to fit Kaggle's 1 MB limit."""
    from PIL import Image
    image = Image.open(path).convert("RGB")
    image.thumbnail((640, 640))
    data = io.BytesIO()
    image.save(data, "JPEG", quality=90)
    return base64.b64encode(data.getvalue()).decode()


tasks = [{"kind": "blind", "name": n, **v, "data": small(SCRATCH / "blind" / f"{n}.jpg")} for n, v in sorted(blind.items())]
tasks += [{"kind": "photo", "name": n, **fresh[n]} for n in sorted(fresh)]
for part in range(4):
    kernel(f"visioninspect-score-{part + 1}", f"VisionInspect Score {part + 1}", tasks[part::4])
