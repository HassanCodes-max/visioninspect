"""One-photo probe from PLAN_ONE_PHOTO.md. Runs on a CPU, on Kaggle or locally for a smoke test."""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

OUT = Path(os.environ.get("PROBE_OUT", "/kaggle/working"))
DATA = Path(os.environ.get("PROBE_DATA", "/kaggle/input"))
WORK = Path(os.environ.get("PROBE_WORK", "/tmp/one_photo"))
PER_CLASS = int(os.environ.get("PROBE_PER_CLASS", "30"))
SEED = 0
IMAGE_SIZE = 518
SIGMA = 4
DPAM_LAYER = 20
PARAMS = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
WEIGHTS = "checkpoints/9_12_4_multiscale_visa/epoch_15.pth"
COMMIT = "3911738c0867544f545a076ad78f3f11d9ecbfdf"


def find_products(root):
    found = {}
    for normal in sorted(Path(root).rglob("Data/Images/Normal")):
        found[normal.parents[2].name] = (sorted(normal.glob("*.JPG")), sorted((normal.parent / "Anomaly").glob("*.JPG")))
    return found


def sample(paths, count, name):
    return random.Random(f"{name}-{SEED}").sample(paths, min(count, len(paths)))


def setup():
    repo = WORK / "AnomalyCLIP"
    if not repo.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/zqhang/AnomalyCLIP", str(repo)], check=True)
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != COMMIT:
            raise ValueError(f"expected AnomalyCLIP {COMMIT}, got {head}")
        loader = repo / "AnomalyCLIP_lib" / "model_load.py"
        loader.write_text(loader.read_text().replace("/remote-home/iot_zhouqihang/root/.cache/clip", str(WORK / "clip")))
        for path in (loader, repo / "prompt_ensemble.py"):
            path.write_text(path.read_text().replace("from pkg_resources import packaging", "import packaging.version"))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ftfy", "regex", "thop", "packaging"], check=True)
    sys.path.insert(0, str(repo))
    return repo


def load_model(repo):
    import torch
    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device="cpu", design_details=PARAMS)
    model.eval()
    preprocess, _ = get_transform(SimpleNamespace(image_size=IMAGE_SIZE))
    learner = AnomalyCLIP_PromptLearner(model.to("cpu"), PARAMS)
    learner.load_state_dict(torch.load(repo / WEIGHTS, map_location="cpu", weights_only=False)["prompt_learner"])
    model.visual.DAPM_replace(DPAM_layer=DPAM_LAYER)
    with torch.no_grad():
        prompts, tokens, compound = learner(cls_id=None)
        text = model.encode_text_learn(prompts, tokens, compound).float()
    text = torch.stack(torch.chunk(text, dim=0, chunks=2), dim=1)
    return model, preprocess, text / text.norm(dim=-1, keepdim=True)


def score_image(model, preprocess, text, path):
    import torch
    import AnomalyCLIP_lib
    from scipy.ndimage import gaussian_filter

    batch = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        image, patches = model.encode_image(batch, [24], DPAM_layer=DPAM_LAYER)
        image = image / image.norm(dim=-1, keepdim=True)
        prob = ((image @ text.permute(0, 2, 1)) / 0.07).softmax(-1)[:, 0, 1]
        maps = []
        for patch in patches:
            patch = patch / patch.norm(dim=-1, keepdim=True)
            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch, text[0])
            result = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], IMAGE_SIZE)
            maps.append((result[..., 1] + 1 - result[..., 0]) / 2.0)
    smooth = gaussian_filter(torch.stack(maps).sum(0)[0].numpy(), sigma=SIGMA)
    return float(prob[0]), float(smooth.max())


def main():
    repo = setup()
    products = find_products(DATA)
    if not products:
        raise ValueError(f"no VisA products under {DATA}")
    print("products", {k: (len(v[0]), len(v[1])) for k, v in products.items()}, flush=True)
    model, preprocess, text = load_model(repo)
    started = time.time()
    rows = []
    for name, (good, bad) in products.items():
        chosen = [(p, 0) for p in sample(good, PER_CLASS, f"{name}-good")] + [(p, 1) for p in sample(bad, PER_CLASS, f"{name}-bad")]
        for path, label in chosen:
            prob, peak = score_image(model, preprocess, text, path)
            rows.append({"product": name, "file": path.name, "label": label, "prob": prob, "map_max": peak})
        (OUT / "one_photo_rows.json").write_text(json.dumps(rows))
        print(name, len(rows), round(time.time() - started), flush=True)
    print("seconds per image", round((time.time() - started) / len(rows), 2), flush=True)


if __name__ == "__main__":
    main()
