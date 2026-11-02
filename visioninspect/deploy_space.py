"""Publish visioninspect/backend as a Hugging Face ZeroGPU Space."""
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

BACKEND = Path(__file__).parent / "backend"
REPO = os.environ.get("VI_REPO", "HASSANALI0987654/visioninspect")
NEEDED = ("known_model.pt", "gallery.npz", "bands.json")
REQUIREMENTS = "torch==2.13.0\ntorchvision\ntransformers==5.17.0\naccelerate\nqwen-vl-utils\nscipy\nnumpy\npillow\nauthlib\nitsdangerous\n"
CARD = """---
title: VisionInspect
emoji: 🔍
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: space_app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: One photo gives PASS, REVIEW or REJECT and marks the defect
hf_oauth: true
---

VisionInspect checks one photo with a fine-tuned Vision Transformer and gives PASS, REVIEW or REJECT, with the defect outlined, its size and severity, and how to fix it. Items outside its 25 supported products get an AI opinion from Qwen3-VL-8B. Visitors can sign in with Hugging Face to put their name on reports.

Measured on 2,323 held-out studio photos of those products: 90% decided automatically, 95% of those decisions correct. A person's own product or phone photo may give different results.

Photos are sent to this server for the check, used once, and not stored. It does not replace certified inspection or safety-critical testing. Licences are in `app/static/NOTICE.txt`.
"""


def main():
    missing = [name for name in NEEDED if not (BACKEND / "models" / name).exists()]
    if missing:
        raise SystemExit(f"missing from backend/models: {missing}")
    api = HfApi()
    api.create_repo(REPO, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g", exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "space"
        shutil.copytree(BACKEND, target, ignore=shutil.ignore_patterns("__pycache__", "make_bands.py", "save_sample_results.py"))
        (target / "README.md").write_text(CARD, encoding="utf-8")
        (target / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
        api.upload_folder(repo_id=REPO, repo_type="space", folder_path=target, commit_message="Publish VisionInspect")
    print("files", api.list_repo_files(REPO, repo_type="space"))
    print("runtime", api.get_space_runtime(REPO).stage)


if __name__ == "__main__":
    main()
