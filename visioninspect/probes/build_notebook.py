"""Build the one-click Colab notebook that holds the fine-tuning code."""
import json
from pathlib import Path

HERE = Path(__file__).parent
TARGET = HERE.parent / "VisionInspect_Train.ipynb"
DATASETS = (("visa", "tensura3607/amazon-visa-anomaly"), ("mvtec", "ipythonx/mvtec-ad"),
            ("phones", "axondata/cracked-and-intact-smartphone-images-dataset"))
INTRO = """# VisionInspect AI: train the Vision Transformer

Press **Runtime, then Run all**. When Google asks, allow access to your Drive, so progress is saved if Colab disconnects.

Keep this tab open. It takes about 2 hours on the free T4 GPU. If it stops, press Run all again and it continues from the last finished part.

When it ends, a file called `visioninspect_results.zip` downloads to your computer.

Your photos and passwords are not used. The data is public.
"""
CHECK = """import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Open Runtime, Change runtime type, choose T4 GPU, then Run all again.")
print("GPU ready:", torch.cuda.get_device_name(0))
"""
DRIVE = """import os
from google.colab import drive
drive.mount("/content/drive")
os.makedirs("/content/drive/MyDrive/VisionInspect", exist_ok=True)
os.environ["TUNE_OUT"] = "/content/drive/MyDrive/VisionInspect"
print("Progress saves to Google Drive, in the VisionInspect folder")
"""
FINISH = """import json, os, zipfile
from pathlib import Path
from google.colab import files
out = Path(os.environ["TUNE_OUT"])
print(json.dumps(json.loads((out / "summary.json").read_text()), indent=2))
with zipfile.ZipFile("/content/visioninspect_results.zip", "w") as package:
    for path in out.glob("*.json"):
        package.write(path, path.name)
files.download("/content/visioninspect_results.zip")
print("Done. The trained model stays in your Drive, in VisionInspect/known_model.pt")
"""


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def download_cell():
    lines = ["%%bash\n", "set -e\n", "mkdir -p /content/data\n", "cd /content/data\n"]
    for name, slug in DATASETS:
        lines += [f"if [ ! -d {name} ]; then\n", f"  echo Downloading {name}\n",
                  f"  curl -sSL --fail -o {name}.zip https://www.kaggle.com/api/v1/datasets/download/{slug}\n",
                  f"  unzip -q {name}.zip -d {name}\n", f"  rm {name}.zip\n", "fi\n"]
    lines.append("du -sh /content/data/*\n")
    return code("".join(lines))


def file_cell(name):
    return code(f"%%writefile {name}\n" + (HERE / name).read_text(encoding="utf-8"))


def build():
    cells = [markdown(INTRO), code(CHECK), code(DRIVE), download_cell(), file_cell("vit_probe.py"), file_cell("finetune.py"),
             code("!python -u finetune.py\n"), code(FINISH)]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    TARGET.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(TARGET, TARGET.stat().st_size)
