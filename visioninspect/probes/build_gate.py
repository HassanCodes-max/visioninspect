"""Join vit_probe.py and gate_probe.py into the one file Kaggle runs. Usage: build_gate.py <folder>"""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
MAIN = 'if __name__ == "__main__":'


def build():
    helpers = (HERE / "vit_probe.py").read_text(encoding="utf-8").split(MAIN)[0]
    code = helpers + (HERE / "gate_probe.py").read_text(encoding="utf-8").replace(
        "from vit_probe import *  # Joined in by build_gate.py\n", "")
    ast.parse(code)
    return code


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "gate_probe_kernel.py").write_text(build(), encoding="utf-8")
    meta = {"id": "hassanagddsbdxfg/visioninspect-check-step", "title": "VisionInspect Check Step", "code_file": "gate_probe_kernel.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
            "dataset_sources": ["tensura3607/amazon-visa-anomaly", "ipythonx/mvtec-ad",
                                "axondata/cracked-and-intact-smartphone-images-dataset",
                                "ravirajsinh45/real-life-industrial-dataset-of-casting-product"],
            "competition_sources": [], "kernel_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
