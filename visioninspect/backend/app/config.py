"""Settings, read from the environment so nothing is hardcoded per machine."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

PRODUCT_NAMES = {
    "candle": "Candles", "capsules": "Capsules (blister)", "cashew": "Cashews", "chewinggum": "Chewing gum",
    "fryum": "Fryums", "macaroni1": "Macaroni (arranged)", "macaroni2": "Macaroni (loose)", "pcb1": "Circuit board 1",
    "pcb2": "Circuit board 2", "pcb4": "Circuit board 4", "pipe_fryum": "Pipe fryums", "mvtec_bottle": "Bottle",
    "mvtec_cable": "Cable", "mvtec_capsule": "Capsule", "mvtec_carpet": "Carpet", "mvtec_grid": "Wire grid",
    "mvtec_hazelnut": "Hazelnut", "mvtec_leather": "Leather", "mvtec_metal_nut": "Metal nut", "mvtec_pill": "Pill",
    "mvtec_tile": "Tile", "mvtec_toothbrush": "Toothbrush", "mvtec_transistor": "Transistor", "mvtec_wood": "Wood",
    "mvtec_zipper": "Zipper",
}


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.environ.get("VI_MODEL", MODELS / "known_model.pt"))
    gallery_path: Path = Path(os.environ.get("VI_GALLERY", MODELS / "gallery.npz"))
    bands_path: Path = Path(os.environ.get("VI_BANDS", MODELS / "bands.json"))
    backbone: str = os.environ.get("VI_BACKBONE", "facebook/dinov2-base")
    recogniser: str = os.environ.get("VI_RECOGNISER", "facebook/dinov2-small")
    max_upload_bytes: int = int(os.environ.get("VI_MAX_UPLOAD_MB", "15")) * 1024 * 1024
    max_pixels: int = 50_000_000
    max_batch: int = 30
    region_threshold: float = 0.5
    model_version: str = "vit-b-finetuned-2026-11-02"
    supported: dict = field(default_factory=lambda: dict(PRODUCT_NAMES))
