"""Outlines of the damaged area: SAM 2 from the AI's box, or the trained model's own map."""
import numpy as np
from PIL import Image

from .quality import GRID as MAP_GRID

MODEL = "facebook/sam2.1-hiera-small"
GRID = 256


def pack_mask(mask):
    """A full-size mask as GRID×GRID bits in hex, small enough to send to the browser."""
    small = Image.fromarray(mask.astype(np.uint8) * 255).resize((GRID, GRID), Image.NEAREST)
    return np.packbits(np.asarray(small) > 0).tobytes().hex()


def unpack_mask(text):
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(text), dtype=np.uint8))
    return bits[:GRID * GRID].reshape(GRID, GRID).astype(bool)


def to_outline(mask):
    return {"mask": pack_mask(mask), "area_percent": round(100 * float(mask.mean()), 2)} if mask.any() else None


def from_heat(heat, threshold):
    """The trained model's map as a smooth outline. No GPU needed."""
    grid = Image.fromarray(np.asarray(heat, dtype=np.float32).reshape(MAP_GRID, MAP_GRID))
    return to_outline(np.asarray(grid.resize((GRID, GRID), Image.BILINEAR)) >= threshold)


class Segmenter:
    def __init__(self, processor, model):
        self.processor, self.model = processor, model

    @classmethod
    def load(cls, device):
        from transformers import Sam2Model, Sam2Processor
        return cls(Sam2Processor.from_pretrained(MODEL), Sam2Model.from_pretrained(MODEL).to(device).eval())

    def outline(self, image, region):
        import torch
        width, height = image.size
        box = [region["x"] * width, region["y"] * height, (region["x"] + region["width"]) * width, (region["y"] + region["height"]) * height]
        inputs = self.processor(images=image, input_boxes=[[box]], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(output.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        return to_outline(masks.squeeze().numpy() > 0)
