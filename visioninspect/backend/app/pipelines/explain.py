"""An open vision language model that names the defect, says where it is and suggests a fix. Needs a GPU."""
import json
import re

from . import batch

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
LONGEST_SIDE = 1024
MAX_TOKENS = 300
TEXT_LIMIT = 240
BOX_SCALE = 1000
VERDICTS = ("good", "defect", "unsure")
SEVERITIES = ("minor", "major", "critical")
PROMPT = """You are a strict visual quality inspector. Look carefully at this one photo of a product or material.
Decide if the item has a defect or damage: cracks, tears, holes, scratches, dents, stains, burns, chips, broken or missing parts, wrong parts, or foreign material.
Normal texture, natural variation, shine, shadows and the background are not defects.
Reply with JSON only, with these keys:
"item": short name of the item
"verdict": "good", "defect" or "unsure"
"defect": short name of the defect, or ""
"where": where it is, in plain words, or ""
"box": [x1, y1, x2, y2] around the defect on a 0 to 1000 scale, or []
"severity": "minor", "major" or "critical", or ""
"fix": one or two sentences on how to repair it or prevent it, or ""
"""
HINT = """A trained inspection model has flagged a likely defect inside the box {box} on the same 0 to 1000 scale. Look closely there.
If you see damage, give its name, severity and a practical fix, such as replacing, re-gluing, sealing, cleaning or re-polishing the part.
"""
FLAGGED = """A trained inspection model is confident this item is defective, and the damage is inside the box {box} on the same 0 to 1000 scale.
Use "defect" as the verdict. Look closely at that area, then give the defect's name, severity and a practical fix, such as replacing, re-gluing, sealing, cleaning or re-polishing the part.
"""


def to_box(region):
    """A marked region as a 0 to 1000 box, the scale the model uses."""
    return [round(BOX_SCALE * v) for v in (region["x"], region["y"], region["x"] + region["width"], region["y"] + region["height"])]


def to_region(box):
    """A 0 to 1000 box as a marked area, or None if the model gave something unusable."""
    if not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, (int, float)) for v in box):
        return None
    x1, y1, x2, y2 = (min(max(v / BOX_SCALE, 0.0), 1.0) for v in box)
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "area_percent": round(100 * (x2 - x1) * (y2 - y1), 2)}


def clean(value):
    return value.strip()[:TEXT_LIMIT] if isinstance(value, str) else ""


def parse_reply(text):
    """Model replies are not always valid JSON, so anything unreadable becomes "unsure"."""
    found = re.search(r"\{.*\}", text, re.S)
    try:
        reply = json.loads(found.group(0)) if found else {}
    except json.JSONDecodeError:
        reply = {}
    verdict = reply.get("verdict") if reply.get("verdict") in VERDICTS else "unsure"
    defective = verdict == "defect"
    return {"item": clean(reply.get("item")), "verdict": verdict, "defect": clean(reply.get("defect")) if defective else "",
            "where": clean(reply.get("where")) if defective else "", "region": to_region(reply.get("box")) if defective else None,
            "severity": reply.get("severity") if defective and reply.get("severity") in SEVERITIES else "",
            "fix": clean(reply.get("fix")) if defective else ""}


class Explainer:
    def __init__(self, processor, model):
        self.processor, self.model = processor, model

    @classmethod
    def load(cls, device):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
        return cls(AutoProcessor.from_pretrained(MODEL), model)

    def explain(self, image, region=None, flagged=False):
        """region: where the trained model found the defect. flagged: the trained model is confident it is defective."""
        import torch
        small = image.copy()
        small.thumbnail((LONGEST_SIDE, LONGEST_SIDE))
        prompt = PROMPT + ((FLAGGED if flagged else HINT).format(box=to_box(region)) if region else "")
        messages = [{"role": "user", "content": [{"type": "image", "image": small}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                    return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
        return parse_reply(self.processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))

    def advise(self, facts):
        """Likely causes and next steps for a batch, as raw model text."""
        import torch
        messages = [{"role": "user", "content": [{"type": "text", "text": batch.PROMPT.format(facts=facts)}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                    return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
        return self.processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
