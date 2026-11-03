"""An open vision language model that names the defect, says where it is and suggests a fix.
The 8B model runs on the GPU. A 2B model from the same family is the backup on the CPU."""
import json
import logging
import os
import re
import threading

from . import batch

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CPU_MODEL = os.environ.get("VI_CPU_LLM", "Qwen/Qwen3-VL-2B-Instruct")
LONGEST_SIDE = 1024
CPU_SIDE = int(os.environ.get("VI_CPU_SIDE", "640"))
log = logging.getLogger("visioninspect")
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
SCORE_PROMPT = """You are a strict visual quality inspector. Look carefully at this one photo of a product or material.
Does the item have a defect or damage: cracks, tears, holes, scratches, dents, stains, burns, chips, broken or missing parts, wrong parts, or foreign material?
Normal texture, natural variation, shine, shadows and the background are not defects.
Answer with one word: defect or good."""
HINT = """A trained inspection model has flagged a likely defect inside the box {box} on the same 0 to 1000 scale. Look closely there.
If you see damage, give its name, severity and a practical fix, such as replacing, re-gluing, sealing, cleaning or re-polishing the part.
"""
LIKELY = """A quick inspection check found that this item is likely defective. Look closely for the damage.
If you see it, give its name, where it is, a box around it, its severity and a practical fix.
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


def place(value):
    """Where the fault is, in words. Smaller models sometimes repeat the box numbers instead, which mean nothing to a reader."""
    text = clean(value)
    return "" if re.search(r"\d{2,}", text) else text


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
            "where": place(reply.get("where")) if defective else "", "region": to_region(reply.get("box")) if defective else None,
            "severity": reply.get("severity") if defective and reply.get("severity") in SEVERITIES else "",
            "fix": clean(reply.get("fix")) if defective else ""}


class Explainer:
    def __init__(self, processor, model, side=LONGEST_SIDE):
        self.processor, self.model, self.side = processor, model, side

    @classmethod
    def load(cls, device, name=MODEL, side=LONGEST_SIDE):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        dtype = getattr(torch, os.environ.get("VI_CPU_DTYPE", "bfloat16")) if device == "cpu" else torch.bfloat16
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype).to(device).eval()
        return cls(AutoProcessor.from_pretrained(name), model, side)

    def explain(self, image, region=None, flagged=False):
        """region: where the trained model found the defect. flagged: the trained model is confident it is defective."""
        import torch
        small = image.copy()
        small.thumbnail((self.side, self.side))
        extra = (FLAGGED if flagged else HINT).format(box=to_box(region)) if region else LIKELY if flagged else ""
        prompt = PROMPT + extra
        messages = [{"role": "user", "content": [{"type": "image", "image": small}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                    return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
        return parse_reply(self.processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))

    def score(self, image):
        """How likely the item is defective, from 0 to 1, read from the model's first answer word.
        One pass with no text written, so it is much faster than explain() on a CPU."""
        import torch
        small = image.copy()
        small.thumbnail((self.side, self.side))
        messages = [{"role": "user", "content": [{"type": "image", "image": small}, {"type": "text", "text": SCORE_PROMPT}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                    return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()

        def word(*forms):
            return torch.logsumexp(logits[[self.processor.tokenizer.encode(f, add_special_tokens=False)[0] for f in forms]], 0)

        return float(torch.sigmoid(word("defect", "Defect") - word("good", "Good")))

    def advise(self, facts):
        """Likely causes and next steps for a batch, as raw model text."""
        import torch
        messages = [{"role": "user", "content": [{"type": "text", "text": batch.PROMPT.format(facts=facts)}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                    return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
        return self.processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


class OnDemand:
    """Loads a model the first time it is needed, so the backup costs nothing until it is used.
    If loading fails, it stays off and the check carries on without it."""

    def __init__(self, load):
        self.load, self.model, self.failed = load, None, False
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            if self.model is None and not self.failed:
                try:
                    self.model = self.load()
                except Exception:
                    log.exception("backup model could not load")
                    self.failed = True
        return self.model


class BackupExplainer:
    """The CPU explainer. Writing an explanation takes it several seconds, so checks use its quick score to decide,
    and the page asks for each explanation afterwards, one photo at a time."""
    backup = True

    def __init__(self, source):
        self.source = source

    def score(self, image):
        explainer = self.source.get()
        return explainer.score(image) if explainer else None

    def explain(self, image, region=None, flagged=False):
        explainer = self.source.get()
        return explainer.explain(image, region, flagged) if explainer else None

    def advise(self, facts):
        explainer = self.source.get()
        if explainer is None:
            raise RuntimeError("backup explainer is off")
        return explainer.advise(facts)


class BackupSegmenter:
    def __init__(self, source):
        self.source = source

    def outline(self, image, region):
        segmenter = self.source.get()
        return segmenter.outline(image, region) if segmenter else None
