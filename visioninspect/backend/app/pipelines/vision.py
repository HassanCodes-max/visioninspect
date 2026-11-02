"""The two vision models: one recognises the product, one looks for defects."""
import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn

SIZE = 448
RECOGNISE_SIZE = 224
TOP_SHARE = 0.01
NEIGHBOURS = 16
NEAREST = 5
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ATTENTION_NAMES = {".attention.q_proj.": ".attention.attention.query.", ".attention.k_proj.": ".attention.attention.key.",
                   ".attention.v_proj.": ".attention.attention.value.", ".attention.o_proj.": ".attention.output.dense."}


def fit_names(state, wanted):
    """The training machine's transformers named the attention layers differently, so match this machine's names."""
    def name(key):
        if key in wanted:
            return key
        for old, new in ATTENTION_NAMES.items():
            if old in key:
                return key.replace(old, new)
        return key
    return {name(key): value for key, value in state.items()}


def to_tensor(image, size):
    resized = image.convert("RGB").resize((size, size), Image.BILINEAR)
    pixels = (np.asarray(resized, dtype=np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)


def as_trained(image):
    """Training read each photo back from a small JPEG. Skipping that step moves scores by up to 0.2."""
    stream = io.BytesIO()
    image.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR).save(stream, "JPEG", quality=95)
    return Image.open(io.BytesIO(stream.getvalue()))


def self_contrast(tokens, neighbours=NEIGHBOURS):
    similar = tokens @ tokens.transpose(1, 2)
    similar.diagonal(dim1=1, dim2=2).fill_(-2)
    return 1 - similar.topk(neighbours, dim=-1).values.mean(-1)


def top_mean(values, share=TOP_SHARE):
    count = max(1, int(values.shape[1] * share))
    return values.topk(count, dim=1).values.mean(1)


class DefectNet(nn.Module):
    """Must keep the same layers as Inspector in probes/finetune.py, so saved weights load."""

    def __init__(self, backbone, size):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(2 * size + 1, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 1))

    def forward(self, images):
        tokens = self.backbone(pixel_values=images).last_hidden_state[:, 1:]
        unit = nn.functional.normalize(tokens, dim=-1)
        contrast = self_contrast(unit)
        features = torch.cat([unit, unit - unit.mean(1, keepdim=True), contrast.unsqueeze(-1)], -1)
        return self.head(features).squeeze(-1)


@dataclass(frozen=True)
class Finding:
    score: float
    heat: np.ndarray


class DefectFinder:
    def __init__(self, network):
        self.network = network.eval()

    @classmethod
    def load(cls, settings):
        from transformers import AutoConfig, AutoModel
        backbone = AutoModel.from_config(AutoConfig.from_pretrained(settings.backbone))
        network = DefectNet(backbone, backbone.config.hidden_size)
        saved = torch.load(settings.model_path, map_location="cpu", weights_only=True)
        network.load_state_dict(fit_names(saved["state"], network.state_dict()))
        return cls(network)

    @torch.inference_mode()
    def inspect(self, image):
        device = next(self.network.parameters()).device
        heat = torch.sigmoid(self.network(to_tensor(as_trained(image), SIZE).to(device)))
        return Finding(float(top_mean(heat)[0]), heat[0].cpu().numpy())


@dataclass(frozen=True)
class Recognition:
    product: str
    closeness: float
    supported: bool


class Recogniser:
    def __init__(self, encoder, gallery, names, cutoff, supported):
        self.encoder = encoder.eval()
        self.gallery = gallery
        self.names = names
        self.cutoff = cutoff
        self.supported = supported

    @classmethod
    def load(cls, settings):
        from transformers import AutoModel
        saved = np.load(settings.gallery_path, allow_pickle=False)
        return cls(AutoModel.from_pretrained(settings.recogniser), saved["features"].astype(np.float32), saved["names"],
                   float(saved["cutoff"]), settings.supported)

    def match(self, feature):
        similar = self.gallery @ feature
        closeness = float(np.sort(similar)[-NEAREST:].mean())
        product = str(self.names[int(similar.argmax())])
        return Recognition(product, closeness, closeness >= self.cutoff and product in self.supported)

    @torch.inference_mode()
    def check(self, image):
        device = next(self.encoder.parameters()).device
        token = self.encoder(pixel_values=to_tensor(image, RECOGNISE_SIZE).to(device)).last_hidden_state[0, 0]
        return self.match(nn.functional.normalize(token, dim=-1).cpu().numpy())
