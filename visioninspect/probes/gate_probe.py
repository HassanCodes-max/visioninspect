"""Check-step experiment from PLAN_CHECK_STEP.md. Runs on a CPU."""
from vit_probe import *  # Joined in by build_gate.py

GATE_MODEL = "facebook/dinov2-small"
GATE_SIZE = 224
NEAREST = 5
QUANTILE = 2
CASTING_SAMPLE = 300


def closeness(queries, gallery, nearest=NEAREST):
    """Mean cosine similarity to the closest gallery photos, for unit-length rows."""
    similar = queries @ gallery.T
    count = min(nearest, len(gallery))
    return np.sort(similar, axis=1)[:, -count:].mean(1)


def closest_product(queries, gallery, names):
    return names[np.argmax(queries @ gallery.T, axis=1)]


def split_half(names, labels, seed=SEED):
    """True marks the gallery half, split evenly within each product and label."""
    rng = np.random.default_rng(seed)
    half = np.zeros(len(names), dtype=bool)
    for name in sorted(set(names)):
        for label in (0, 1):
            members = rng.permutation(np.flatnonzero((names == name) & (labels == label)))
            half[members[:len(members) // 2]] = True
    return half


def evaluate(features, names, labels, outside, seed=SEED):
    rng = np.random.default_rng(seed)
    half = split_half(names, labels, seed)
    keep_aside = rng.random(len(names)) < CAL_SHARE
    gallery = half & ~keep_aside
    calibration = half & keep_aside
    test = ~half
    known, left_out, others = [], [], {key: [] for key in outside}
    for name in sorted(set(names)):
        store = gallery & (names != name)
        cutoff = np.percentile(closeness(features[calibration & (names != name)], features[store]), QUANTILE)
        known.append((closeness(features[test & (names != name)], features[store]) >= cutoff).mean())
        left_out.append((closeness(features[names == name], features[store]) < cutoff).mean())
        for key, values in outside.items():
            others[key].append((closeness(values, features[store]) < cutoff).mean())
    full = gallery | calibration
    cutoff = np.percentile(closeness(features[calibration], features[gallery]), QUANTILE)
    accepted = test & (closeness(features, features[gallery]) >= cutoff)
    right = closest_product(features[accepted], features[full], names[full]) == names[accepted]
    result = {"accepted": float(np.mean(known)), "left_out_review": float(np.mean(left_out)),
              "right_product": float(right.mean()), "by_type": {n: float(v) for n, v in zip(sorted(set(names)), left_out)}}
    result.update({f"{key}_review": float(np.mean(values)) for key, values in others.items()})
    outside_all = np.mean([result[f"{key}_review"] for key in outside])
    result["passed"] = bool(result["accepted"] >= 0.95 and result["left_out_review"] >= 0.90
                            and outside_all >= 0.95 and result["right_product"] >= 0.98)
    return result


def final_gallery(features, names, labels, seed=SEED):
    """The same gallery and cut-off that were tested, for the app to load."""
    rng = np.random.default_rng(seed)
    half = split_half(names, labels, seed)
    keep_aside = rng.random(len(names)) < CAL_SHARE
    gallery = half & ~keep_aside
    calibration = half & keep_aside
    return gallery, float(np.percentile(closeness(features[calibration], features[gallery]), QUANTILE))


def embed(paths, model):
    import torch
    out = []
    for start in range(0, len(paths), 32):
        arrays = []
        for path in paths[start:start + 32]:
            image = Image.open(path).convert("RGB").resize((GATE_SIZE, GATE_SIZE), Image.BILINEAR)
            arrays.append(((np.asarray(image, dtype=np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))
        with torch.no_grad():
            token = model(pixel_values=torch.from_numpy(np.stack(arrays))).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(token, dim=-1).numpy())
    return np.concatenate(out)


def outside_sets(root):
    phones = [p for p in Path(root).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg") and "phone" in str(p).lower()]
    phones = [p for p in phones if "casting" not in str(p).lower()]
    casting = sorted(p for p in Path(root).rglob("*.jpeg") if "casting_512x512" in str(p))
    return phones, random.Random(SEED).sample(casting, min(CASTING_SAMPLE, len(casting)))


def run_gate():
    from transformers import AutoModel
    model = AutoModel.from_pretrained(GATE_MODEL).eval()
    kinds = visa_types(DATA) + mvtec_types(DATA)
    if len(kinds) != 27:
        raise ValueError(f"expected 27 product types, found {len(kinds)}")
    paths, names, labels = [], [], []
    for kind in kinds:
        good, bad = pick(kind)
        for items, label in ((good, 0), ([p for p, _ in bad], 1)):
            paths += items
            names += [kind["name"]] * len(items)
            labels += [label] * len(items)
    phones, casting = outside_sets(DATA)
    print("photos", len(paths), "phones", len(phones), "casting", len(casting), flush=True)
    if not phones or not casting:
        raise ValueError("outside photo sets are missing")
    started = time.time()
    features = embed(paths, model)
    outside = {"phones": embed(phones, model), "casting": embed(casting, model)}
    print("seconds", round(time.time() - started), flush=True)
    names, labels = np.array(names), np.array(labels)
    result = evaluate(features, names, labels, outside)
    (OUT / "gate_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    gallery, cutoff = final_gallery(features, names, labels)
    np.savez_compressed(OUT / "gallery.npz", features=features[gallery].astype(np.float16), names=names[gallery], cutoff=cutoff)
    print("gallery", int(gallery.sum()), cutoff, flush=True)


if __name__ == "__main__":
    run_gate()
