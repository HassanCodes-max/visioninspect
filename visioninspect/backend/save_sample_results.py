"""Run the real models on the example photos and save the answers, so the example buttons work without using GPU quota."""
import base64
import io
import json
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.main import STATIC, Service
from app.pipelines import batch

KEEP = ("decision", "reason", "supported", "product", "model_version", "tested_share", "score", "regions", "affected_percent", "heat", "outline", "explanation")


def photo(item):
    return Image.open(io.BytesIO(base64.b64decode(item["src"].split(",", 1)[1]))).convert("RGB")


def main():
    service = Service(Settings())
    for path in sorted(Path(STATIC / "samples").glob("*.json")):
        sample = json.loads(path.read_text())
        for item in sample["items"]:
            answer = service.inspect(photo(item))
            item["result"] = {key: answer[key] for key in KEEP}
        sample["analysis"] = service.analyse([item["result"] for item in sample["items"]], batch.DEFAULT_LIMIT, None)
        path.write_text(json.dumps(sample))
        print(path.name, [item["result"]["decision"] for item in sample["items"]])


if __name__ == "__main__":
    main()
