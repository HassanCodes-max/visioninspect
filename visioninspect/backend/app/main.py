"""VisionInspect web service. Photos are read from memory, then discarded: nothing is saved to disk or logged."""
import hashlib
import io
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .pipelines import batch, quality, segment

STATIC = Path(__file__).parent / "static"
VERSIONED = ("styles.css", "fonts.css", "text.js", "app.js", "motion.js", "fluid.js")


def versioned_page():
    """The page with each style and script address tagged by its content, so browsers never keep an old copy after an update."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for name in VERSIONED:
        path = STATIC / name
        if path.exists():
            tag = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
            html = html.replace(f"/static/{name}\"", f"/static/{name}?v={tag}\"")
    return html
log = logging.getLogger("visioninspect")


class Service:
    """Loads the models once, on first use, and runs one photo at a time to protect memory."""

    def __init__(self, settings, recogniser=None, finder=None, bands=None, explainer=None, segmenter=None):
        self.settings = settings
        self.recogniser, self.finder, self.bands, self.explainer, self.segmenter = recogniser, finder, bands, explainer, segmenter
        self.lock = threading.Lock()

    def ready(self):
        if self.finder is None:
            from .pipelines.vision import DefectFinder, Recogniser
            self.bands = quality.Bands(**json.loads(self.settings.bands_path.read_text())["bands"])
            self.recogniser = Recogniser.load(self.settings)
            self.finder = DefectFinder.load(self.settings)

    def warm(self):
        with self.lock:
            self.ready()

    def inspect(self, image, request=None):
        """With the backup AI, only the quick decision is made here. The page asks for the explanation afterwards."""
        later = getattr(self.explainer, "backup", False)
        with self.lock:
            self.ready()
            match = self.recogniser.check(image)
            if not match.supported:
                if later:
                    decision, reason = quality.decide_by_score(self.explainer.score(image))
                    return self.result(decision, reason, match, None, pending=decision != quality.PASS and reason != "not_supported")
                explanation = self.explainer.explain(image) if self.explainer else None
                decision, reason = quality.decide_by_ai(explanation)
                return self.result(decision, reason, match, None, explanation, self.outline(image, explanation))
            found = self.finder.inspect(image)
            decision, reason = quality.decide(found.score, True, self.bands)
            marked = quality.find_regions(found.heat, self.settings.region_threshold)
            explanation = outline = None
            if self.explainer and decision != quality.PASS and not later:
                explanation = self.explainer.explain(image, marked[0] if marked else None, decision == quality.REJECT)
                outline = None if marked else self.outline(image, explanation)
        return self.result(decision, reason, match, found, explanation, outline, pending=later and decision != quality.PASS)

    def explain_one(self, image, region=None, flagged=False):
        """The explanation for one photo, asked for after its decision is already on screen."""
        with self.lock:
            explanation = self.explainer.explain(image, region, flagged) if self.explainer else None
            outline = None if region else self.outline(image, explanation)
        return {"explanation": explanation, "outline": outline}

    def outline(self, image, explanation):
        region = explanation["region"] if explanation else None
        return self.segmenter.outline(image, region) if self.segmenter and region else None

    def inspect_many(self, images, request=None, limit=batch.DEFAULT_LIMIT, stock=None):
        results = [self.inspect(image, request) for image in images]
        return {"results": results, "analysis": self.analyse(results, limit, stock)}

    def analyse(self, results, limit, stock):
        stats = batch.analyse(results, limit, stock)
        if self.explainer and not getattr(self.explainer, "backup", False) and stats["rejected"]:
            try:
                stats.update(batch.parse_advice(self.explainer.advise(batch.facts_for_ai(stats))))
            except Exception:
                log.exception("batch advice failed")
        return stats

    def result(self, decision, reason, match, found, explanation=None, outline=None, pending=False):
        names = self.settings.supported
        answer = {"decision": decision, "reason": reason, "supported": match.supported,
                  "product": names.get(match.product) if match.supported else None,
                  "model_version": self.settings.model_version, "tested_share": self.bands.evidence(decision) if match.supported else quality.AI_EVIDENCE.get(decision),
                  "score": None, "regions": [], "affected_percent": 0.0, "heat": None, "explanation": explanation, "outline": None, "pending": pending}
        if found is not None:
            regions = quality.find_regions(found.heat, self.settings.region_threshold)
            answer.update(score=round(found.score, 4), regions=regions,
                          affected_percent=round(sum(r["area_percent"] for r in regions), 2),
                          heat=[round(float(v), 3) for v in found.heat])
            mark = segment.from_heat(found.heat, self.settings.region_threshold) if regions and decision != quality.PASS else None
            if mark:
                answer.update(outline=mark["mask"], affected_percent=mark["area_percent"])
        if not answer["regions"] and explanation and explanation["region"]:
            answer.update(regions=[explanation["region"]], affected_percent=explanation["region"]["area_percent"])
            if outline:
                answer.update(outline=outline["mask"], affected_percent=outline["area_percent"])
        return answer


def on_cpu(service, images, limit=batch.DEFAULT_LIMIT, stock=None):
    """The check when the shared GPU refuses it. The trained model still decides listed products on the CPU,
    and a smaller backup AI scores other items. If the backup AI cannot load, those items wait in REVIEW."""
    answer = Service.inspect_many(service, images, None, limit, stock)
    for result in answer["results"]:
        if result["reason"] == "not_supported":
            result["reason"] = "ai_resting"
        if result["reason"].startswith("ai_"):
            result["tested_share"] = quality.BACKUP_EVIDENCE.get(result["decision"])
    answer["fallback"] = True
    return answer


def read_photo(data, settings):
    Image.MAX_IMAGE_PIXELS = settings.max_pixels
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(415, "This file could not be opened as a photo.") from error
    return ImageOps.exif_transpose(image).convert("RGB")


def read_region(text):
    """A marked area sent back by the page, checked so it can only be four numbers between 0 and 1."""
    if not text:
        return None
    try:
        region = {key: float(json.loads(text)[key]) for key in ("x", "y", "width", "height")}
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(422, "The marked area could not be read.") from error
    if not all(0.0 <= value <= 1.0 for value in region.values()):
        raise HTTPException(422, "The marked area could not be read.")
    return region


def tidy_result(item):
    """Only the fields the statistics use, with the AI's words cut short."""
    found = item.get("explanation") or {}
    decision = item["decision"] if item.get("decision") in (quality.PASS, quality.REVIEW, quality.REJECT) else quality.REVIEW
    explanation = {"defect": str(found.get("defect") or "")[:80], "severity": str(found.get("severity") or "")[:20]} if found else None
    return {"decision": decision, "affected_percent": min(max(float(item.get("affected_percent") or 0.0), 0.0), 100.0), "explanation": explanation}


def create_app(service=None, settings=None):
    settings = settings or Settings()
    service = service or Service(settings)

    @asynccontextmanager
    async def warm_up(_):
        threading.Thread(target=service.warm, daemon=True).start()
        yield

    app = FastAPI(title="VisionInspect", version="0.1.0", lifespan=warm_up)

    @app.middleware("http")
    async def limit_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_upload_bytes + 4096:
            return JSONResponse({"detail": "This photo is too large. The limit is 15 MB."}, status_code=413)
        return await call_next(request)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model_version": settings.model_version}

    @app.get("/api/me")
    def me(request: Request):
        """Who is signed in with Hugging Face. The sign-in session only exists on the Space, where Gradio adds it."""
        info = (request.scope.get("session") or {}).get("oauth_info") or {}
        user = info.get("userinfo") or {}
        if not user.get("preferred_username"):
            return {"signed_in": False}
        return {"signed_in": True, "name": user.get("name") or user["preferred_username"], "username": user["preferred_username"]}

    @app.post("/api/explain")
    def explain_later(photo: UploadFile, region: str = Form(""), flagged: bool = Form(False)):
        """The name, place and fix of a fault, for a photo whose decision is already shown. Used when checks run on the CPU."""
        data = photo.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "This photo is too large. The limit is 15 MB.")
        try:
            return service.explain_one(read_photo(data, settings), read_region(region), flagged)
        except HTTPException:
            raise
        except Exception as error:
            log.exception("explanation failed")
            raise HTTPException(500, "The explanation could not finish.") from error

    @app.post("/api/analyse")
    def analyse(body: dict):
        """Batch statistics again, once the late explanations have named the faults. Computed in code only."""
        try:
            results = [tidy_result(item) for item in body.get("results", [])[:settings.max_batch]]
            limit = min(max(float(body.get("limit", batch.DEFAULT_LIMIT)), 0.0), 100.0)
            stock = int(body.get("stock") or 0)
        except (TypeError, ValueError, AttributeError) as error:
            raise HTTPException(422, "These results could not be read.") from error
        return batch.analyse(results, limit, stock if stock > 0 else None)

    @app.get("/api/products")
    def products():
        return {"products": sorted(settings.supported.values())}

    @app.post("/api/inspect")
    def inspect(photo: UploadFile, request: Request):
        data = photo.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "This photo is too large. The limit is 15 MB.")
        try:
            return service.inspect(read_photo(data, settings), request)
        except HTTPException:
            raise
        except Exception as error:
            log.exception("inspection failed")
            raise HTTPException(500, "The check could not finish. Please try again.") from error

    @app.post("/api/inspect-batch")
    def inspect_batch(photos: list[UploadFile], request: Request, limit: float = Form(batch.DEFAULT_LIMIT), stock: int = Form(0),
                      cpu: bool = Form(False)):
        """Checks a batch together, so it uses one GPU run. A bad photo is reported without stopping the others.
        cpu: the page knows the GPU refused this visitor recently, so the check goes straight to the CPU."""
        if len(photos) > settings.max_batch:
            raise HTTPException(413, f"Send at most {settings.max_batch} photos at a time.")
        images, slots = [], []
        for photo in photos:
            data = photo.file.read(settings.max_upload_bytes + 1)
            try:
                if len(data) > settings.max_upload_bytes:
                    raise HTTPException(413)
                images.append(read_photo(data, settings))
                slots.append(None)
            except HTTPException as error:
                slots.append({"error": "tooLarge" if error.status_code == 413 else "notImage"})
        limit = min(max(limit, 0.0), 100.0)
        try:
            stock = stock if stock > 0 else None
            if not images:
                checked = {"results": [], "analysis": None}
            elif cpu and hasattr(service, "on_cpu"):
                checked = service.on_cpu(images, limit, stock)
            else:
                checked = service.inspect_many(images, request, limit, stock)
            found = iter(checked["results"])
            return {"results": [slot or next(found) for slot in slots], "analysis": checked["analysis"], "fallback": checked.get("fallback", False)}
        except HTTPException:
            raise
        except Exception as error:
            log.exception("inspection failed")
            raise HTTPException(500, "The check could not finish. Please try again.") from error

    html = versioned_page() if STATIC.exists() else ""

    @app.get("/")
    def page():
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
