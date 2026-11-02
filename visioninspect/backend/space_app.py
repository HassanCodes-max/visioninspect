"""VisionInspect on a Hugging Face ZeroGPU Space. Keeps the same site and API as app/main.py, with checks run on a shared GPU."""
import spaces  # must be imported before torch

import json
import logging

import gradio as gr
from gradio.context import LocalContext

from app.config import Settings
from app.main import Service, create_app, on_cpu
from app.pipelines import batch, quality
from app.pipelines import explain
from app.pipelines.explain import BackupExplainer, BackupSegmenter, Explainer, OnDemand
from app.pipelines.segment import Segmenter
from app.pipelines.vision import DefectFinder, Recogniser

settings = Settings()
bands = quality.Bands(**json.loads(settings.bands_path.read_text())["bands"])
recogniser = Recogniser.load(settings)
recogniser.encoder.to("cuda")
finder = DefectFinder.load(settings)
finder.network.to("cuda")
local = Service(settings, recogniser, finder, bands, Explainer.load("cuda"), Segmenter.load("cuda"))
backup_ai = BackupExplainer(OnDemand(lambda: Explainer.load("cpu", explain.CPU_MODEL, explain.CPU_SIDE)))
backup_outline = BackupSegmenter(OnDemand(lambda: Segmenter.load("cpu")))
cpu = Service(settings, Recogniser.load(settings), DefectFinder.load(settings), bands, backup_ai, backup_outline)
log = logging.getLogger("visioninspect")


def seconds(images, limit, stock):
    return 8 + 3 * len(images)


@spaces.GPU(duration=seconds)
def check(images, limit, stock):
    return Service.inspect_many(local, images, None, limit, stock)


class SharedGpuService(Service):
    def inspect_many(self, images, request=None, limit=batch.DEFAULT_LIMIT, stock=None):
        """Passing the visitor's request lets ZeroGPU count this check against that visitor's daily allowance.
        When the GPU refuses, usually because that allowance is used up, the check runs on the CPU instead of failing."""
        LocalContext.request.set(request)
        try:
            return check(images, limit, stock)
        except gr.Error as error:
            log.warning("GPU refused a check (%s), so it runs on the CPU", error.title)
            return on_cpu(cpu, images, limit, stock)

    def inspect(self, image, request=None):
        return self.inspect_many([image], request)["results"][0]


with gr.Blocks(title="VisionInspect") as demo:
    gr.Markdown("VisionInspect")
    gr.LoginButton(visible=False)  # makes Gradio add the Hugging Face sign-in routes the site uses

demo.launch(prevent_thread_lock=True, server_name="0.0.0.0", server_port=7860, ssr_mode=False)
demo.app.router.routes[:0] = create_app(SharedGpuService(settings), settings).routes
demo.block_thread()
