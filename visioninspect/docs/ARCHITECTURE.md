# VisionInspect AI: architecture

Last updated 2026-11-02.

## Overview

```text
Browser (HTML, CSS, plain JavaScript, English and Urdu)
  │  resizes photos to at most 1,600 px, sends up to 30 in one request
  ▼
FastAPI routes                                       backend/app/main.py
  │  POST /api/inspect-batch   (photos, limit, stock)
  ▼
One ZeroGPU run for the whole batch                  backend/space_app.py
  │
  ├─ Recogniser: DINOv2 ViT-S + gallery ── listed product? ──┐
  │                                                         │
  │  yes                                                    │  no
  ▼                                                         ▼
  Defect finder: fine-tuned DINOv2 ViT-B        Qwen3-VL-8B: item, verdict, defect,
  score + 32×32 map                             box, severity, fix
  │                                                         │
  ▼                                                         ▼
  PASS / REVIEW / REJECT from two cut-offs      REJECT if defect, PASS if good
  if not PASS: Qwen3-VL-8B is told where the     SAM 2.1 outlines the box
  defect is and names it with a fix
  outline from the smoothed map                 
  │                                                         │
  └──────────────────────────┬──────────────────────────────┘
                             ▼
  Batch statistics in code: defect rate, Wilson 95% range, Pareto of defects,
  severity counts, mean damaged area, stock estimate, hold or accept
  Qwen3-VL-8B: likely causes and next steps, with no numbers
                             │
                             ▼
Browser: cards with outlines, detail view, batch analysis, history,
report by person and product (local storage), spreadsheet and PDF export
```

## Back end

| File | Job |
|---|---|
| `app/main.py` | Routes, photo reading, the `Service` that runs the models and builds each result |
| `app/config.py` | Settings from the environment, and the 25 listed products |
| `app/pipelines/vision.py` | Recogniser and defect finder |
| `app/pipelines/quality.py` | Decisions and marked regions. No model code |
| `app/pipelines/explain.py` | Qwen3-VL prompt and strict parsing of its reply |
| `app/pipelines/segment.py` | SAM 2 outlines, map outlines, and mask packing for the browser |
| `app/pipelines/batch.py` | Batch statistics and the AI's causes and next steps |
| `space_app.py` | ZeroGPU entry point: loads models onto the GPU and serves the site's routes |

### Decision rules

| Route | Result |
|---|---|
| Listed product, score at or above the high cut-off | REJECT |
| Listed product, score at or below the low cut-off | PASS, with no AI call |
| Listed product, score in between | REVIEW |
| Other item, AI says "defect" | REJECT |
| Other item, AI says "good" | PASS, shown with its tested share |
| Other item, AI says "unsure" | REVIEW |

### Results returned for each photo

Decision, reason, recognised product, defect score, marked regions, damaged area as a share of the photo, the 32×32 map, the outline as 256×256 packed bits, and the AI's explanation: item, defect, where, severity and fix.

### Batch statistics

All numbers are computed in `batch.py`. The defect rate comes with a Wilson 95% range. The lot advice compares that range with the buyer's limit: hold if the lower end is above it, accept if the upper end is within it even when every REVIEW photo is counted as faulty, and check more items otherwise. The AI is given these facts and asked only for causes and actions. Any suggestion that contains a digit is dropped.

## Front end

`app/static/` holds the page, styles, the English and Urdu text, and the script. There is no build step. The script:

- resizes photos in the browser and sends them in one request, with a scanning view while they are checked;
- shows a result panel for the worst item first: outlined fault, damage gauge, severity meter and how to fix it;
- lists every fault with its fix under the results;
- draws outlines from the packed mask, or the heat map when there is no outline;
- shows the batch analysis, the detail view with a compare slider, and the filters;
- keeps past batches and the report by person and product in local storage;
- exports results and reports as spreadsheets, and prints the report or saves it as a PDF.

The page has five views (Inspect, Report, Products, How it works, Account), switched by the address hash with animated transitions, so links and the back button work.

Sign-in uses Hugging Face accounts. The Space sets `hf_oauth: true`, a hidden `gr.LoginButton` makes Gradio add `/login/huggingface`, `/login/callback` and `/logout`, and `/api/me` reads the signed-in name from Gradio's session. The name is only used to fill the person field for reports. The site has no password, database or accounts of its own.

`fluid.js` runs a small fluid simulation on the GPU with WebGL 2 behind the page: ink that follows the pointer and drifts when idle. It pauses in hidden tabs, stays off for reduced motion, and the page falls back to a plain gradient where WebGL 2 is missing.

The page route tags every style and script address with a hash of its content and sends `Cache-Control: no-cache`, so browsers always load the newest version after an update.

`motion.js` adds the motion layer: sections that ease in on scroll, a scroll progress line, the current section marked in the menu, cursor glow and tilt on cards, button ripples, smooth dialog closing, short confirmation notes and a language crossfade. All motion stops when the device asks for reduced motion, and the page works without this file.

Example batches are saved results in `static/samples/`, so they use no GPU.

## Hosting

Free Hugging Face accounts cannot host Docker or CPU Spaces, so the site runs as a Gradio Space on ZeroGPU. The site's FastAPI routes are placed ahead of Gradio's, and Gradio's own page is turned off. Each batch asks for `8 + 3 × photos` seconds of GPU. Visitors who are not signed in were refused after about 3 GPU runs, so batches count as one run and the example buttons need none. When the GPU refuses a check, `on_cpu` runs it on the Space's CPU with CPU copies of the recogniser and trained model. Listed products still get their decision, outline and size. Other items need the GPU explainer, so they go to REVIEW with the reason `ai_resting`, and the page shows a note instead of an error. On the live Space, one photo took 3.3 s and a batch of 10 took 7.6 s this way.

`deploy_space.py` copies `backend/` to the Space and writes its requirements file.

## Privacy

Photos are read in memory, used once and not stored or logged. The report, history and the person's name stay in the browser. If a visitor signs in, Hugging Face shares their name and username with the site through a signed session cookie. Nothing is stored on the server.

## Tests

`tests/test_visioninspect_*.py`: 93 tests covering decisions, regions, outlines, AI reply parsing, batch statistics, the API routes and model loading. The site was also tested in headless Edge on desktop and phone sizes, in English and Urdu.
