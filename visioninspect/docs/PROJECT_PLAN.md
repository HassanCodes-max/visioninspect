# VisionInspect AI: project plan

Last updated 2026-11-02. Live at <https://huggingface.co/spaces/HASSANALI0987654/visioninspect>.

## Goal

Give a clear inspection result from one photo: PASS, REVIEW or REJECT, with where the fault is, how large and how serious it is, and how to handle it. Batches of photos also get statistics a factory or shop can act on, such as the share of damaged stock with a confidence range and a hold or accept decision.

## Constraints

- **Free tools only.** Open models, Hugging Face ZeroGPU, Kaggle and Colab. No paid APIs or hosting.
- **85% target.** At least 85% balanced accuracy from one photo, using fine-tuned Vision Transformers.
- **Honest limits.** Only measured numbers are shown. The tool does not replace certified inspection or safety-critical testing.
- **Licences.** MVTec AD is CC BY-NC-SA 4.0, so the trained model is for non-commercial use.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 1. Documents | Dataset plan, project plan, architecture, model selection | Done: [DATASET_PLAN.md](DATASET_PLAN.md), this file, [ARCHITECTURE.md](ARCHITECTURE.md), [MODEL_SELECTION.md](MODEL_SELECTION.md) |
| 2. Data pipeline | VisA and MVTec AD loaders, fixed seeds, held-out product types | Done |
| 3. Training | Fine-tuned DINOv2 ViT-B with defect masks | Done: 91.7% balanced accuracy on listed products ([results](../results/RESULTS_STAGE2.md)) |
| 4. Segmentation | Outline of the damaged area | Done: the trained model's map for listed products, SAM 2 for other items ([results](../results/RESULTS_SEGMENT.md)) |
| 5. Explanation | What the fault is, where it is, how serious it is, how to fix it | Done: Qwen3-VL-8B ([results](../results/RESULTS_FREE_LLM.md)) |
| 6. Analytics and reports | Batch statistics, report by person and product, spreadsheet and PDF export | Done |
| 7. Full stack | FastAPI back end and a web front end in English and Urdu | Done, with plain JavaScript in place of React and browser storage in place of PostgreSQL |
| 8. Testing | Unit, API and browser tests, plus a live check | Done: 93 VisionInspect tests, [site check](../results/RESULTS_SITE_CHECK.md), [live check](../results/RESULTS_LIVE.md) |
| 9. Deployment | Free public hosting | Done on Hugging Face ZeroGPU |
| 10. Video and tracking | Inspection of moving items on a line | Not started. No public dataset has labelled conveyor video, see [DATASET_PLAN.md](DATASET_PLAN.md) |

React, PostgreSQL and the video phase need Hassan's approval before work starts. PostgreSQL would also need a free database host, because the free Space has no permanent storage.

## Result against the 85% target

| Scope | Balanced accuracy | Meets 85% |
|---|---:|---|
| The 25 listed products, one photo | 91.7% | Yes |
| The same 25 products, automatic decisions only | 95% correct, 90% of photos decided | Yes |
| Product types never seen in training | 64.3% | No |
| Other items, judged by Qwen3-VL-8B | 77.5% on 80 fresh photos | No |

The site therefore lists the products it was trained on and claims accuracy only for those. Other items get the AI's opinion. A "defect" answer is a REJECT, and a "good" answer is a PASS shown with its tested share of 70%.

## Open work

1. Measure how many free checks visitors get and when the limit resets.
2. Gather a small set of real phone photos, because nearly all training and test photos come from studio cameras.

## Rejected options

- **AnomalyCLIP** for unseen items: 62.0% of automatic decisions were correct.
- **YOLO11n** defect boxes: 57.7% balanced accuracy on new product types, with false alarms on 34% of good photos.
- **Gemma 4** vision models: the 12B model scored lower than Qwen3-VL-8B, and the 26B model did not fit on the 48 GB GPU.

Details are in [MODEL_SELECTION.md](MODEL_SELECTION.md).
