# VisionInspect AI: model selection

Last updated 2026-11-02. Every model below was tested on held-out photos before it was chosen or rejected. "Balanced accuracy" is the average of defects caught and good items passed, at one fixed cut-off, from one photo with no reference photo.

## Chosen models

| Job | Model | Licence | Why |
|---|---|---|---|
| Find defects on listed products | DINOv2 ViT-B, fine-tuned | Apache 2.0 (DINOv2); trained on CC BY-NC-SA data | 91.7% balanced accuracy on the listed products |
| Recognise the product | DINOv2 ViT-S with a gallery of 2,004 photos | Apache 2.0 | Sends photos that are not close to a listed product to the AI route |
| Explain the fault on any item | Qwen3-VL-8B-Instruct | Apache 2.0 | Best free model tested: 77.5% on 80 fresh photos, with 1 false alarm on 40 good photos |
| Outline the damaged area | Trained model's map, or SAM 2.1 hiera-small | Apache 2.0 | The map for listed products, SAM 2 on the AI's box for others |

## Defect finder

DINOv2 ViT-B at 448×448 gives 1,024 patch tokens. Each patch gets three inputs: its token, its token minus the photo's mean, and its distance to its 16 most similar patches in the same photo. A small head scores each patch, and the photo score is the mean of the top 1% of patches, so a small defect can still raise it.

Training settings, from [finetune.py](../probes/finetune.py):

- Last 4 transformer blocks unfrozen, with layer-wise learning rate decay of 0.75 from a top rate of 5e-5, and 1e-3 for the head.
- Focal loss plus Dice loss on the defect masks, plus an image-level loss.
- Photometric and geometric augmentation, and synthetic cut-paste defects.
- AdamW with cosine decay, 5 epochs, on a free Colab GPU.

| Model | Listed products | New product types |
|---|---:|---:|
| DINOv2 ViT-S, frozen, best head (Stage 1, CPU) | 79.4% | 62.0% |
| DINOv2 ViT-B, fine-tuned (Stage 2) | **91.7%** | 64.3% |

Of the 27 trained product types, 25 reached 85%. Circuit board 3 (74.0%) and the MVTec screw (83.0%) did not, so they are left off the site's list. Full results: [RESULTS_STAGE2.md](../results/RESULTS_STAGE2.md).

Scores are mapped to PASS, REVIEW or REJECT with two cut-offs set from held-out results. On 2,323 test photos, 90% were decided automatically and 95% of those decisions were correct. The other 10% go to REVIEW.

## Recognition

A photo is matched against 2,004 gallery photos of the 27 trained product types. If the mean similarity of the 5 closest matches is below 0.847, or the closest product is not listed, the photo takes the AI route. This keeps the trained model off items it was not trained for: on intact phones, which it never saw, it flagged 88% as defective.

## Explanation model

Run on 32 blind VisA photos in a private ZeroGPU Space, with the prompt in [explain.py](../backend/app/pipelines/explain.py):

| Model | Balanced accuracy | Defects caught | Good passed | Seconds per photo |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B | **75.0%** | 8 of 16 | 16 of 16 | 1.9 |
| Gemma 4 12B | 68.8% | 6 of 16 | 16 of 16 | 3.8 |
| Gemma 4 26B | Did not fit in 48 GB | | | |

On 80 fresh photos it then reached 77.5%: 23 of 40 defects caught and 1 false alarm on 40 good photos ([results](../results/RESULTS_FRESH.md)). It misses many defects. A "defect" answer becomes a REJECT (23 of 24 were right), and a "good" answer becomes a PASS that shows its tested share (39 of 56 were right). An unsure answer goes to REVIEW. Batch numbers are always computed in code. The AI only suggests causes and next steps, and any suggestion containing a digit is dropped. Full results: [RESULTS_FREE_LLM.md](../results/RESULTS_FREE_LLM.md).

When the trained model flags a listed product, the AI is told where the defect is, so it names the fault and a fix even when it would have missed it. On the live site, 2 of 2 rejected photos the AI had missed on its own then got a named fault and a fix.

## Outlines

Mean overlap (intersection over union) with the true defect masks, on 79 defective VisA photos:

| Method | Overlap |
|---|---:|
| SAM 2 from a good box | 0.386 |
| Trained model's map, smoothed | **0.314** |
| Trained model's map, blocky | 0.290 |
| SAM 2 from the trained model's own box | 0.242 |

Listed products use the smoothed map. Other items use SAM 2 on the AI's box, because an outline beats a box when the box is in the right place. The marked area is a rough size: its median was 1.75 times the true area. Full results: [RESULTS_SEGMENT.md](../results/RESULTS_SEGMENT.md).

## Rejected models

| Model | Test | Result | Reason |
|---|---|---|---|
| AnomalyCLIP | 720 VisA photos of unseen items | 62.0% of automatic decisions correct, 12.8% decided | Needs a separate scale per product ([results](../results/RESULTS_ONE_PHOTO.md)) |
| YOLO11n, one "defect" class | 1,148 photos of 6 unseen product types | 57.7% balanced, 34% false alarms | Barely above chance on new products ([results](../results/RESULTS_YOLO.md)) |
| Gemma 4 12B and 26B | 32 blind photos | 68.8%; 26B did not fit | Lower than Qwen3-VL-8B |

## Known limits

- Test photos are studio photos from VisA and MVTec AD, and half of each listed product's photos were used in training. Results on phone photos will be lower.
- The AI was tested on 80 fresh photos, but all from the same 4 VisA product types as the first test.
- Each result comes from one run, with no repeats.
