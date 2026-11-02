# Probe: can one photo of an unseen item give a confident answer?

Fixed on 2026-11-01, before any run. This tests the riskiest idea in VisionInspect AI: one photo of almost any item, no good reference photo, and a result people can rely on.

## Method under test

AnomalyCLIP (ICLR 2024, MIT licence), with the authors' published `9_12_4_multiscale_visa` weights, which were trained on MVTec AD and have never seen VisA. Settings are the authors' own: CLIP ViT-L/14@336px, 518×518 input, depth 9, n_ctx 12, t_n_ctx 4, DPAM layer 20.

The primary score is the model's image-level probability, the same score the paper reports. The peak of its anomaly map is reported alongside and decides nothing.

## Data

VisA, all 12 product types, 30 good and 30 defective images per type, seed 0. That is 720 images. The sample is half defective, which is far more than a real line.

## What is measured

1. **Per-type AUROC.** A sanity check against the 82.1 reported for VisA. It allows a separate scale per product, which a public website cannot have.
2. **Pooled AUROC.** All 12 types on one scale.
3. **Three-way decision on an unseen type.** For each type in turn, two cut-offs are chosen on the other 11 types and applied to the type left out. Scores below the low cut-off become GOOD, scores above the high cut-off become DEFECT, and the rest become REVIEW. Each cut-off is set for 95% correct on the 11 types, keeping as many automatic decisions as possible.

## Decision rule

Averaged over all 720 images, measured on left-out types only.

| Check | Limit |
|---|---:|
| Automatic decisions that are correct | At least 95% |
| Photos decided automatically (coverage) | At least 60% |
| Defective items wrongly called GOOD, as a share of all defective items | At most 5% |

**Pass:** one photo of an unseen item is viable with open models, and the plan can build on it.

**Fail:** do not promise confident results on any item from this method. The plan moves to a named product scope, a vision LLM test, or both.

## Limits known before the run

- VisA has neat studio photos, so phone photos will be harder.
- 12 product types is a small sample of "any item".
- VisA was used in the Nazar tests, but this model and these cut-offs have not been tuned on it.
