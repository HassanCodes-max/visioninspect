# VisionInspect Stage 2 results

Run on 2026-11-02 on Google Colab, using the fine-tuned ViT-B from [finetune.py](../probes/finetune.py), one run with no variants compared. Rules: [PLAN_VIT_85.md](../probes/PLAN_VIT_85.md). Raw numbers: [colab_stage2/](colab_stage2/).

## Verdict

**Known products passed; new products did not.** Under the plan, the site lists its supported products, claims 85% only for those, and sends other items to REVIEW.

Balanced accuracy at one fixed cut-off, one photo, no reference photo. Half the test photos are defective.

| Level | All | VisA | MVTec | Defects caught | False alarms |
|---|---:|---:|---:|---:|---:|
| Known products | **91.7%** | 88.5% | 94.7% | 93.0% | 9.7% |
| New products | 64.3% | 64.4% | 65.2% | 68.1% | 39.4% |

Of the 27 known product types, 25 reached 85%. The two below were pcb3 (74.0%) and mvtec_screw (83.0%). Of the new product types, 3 of 27 reached 85%.

## Stage 1 comparison

Stage 1 ran ViT-S on a CPU with a frozen transformer, 9 folds, on the same photo sample. Raw numbers: [kaggle_stage1/](kaggle_stage1/). Stage 2 also changed the backbone, so the gain is not from fine-tuning alone.

| Level | Best Stage 1 variant | Stage 2 |
|---|---:|---:|
| Known products | 79.4% (`blend`) | 91.7% |
| New products | 62.0% (`patch`, `patch_self`) | 64.3% |

Self-comparison gave no gain on new products in Stage 1: `patch` and `patch_self` tied at 62.0%. On known products it helped a little, 74.5% to 77.0%.

## Phone photos

Only 71 photos were found, not the 300 planned: 37 cracked and 34 intact phones.

| Model | Balanced accuracy | Defects caught | False alarms |
|---|---:|---:|---:|
| Models that never saw the phone's product type, 3 folds | 70.2% | 87.4% | 47.1% |
| Known-products model | 55.9% | 100.0% | 88.2% |

The last row matters most: on an item outside its list, the known-products model flagged nearly every intact phone. The site must not score unsupported items with it.

## Why new products failed

Many new products ranked well but landed on the wrong side of the shared cut-off, such as mvtec_grid (AUROC 1.000, balanced 81.6%) and pcb1 (0.939, 55.5%). The cut-off itself swung from 0.46 to 0.998 between folds.

## Follow-up: a cut-off from good photos of the new product

Chosen after seeing these results, so it is a lead and not a measured claim. [product_cutoff.py](../probes/product_cutoff.py) reused the saved scores with no retraining.

| Good photos | Balanced accuracy | Types at 85% |
|---:|---:|---:|
| 3 | 70.6% | 7 of 27 |
| 5 | 70.2% | 7 of 27 |
| 10 | 69.8% | 7 of 27 |
| 20 | 70.1% | 7 of 27 |

This helps but does not reach 85%. Cut-offs alone will not make new products work, and catch rates fell to about 42% to 47%.

## Limits

- The 91.7% is for these exact dataset products in their studio setups. A seller's own bottle is a different product from MVTec's bottle, so it may behave more like the new-product row.
- Half of each product's photos were used for training, so test photos are close to training photos.
- MVTec and the phone set are non-commercial, so this model is for a portfolio demo. A commercial version would need retraining on data that allows it.
- Each result comes from one run. No repeat runs were made.
