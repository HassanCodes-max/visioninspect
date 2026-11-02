# One photo of an unseen item: AnomalyCLIP result

Run on 2026-11-02 on a Kaggle CPU, 720 VisA images, 19.75 seconds per image. Rules: [PLAN_ONE_PHOTO.md](../probes/PLAN_ONE_PHOTO.md). Raw numbers: [one_photo/](one_photo/).

## Verdict

**The check did not pass.** Under the plan, this method cannot promise confident results on any item, so VisionInspect stays with its named product list.

| Check | Limit | Result |
|---|---:|---:|
| Automatic decisions that are correct | At least 95% | 62.0% |
| Photos decided automatically | At least 60% | 12.8% |
| Defective items wrongly called GOOD | At most 5% | 1.4% |

The low miss rate does not help, because 87.2% of photos went to REVIEW. The peak of the anomaly map, reported alongside and deciding nothing, also fell short: 15.3% decided automatically and 80.9% of those correct.

## Other measures

| Measure | Result |
|---|---:|
| AUROC, each type on its own scale | 0.808 |
| AUROC, all types on one scale | 0.668 |
| Balanced accuracy at one cut-off, on left-out types | 54.2% |

The first row is close to the 82.1 reported by the authors, so the setup looks right. It needs a separate scale for each product, and a public website has no such scale for an unseen item. That is the same weakness the fine-tuned ViT showed: it can rank an unseen product's photos, but a single cut-off does not carry across products.

The balanced accuracy compares with 64.4% for the fine-tuned ViT on VisA products it never saw. The cut-off here was chosen differently, so this is a rough comparison.

## What this leaves

Three open methods have now fallen short on unseen items: the Nazar comparison method, the fine-tuned ViT and AnomalyCLIP. The option left in the plan is a vision LLM test. It has not been run, it needs an API key, and it costs money per photo.

## Limits

- 12 product types is a small sample of "any item".
- Each type used 30 good and 30 defective images, far more defects than a real line.
- Each result comes from one run.
