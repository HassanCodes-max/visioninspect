# Outlining the damaged area

Run on 2026-11-02 on the local CPU with SAM 2.1 hiera-small (Apache 2.0), using [sam_check.py](../probes/sam_check.py). Overlap is intersection over union with the VisA defect masks, on 79 defective photos of 4 product types. Half of these photos were in the trained model's training set, so its row is flattering.

| Method | Mean overlap |
|---|---:|
| SAM 2 from a good box, 25% wider than the true defect | 0.386 |
| Trained model's own map | 0.290 |
| SAM 2 from the trained model's own box | 0.242 |
| The same wider box with no SAM 2 | 0.237 |

SAM 2 beat the trained map on 24 of 79 photos when it used the trained model's own box.

## Decision

Items with a trained model keep their own map. Other items use SAM 2 on the box from the vision language model, because an outline beats a box when the box is in the right place.

The first row is a best case. The AI's boxes are looser and sometimes in the wrong place, and how much that lowers the score has not been measured.

VisA defects are small, which keeps these scores low. On one large tear in a web photo, SAM 2 traced the damage closely by eye. That is one photo, not a measurement.

## Smooth outlines from the trained model's map

Run on 2026-11-02 on the local CPU on the same 79 photos, using [outline_check.py](../probes/outline_check.py). The same warning about the training set applies.

| Outline | Mean overlap |
|---|---:|
| The 32×32 map, smoothed to 256×256 | 0.314 |
| The 32×32 map, blocky | 0.290 |

Smoothing was better on 65 photos and worse on 8. The site now shows the smooth outline for every REJECT or REVIEW on a listed product, and SAM 2 on the AI's box when the map marks nothing.

The marked area is only a rough size. On the 76 photos where both sizes were available, the median marked area was 1.75 times the true area, and 38 were within double or half of it. The site says so under "How far to trust it".
