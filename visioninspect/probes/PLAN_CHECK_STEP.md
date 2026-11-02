# Experiment: can the site tell which photos it supports?

Fixed on 2026-11-02, before any run. The known-products model scored 91.7% on its 27 product types but flagged 88% of intact phones. So the site must recognise a supported product before it gives a verdict, and send everything else to REVIEW.

## Method

DINOv2 small at 224×224 gives one whole-image token per photo. Supported photos, good and defective, form a gallery. A new photo's score is its mean cosine similarity to its 5 closest gallery photos, and its product is the product of its closest gallery photo.

The cut-off is the 2nd percentile of scores from calibration photos of supported products, so it never uses unsupported photos. It was set to the 5th percentile until a test, before any run, showed that would make the 95% limit a coin flip.

## Data

The same 27 product types and sample as Stage 2, split in half per product and label: a gallery half and a test half. A fifth of the gallery half is kept for calibration.

Unsupported photos come from three sources:

| Source | Why it is included |
|---|---|
| Each of the 27 types, removed from the gallery in turn | The hardest case: similar studio photos of a product not on the list |
| 71 cracked and intact phones | Everyday photos |
| Up to 300 casting parts | Industrial photos from a dataset never used in training |

## Decision rule

| Check | Limit |
|---|---:|
| Supported photos accepted | At least 95% |
| Left-out product photos sent to REVIEW | At least 90% |
| Phone and casting photos sent to REVIEW | At least 95% |
| Accepted supported photos given the right product | At least 98% |

**Pass:** the site recognises the product by itself.

**Fail:** the user picks the product from a list, and the site says plainly that results apply only to the listed products. The check stays on as a warning if it is still useful.

## Limits known before the run

- Everyday clutter, people and screenshots are not in the test.
- A seller's own version of a listed product, such as a different bottle, may be accepted without being a good match for the model.
