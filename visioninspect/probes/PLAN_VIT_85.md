# Experiment: a Vision Transformer that reaches 85%

Fixed on 2026-11-02, before any run. Hassan set the goal: at least 85% accuracy on the site, using fine-tuned Vision Transformers and different techniques.

## What 85% means here

**Balanced accuracy** at one fixed cut-off: the average of defects caught and good items correctly passed. Plain accuracy is not used, because on a line with 2% defects a model that calls everything good would score 98%. One photo, no reference photo.

Two levels are measured, because they answer different questions:

| Level | Test photos come from | Question |
|---|---|---|
| New products (primary) | Product types left out of training and cut-off selection | Does it work on an item it has never seen? |
| Known products (secondary) | Product types in training, but different photos | Does it work on items it was trained for? |

## Data

VisA (12 types) and MVTec AD (15 types). For each type: up to 100 defective photos with masks, and 100 good photos, seed 0. MVTec is non-commercial, so a model trained on it stays a portfolio demo unless it is retrained.

## Stage 1: frozen transformer, trained heads

DINOv2 at 448×448 gives 1,024 patch tokens per photo. ViT-B on GPU, or ViT-S if only a CPU is available. Four variants:

| Variant | Technique |
|---|---|
| `cls` | Logistic regression on the whole-image token |
| `patch` | Small network on each patch, trained with defect masks |
| `patch_self` | The same, plus self-comparison: each patch minus the photo's average, and its distance to the 16 most similar patches in the same photo |
| `blend` | Average of `cls` and `patch_self` |

The photo score is the mean of its top 1% of patch scores, so a small defect can still raise it.

For new products, types are split into 9 groups, each left out once. Within each fold, 20% of the remaining types are kept aside to set the cut-off and are not used to fit the model. For known products, each type's photos are split in half, and the cut-off comes from 20% of the training half.

## Stage 2: fine-tuning

Runs only if Stage 1 is below 85% on new products and a GPU is available. It unfreezes the last 4 transformer blocks of the best variant, with a lower learning rate for earlier blocks, focal and Dice losses, photometric and geometric augmentation, and synthetic cut-paste defects. It follows the same fold rules, and its design may use what Stage 1 shows.

## Amendment, 2026-11-02: Stage 2 on Colab

Written before any Stage 1 result. Hassan asked to start fine-tuning on Google Colab now, without waiting for Stage 1 or the Kaggle GPU. So Stage 2 runs regardless of Stage 1. To fit a free Colab session, it uses 3 folds instead of 9 and 5 epochs, with every type still left out once. `finetune.py` holds the exact settings.

It adds a phone-photo check, chosen by Hassan in place of photos of his own: up to 150 cracked and 150 intact phones from the public Cracked & Intact Smartphone Images dataset (CC BY-NC 4.0, testing only). These are indoor photos with varied angles, backgrounds and lighting. They are never used for training or cut-offs. They cover one kind of product and one kind of defect, so the result is a real-world warning sign and not proof for phone photos of every item.

## Decision rule

| Result | What the site may say and do |
|---|---|
| New products at 85% or more on both VisA and MVTec | May claim 85% on new product types of this kind, and offer one-photo checks on general items |
| Only known products at 85% or more | The site lists the supported product types, claims 85% only for those, and sends other items to REVIEW |
| Neither | The target is not met. No 85% claim appears anywhere, and I report the best honest number and the next option |

The claim on the site must state the conditions it was measured under.

## Limits known before the run

- Both datasets are clean studio photos, so phone photos will be harder. A phone-photo test set is needed before any claim about phones.
- Half the test photos are defective, so catch rate and false-alarm rate are reported alongside balanced accuracy.
- The folds are used to compare variants, so a final claim needs one untouched test, ideally the phone set.
