# VisionInspect AI: dataset plan

Phase 1, written 2026-11-01. "Verified" means read on the dataset's own page or in its own licence file. "Secondary" means seen only in a mirror, overview or search summary, and it must be checked before use.

## The main constraint

A dataset can only train the task its labels support. A classification label cannot train a detector, and a box cannot train a segmentation model without weak-label methods. Each model in the plan is therefore tied to a dataset with the right labels.

| Dataset | Content | Labels | Tasks it supports | Licence | Status |
|---|---|---|---:|---|---|
| VisA | 12 product types, 9,621 good and 1,200 defective images | Image labels and pixel masks | Anomaly detection, segmentation, one-class evaluation | CC BY 4.0 | Verified from `LICENSE-DATASET`; counts from our own runs |
| MVTec AD | 15 categories, over 5,000 images | Pixel masks | Anomaly detection, segmentation | CC BY-NC-SA 4.0, no commercial use | Verified on the MVTec page |
| MVTec AD 2 | Harder scenes with varied lighting | Pixel masks | Robustness evaluation | CC BY-NC-SA 4.0, no commercial use | Licence verified on the MVTec page |
| NEU-DET and NEU-CLS | 1,800 steel images, 6 classes, 300 each, 200×200 greyscale | Boxes in NEU-DET, class labels in NEU-CLS | Detection, classification | Unclear: authors ask for citation; mirrors disagree | Secondary, treat as research use and ask the authors before any commercial use |
| KolektorSDD2 | 3,335 images, 356 with defects, about 230×630 | Defect labels; the paper describes masks | Segmentation, classification | CC BY-NC-SA 4.0; contact the authors for commercial use | Licence verified on the ViCoS page; mask format to confirm |
| DAGM 2007 | 10 texture classes, synthetic | Coarse weak labels | Weakly supervised segmentation | CC BY 4.0 on the Heidelberg page | Licence seen via search summary of the official page; open and confirm |
| CarDD | 4,000 car photos, 9,163 damage instances: dent, scratch, crack, glass, tyre, lamp | Boxes and instance masks | Detection, instance segmentation of real-camera damage | Research only; image rights sit with Flickr and Shutterstock | Secondary, confirm on the official site |
| Casting parts | 512×512 top-view impellers, 519 good and 781 defective, plus an augmented set | Image labels only | Classification only | CC BY-NC-ND 4.0, no adapted versions | Verified from Kaggle metadata |
| Severstal | 12,568 steel images, 4 classes | Masks | Segmentation | Kaggle competition rules | Terms not checked |
| Real-IAD | About 150,000 images, 30 categories | Pixel masks | Large-scale anomaly evaluation | Not found | Terms not checked |

## What this means

- **No dataset gives boxes, masks and classes for "any item".** VisA and MVTec give masks but no defect-type classes across products. NEU-DET gives boxes and classes, but only for steel.
- **The biggest gap is phone photos.** Nearly all of these come from fixed industrial cameras or line scanners. Test 1 on Nazar showed accuracy dropping sharply under simulated handheld conditions. Results on these datasets cannot be quoted as results on phone photos.
- **We need our own small test set.** At least 200 real phone photos, about half with real defects, kept out of all training and tuning. Only you can create this. It is the only honest source of a number for your actual users.
- **Licences split the project in two.** Using CC BY-NC-SA data is fine for a portfolio demo. A model trained on it must not be sold. DAGM and VisA are the safest base for anything commercial.
- **Example photos on the website** should come from VisA only, with credit, because it is the clearest licence that allows reuse.

## Pipeline

One loader per dataset writes one shared record format, so every model reads the same structure.

```text
datasets/
  raw/<name>/            original files, never edited, git-ignored
  prepared/<name>/
    images/
    masks/               only where the dataset has masks
    labels/              boxes in YOLO or COCO format, only where they exist
    manifest.parquet     one row per image: split, product, label, source, checksum
```

Each loader must do the following:

- Validate that files open, masks match their images, and boxes stay inside the image.
- Report class counts and write a contact sheet of labels drawn on images, for checking by eye.
- Split by product or production group where that is known, so near-identical frames cannot land in both training and test sets.
- Record a checksum for every image, and fail the build if the same image appears in two splits.
- Apply augmentation to the training split only, never to validation or test sets.

## Task to dataset map

| Task | Train on | Evaluate on |
|---|---|---|
| Steel defect detection | NEU-DET | Held-out NEU-DET, then our phone set |
| Defect classification | NEU-CLS | Held-out NEU-CLS |
| Pixel segmentation | KolektorSDD2, MVTec masks | Held-out split, then VisA masks |
| Unseen-item anomaly detection | MVTec | VisA, the reverse of the training set |
| Real-camera damage detection | CarDD, research use only | Held-out CarDD, then our phone set |
| Video tracking | Needs a dataset | See below |

## Open gap: tracking

None of these datasets includes video of products on a conveyor, so ID-switch, IDF1 and HOTA cannot be measured from them. Phase 5 needs either a short video we record and label by hand, or a synthetic conveyor made by moving product images across a background, which would be labelled clearly as synthetic.
