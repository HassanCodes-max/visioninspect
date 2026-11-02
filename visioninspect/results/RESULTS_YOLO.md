# YOLO defect boxes on new product types

Run on Kaggle on CPU with [yolo_train.py](../probes/yolo_train.py), because the weekly GPU allowance was used up. YOLO11n learned one class, "defect", from the VisA and MVTec AD masks of 21 product types. It stopped at the 9-hour limit after 22 epochs, so it is undertrained.

The test used 6 product types it never saw: candle, pcb2, pipe_fryum, bottle, leather and metal nut. That is 1,148 photos, 548 with a defect and 600 good. "Located" means a box overlapped the true defect by at least 0.1 intersection over union.

| Confidence | Balanced accuracy | Defects caught | False alarms on good photos | Defects located |
|---:|---:|---:|---:|---:|
| 0.10 | 60.4% | 71.4% | 50.5% | 34.9% |
| 0.25 | 57.7% | 49.6% | 34.2% | 22.4% |
| 0.50 | 52.0% | 21.0% | 17.0% | 8.6% |

At a confidence of 0.25, the results by product type were:

| Product type | Caught | False alarms | Located |
|---|---:|---:|---:|
| candle | 50% | 54% | 14% |
| bottle | 24% | 1% | 13% |
| leather | 0% | 0% | 0% |
| metal nut | 100% | 100% | 8% |
| pcb2 | 67% | 50% | 49% |
| pipe_fryum | 47% | 0% | 45% |

## Decision

YOLO is not added to the site. On new product types it is barely better than chance, and it raises false alarms on a third to a half of good photos. The vision language model gave no false alarms on its 16 good photos (see [RESULTS_FREE_LLM.md](RESULTS_FREE_LLM.md)), and SAM 2 already turns its box into an outline. The weights are not kept in the project.
