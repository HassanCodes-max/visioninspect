# The backup AI on fresh photos

When the free GPU is used up, the site runs on the CPU, and a smaller AI, Qwen3-VL-2B-Instruct, explains items that have no trained model. This test checks how far its answers can be trusted.

Run on 2026-11-03 on Kaggle CPU, with the same code as the site ([explain.py](../backend/app/pipelines/explain.py)): bfloat16, photos shrunk to 640 px, and no hint from a trained model, as for unlisted items. The photos are 60 of the 80 fresh VisA photos used for the GPU AI ([RESULTS_FRESH.md](RESULTS_FRESH.md)): every photo from three of the four equal, mixed parts of that set. The last part was not run, because the result was already clear and each part takes about 50 minutes on Kaggle CPU.

| Measure | Result | 95% range |
|---|---:|---:|
| Balanced accuracy | **56.2%** | |
| Defects caught | 4 of 32 (12.5%) | 5.0% to 28.1% |
| False alarms on good photos | 0 of 28 (0%) | 0.0% to 12.1% |
| "Defect" answers that were right | 4 of 4 | 51.0% to 100.0% |
| "Good" answers that were right | 28 of 56 (50%) | 37.3% to 62.7% |
| Answers that were unsure or unreadable | 0 | |
| Seconds per photo on Kaggle CPU | 157 (median) | |

| Product | Defects caught | False alarms |
|---|---:|---:|
| Candle | 0 of 7 | 0 of 7 |
| Chewing gum | 4 of 9 | 0 of 4 |
| PCB4 | 0 of 7 | 0 of 8 |
| Pipe fryum | 0 of 9 | 0 of 9 |

All 4 caught defects were cracks in chewing gum, each with a severity and a fix.

## What changed on the site

A "good" answer from the backup AI is right only half the time on these photos, so it says nothing useful. The site now sends those items to REVIEW with the reason `ai_backup_good`, instead of passing them. A "defect" answer still rejects, with its name, outline and fix. The site shows 100% as its tested share, but that rests on only 4 answers, so the true figure could be as low as 51%.

On the GPU, the 8B AI caught 23 of 40 defects on the full set, so a photo the backup AI sent to REVIEW can be checked again later for a full opinion. The 25 listed products do not depend on either AI for their decision.

## Limits

- 60 studio photos from 4 VisA product types. One run with no repeats.
- The live Space's CPU is faster than Kaggle's: there, one photo took about 10 to 29 s.
