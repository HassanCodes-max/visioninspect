# The backup AI on fresh photos

When the free GPU is used up, the site runs on the CPU, and a smaller AI, Qwen3-VL-2B-Instruct, decides items that have no trained model. Two ways of using it were measured on Kaggle CPU with the site's own code ([explain.py](../backend/app/pipelines/explain.py)), in bfloat16.

## First try: the written answer (not used)

The AI wrote its full answer, and its "defect" or "good" verdict decided. On 60 of the 80 fresh VisA photos ([RESULTS_FRESH.md](RESULTS_FRESH.md)), it caught only 4 of 32 defects, with no false alarms on 28 good items. Its "good" answers were right only 28 of 56 times. Writing each answer took about 10 to 30 s on the live Space.

## Used now: the defect score

The AI is asked one question, "defect or good", and the site reads how likely its first word is "defect" instead of letting it write. This takes one pass, with no text written. Photos are shrunk to 640 px.

Rules fixed before the fresh photos were scored:

- Cut-offs are set on the 32 photos of the earlier test, and checked once on the 80 fresh photos.
- Photo size: the one with the higher ROC area on the 32 (640 px: 0.832, 448 px: 0.812).
- One cut at the best balanced accuracy on the 32, then a REVIEW band around it, holding at most 25% of the 32.

The cut-offs are 0.005912 and 0.01243. At or below the first, the item passes. At or above the second, it is rejected, and anything in between goes to REVIEW.

| On the 80 fresh photos | Result | 95% range |
|---|---:|---:|
| Automatic decisions that were right | **63 of 69 (91.3%)** | 82.3% to 96.0% |
| REJECT answers that were right | 28 of 29 | 82.8% to 99.4% |
| PASS answers that were right | 35 of 40 | 73.9% to 94.5% |
| Photos sent to REVIEW | 11 of 80 (13.8%) | |
| Defects caught as REJECT | 28 of 40 | |
| Defects passed | 5 of 40 | |
| Good items rejected | 1 of 40 | |
| Balanced accuracy at the single cut | 87.5% | |
| ROC area | 0.937 | |

| Product | Defects rejected | Defects passed | Good items passed |
|---|---:|---:|---:|
| Candle | 6 of 10 | 4 | 10 of 10 |
| Chewing gum | 10 of 10 | 0 | 10 of 10 |
| PCB4 | 9 of 10 | 1 | 9 of 10 |
| Pipe fryum | 10 of 10 | 0 | 6 of 10 |

On the 32 photos used to set the cut-offs, 20 of 24 automatic decisions were right, and 8 went to REVIEW.

## On the live site

The decision comes from the score at once, and the page then asks for the fault's name, place, severity and fix, one photo at a time, worst first, and fills them in when they arrive. Timed on the live Space on 2026-11-03, with checks sent straight to the CPU:

| Check | Time |
|---|---:|
| One item without a trained model | 1.5 to 2.5 s |
| One listed product | 2.1 to 2.3 s |
| A batch of 5 mixed photos | 5.5 s |
| Each written explanation, afterwards | about 11 s |

## Limits

- All photos are studio photos from 4 VisA product types, which also have a trained model. Other items and phone photos have not been measured.
- The 32 tuning photos are few, so the cut-offs may shift with more data. One run with no repeats.
- Kaggle's CPU took about 110 s per score. The live Space's CPU is much faster, so the times above are from the live Space.
