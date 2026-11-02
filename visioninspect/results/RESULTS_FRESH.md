# The AI on fresh photos

Run on 2026-11-03 on Hugging Face ZeroGPU in a private test Space, with Qwen3-VL-8B and the same prompt as the site ([explain.py](../backend/app/pipelines/explain.py)). The prompt and model were fixed before this run, and none of these photos had been shown to any model before.

80 VisA photos: 10 defective and 10 good from each of candle, chewing gum, pcb4 and pipe fryum, chosen at random and given random names. The 32 photos from the earlier test were excluded.

| Measure | Result | 95% range |
|---|---:|---:|
| Balanced accuracy | **77.5%** | |
| Defects caught | 23 of 40 (57.5%) | 42.2% to 71.5% |
| False alarms on good photos | 1 of 40 (2.5%) | 0.4% to 12.9% |
| Answers that were unsure or unreadable | 0 | |
| Seconds per photo | 2.0 | |

| Product | Defects caught | False alarms |
|---|---:|---:|
| Candle | 3 of 10 | 0 of 10 |
| Chewing gum | 9 of 10 | 0 of 10 |
| PCB4 | 5 of 10 | 1 of 10 |
| Pipe fryum | 6 of 10 | 0 of 10 |

## What this means

The result matches the earlier test of 32 photos (75.0%): the AI rarely raises a false alarm but misses many defects, especially small ones on candles. The site treats a "defect" answer as a REJECT (23 of 24 right here) and a "good" answer as a PASS that shows its tested share (39 of 56 right here). It does not reach the 85% target, so the site claims 85% only for its 25 listed products.

## Limits

- All photos are studio photos from 4 VisA product types. Other items and phone photos have not been measured.
- One run with no repeats.
