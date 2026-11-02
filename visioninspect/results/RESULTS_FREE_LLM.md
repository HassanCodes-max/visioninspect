# Free vision language models

Run on 2026-11-02 on Hugging Face ZeroGPU in a private test Space, with the same 32 blind photos as [RESULTS_EYE_TEST.md](RESULTS_EYE_TEST.md). The goal was at least 85%, using free tools only. The prompt is in [explain.py](../backend/app/pipelines/explain.py).

| Model | Licence | Balanced accuracy | Defects caught | Good passed | Seconds per photo |
|---|---|---:|---:|---:|---:|
| Qwen3-VL-8B | Apache 2.0 | **75.0%** | 8 of 16 | 16 of 16 | 1.9 |
| Gemma 4 12B | Apache 2.0 | 68.8% | 6 of 16 | 16 of 16 | 3.8 |
| Gemma 4 26B | Apache 2.0 | Did not run | | | |

Gemma 4 26B needs 52 GB and the GPU has 48 GB.

## What this means

Neither model reached 85%. Both are cautious: they raised no false alarms on these photos, but they missed many defects. So the site treats a "defect" answer as a REJECT and never treats a "good" answer as a PASS.

Some catches were lucky. On the circuit boards, Qwen often blamed the dirty USB connector and not the real fault, so a correct verdict did not always come with a correct reason. It also named items wrongly at times, such as calling chewing gum soap.

## The visitor's leather photos

| Photo | Trained site before | Qwen3-VL-8B |
|---|---|---|
| Torn leather | REVIEW, not recognised | Defect: hole, major, with a box and a repair suggestion |
| Clean leather | REVIEW, not recognised | Good |

These are two photos, so this shows the idea works on them and nothing more.

## Zoom test

Adding four close-ups of each photo stopped after 6 photos when the daily GPU quota ran out: 2 of 5 defects caught, and the good photo passed. This is too few to judge.

## Limits

- All 32 photos come from one studio dataset, and three models have now been compared on them, so they are no longer a clean test.
- Fresh photos are needed for any number shown as a final result.
- A free account gets about 5 minutes of GPU a day, and guests were refused after about 3 runs.
