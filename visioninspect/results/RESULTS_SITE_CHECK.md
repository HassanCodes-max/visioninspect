# VisionInspect site check

Run on 2026-11-02 on the local CPU, with the real trained model and recognition data, in headless Edge.

## Does the site reproduce the measured model?

The test split can be rebuilt from its seed, so 27 local chewing gum and candle photos were matched to their saved Colab scores.

| Check | Result |
|---|---:|
| Mean score gap, site vs Colab | 0.001 |
| Largest score gap | 0.005 |
| Same decision as Colab | 27 of 27 |

The match was reached after two fixes:

- Colab's `transformers` names the attention layers differently, so `fit_names` maps them on load.
- Training read each photo back from a 448 px JPEG at quality 95. Without that step, scores moved by up to 0.20 (mean 0.046), so the site now does the same with `as_trained`. The browser sends photos up to 1,600 px unchanged, because re-saving them could shift scores in the same way.

This also shows that the model is sensitive to small changes in compression. Scores on a photo near a band edge can move across it, which is one more reason results on other people's photos may differ from the measured numbers.

## Result in the browser

| Photos | PASS | REVIEW | REJECT |
|---|---:|---:|---:|
| 4 good | 4 | 0 | 0 |
| 23 defective | 0 | 4 | 19 |
| 3 non-product photos | 0 | 3 | 0 |

This set is small and mostly defective, so it confirms that the site works. It does not measure accuracy. The measured numbers come from the 2,323 test photos in [RESULTS_STAGE2.md](RESULTS_STAGE2.md).

Also checked: no browser errors, no sideways scrolling on a 390 px phone screen, English and Urdu layouts, filters, the detail view, the example buttons and the licence page.

## Speed

| Step | Time |
|---|---:|
| Server answers | 2 s |
| First check after a start, while models load | about 20 s |
| Each later check | 2 to 3 s |

These are laptop times. The free hosting CPU will be different, and it has not been measured.

## Not checked

- The Docker image has not been built. Hugging Face hosting went another way, described in [RESULTS_LIVE.md](RESULTS_LIVE.md).
- No real phone photos of supported products have been tried. Most ordinary photos are expected to go to REVIEW as not recognised.
- Results come from one trained model. No repeat runs were made.
