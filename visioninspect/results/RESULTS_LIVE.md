# VisionInspect live check

Published on 2026-11-02 at https://huggingface.co/spaces/HASSANALI0987654/visioninspect, on Hugging Face ZeroGPU, using [deploy_space.py](../deploy_space.py) and [space_app.py](../backend/space_app.py).

## Hosting

Free accounts can no longer host Docker or CPU Spaces. ZeroGPU is free, so the site runs there. The Dockerfile was removed on 2026-11-02, because the project uses free hosting only.

Gradio's page is replaced with the site's own routes, so the design is unchanged. Gradio's server-side rendering has to stay off, or Gradio's page shows at `/`.

## Result on the live GPU

| Check | Result |
|---|---:|
| One photo at a time, signed in | 30 of 30 answered, median 0.55 s |
| Batch of 30, signed in | 8.7 s in one GPU run |
| Same decision as Colab | 27 of 27 product photos |
| Largest score gap to Colab | 0.005 |
| Non-product photos sent to REVIEW | 3 of 3 |

This confirms that the live site matches the measured model. The measured accuracy itself is in [RESULTS_STAGE2.md](RESULTS_STAGE2.md).

## Visitors who are not signed in

Guest use is limited by Hugging Face. One connection was refused after 3 GPU runs, much lower than the 2 minutes a day the documentation suggests. The server's message was "You have exceeded your ZeroGPU runs limit". When and how the limit resets was not measured.

Since 2026-11-03, a refused check runs on the Space's CPU instead of failing. On the live Space a guest check took 3.3 s for one photo and 7.6 s for 10. Listed products keep full results. Other items are decided by a smaller backup AI's defect score: on the live Space a decision took 1.5 to 2.5 s per photo and 5.5 s for a batch of 5, and each written explanation followed about 11 s later. Its accuracy is in [RESULTS_BACKUP_AI.md](RESULTS_BACKUP_AI.md).

To work within this:

- The example buttons show saved results from real checks and use no GPU.
- A batch of up to 30 photos counts as one GPU run.
- When the limit is reached, the site explains it, and Hugging Face shows its own free sign-up prompt.

Signed-in accounts did not reach the limit in these tests. Hugging Face lists 5 minutes a day for free accounts.

## Limits

- A Space can go to sleep when it is not used, and the first visitor then waits for it to start.
- GPU availability depends on Hugging Face's shared pool, so waits may vary.
- A signed-in visitor's browser flow was tested with a token, not by clicking through a real login.

## AI layer on the live GPU

Tested on 2026-11-02 at 05:24 as the signed-in owner, with two leather photos in one batch and a stock of 500.

| Photo | Decision | AI answer | Outline |
|---|---|---|---|
| Torn leather | REJECT | Hole, major, patch suggested | Yes, 18.08% of the photo |
| Clean leather | REVIEW | No defect seen | None |

The batch analysis gave 50% damaged, a likely range of 9.5% to 90.5%, about 250 of 500 items, and Hold. The AI's suggested causes were general, and one, misaligned drilling or punching, does not fit a leather tear well. They are shown as suggestions for this reason.

This confirms that every part works together on the live GPU. It says nothing about accuracy, because it was two photos.
