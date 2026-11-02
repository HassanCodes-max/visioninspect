# VisionInspect AI

Visual quality inspection from one photo, built with free tools only. Live at <https://hassanali0987654-visioninspect.hf.space> (also on its [Hugging Face page](https://huggingface.co/spaces/HASSANALI0987654/visioninspect)).

Upload one product photo, or a batch of up to 30. A scanning view shows each check, then a result panel shows the outlined fault, a damage gauge and a severity meter. Each photo gets PASS, REVIEW or REJECT, an outline of the damaged area, how much of the photo it covers, how serious it is and how to fix it. Batches get a defect rate with a 95% range, the most common defects, a stock estimate and a hold or accept decision. A report builds up for each person and product, and can be saved as a spreadsheet or PDF. Visitors can sign in with their Hugging Face account to put their name on reports.

## How it works

- **Listed products:** a fine-tuned DINOv2 ViT-B scores the photo and maps the defect. It reached 91.7% balanced accuracy on 25 products, and on 2,323 test photos 95% of its automatic decisions were correct.
- **Other items:** Qwen3-VL-8B names the fault, rates its severity and suggests a fix. SAM 2 outlines the damage. A "good" answer from the AI is a PASS, shown with how often that was right in testing (70%), because the AI misses small defects.
- **When the free GPU is used up:** the check runs on the CPU instead of failing. Listed products keep full results. Other items get a smaller backup AI, Qwen3-VL-2B. Its "defect" answers reject, with a name and a fix. It caught only 4 of 32 defects in testing, so its "good" answers go to REVIEW ([results](visioninspect/results/RESULTS_BACKUP_AI.md)).
- **Statistics:** every number is computed in code. The AI only suggests likely causes and next steps.

See the [project plan](visioninspect/docs/PROJECT_PLAN.md), [architecture](visioninspect/docs/ARCHITECTURE.md), [model selection](visioninspect/docs/MODEL_SELECTION.md) and the [results](visioninspect/results/).

## Layout

- `visioninspect/backend/`: FastAPI service, model pipelines and the web page
- `visioninspect/probes/`: training and test scripts behind every reported number, with the rules fixed before each run
- `visioninspect/results/`: results, with raw numbers
- `visioninspect/docs/`: plan, architecture, model and dataset choices
- `tests/`: unit, API and model tests

## Model weights

The fine-tuned model, `known_model.pt` (333 MB), is too large for GitHub. Download it from the Space's files, <https://huggingface.co/spaces/HASSANALI0987654/visioninspect/tree/main/models>, into `visioninspect/backend/models/`. The tests do not need it.

## Run the tests

```
pip install torch -r visioninspect/backend/requirements.txt pytest httpx
pytest tests
```

## Licences

DINOv2, SAM 2 and Qwen3-VL are Apache 2.0. Training used VisA (CC BY 4.0) and MVTec AD (CC BY-NC-SA 4.0), so the trained model is for non-commercial use. This tool does not replace certified inspection or safety-critical testing.
