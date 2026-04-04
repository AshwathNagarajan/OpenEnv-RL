---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
# Email Triage OpenEnv

A real-world OpenEnv benchmark for customer-support email triage.

## Included tasks
- **easy**: predict category and priority
- **medium**: predict category, priority, route, and disposition
- **hard**: full triage including escalation and response drafting

## API endpoints
- `POST /reset`
- `GET /state`
- `POST /step`
- `POST /set_task/{task_name}`
- `GET /run` → runs demo inference for **easy + medium + hard** and returns JSON with `stdout`

## Notes
- `inference.py` keeps the strict `[START] / [STEP] / [END]` logging format for evaluation.
- `runner.py` is used by `/run` to produce richer human-readable output with task summaries and episode summaries.
- If `HF_TOKEN` and `API_BASE_URL` are missing, the project falls back to a deterministic heuristic agent so the Hugging Face demo still shows output.
