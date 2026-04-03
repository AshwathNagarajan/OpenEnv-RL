# Email Operations Environment (OpenEnv)

An OpenEnv-compliant AI training environment for email triage and workflow automation using the Enron Email Dataset.

## 🎯 Overview

**Email Operations Environment** simulates a real-world enterprise email workflow where AI agents must classify, prioritize, and respond to emails at scale. This environment is designed to train and evaluate LLM-based agents on realistic email management tasks with meaningful reward signals and difficulty progression.

### Motivation

Modern organizations process thousands of emails daily. Manual triage is inefficient and error-prone. This environment enables:
- **Training**: Teach AI agents to automate email classification and routing
- **Evaluation**: Benchmark agent performance on real-world task complexity
- **Research**: Study agent behavior on human workflows and decision-making

### Real-World Applications
- Customer support automation
- Inbox triage and prioritization  
- Spam/fraud detection
- Escalation routing
- Email response generation

---

## 📊 Task Descriptions

### Task 1: Easy — Email Classification
**Objective**: Classify each email into one of four action categories.

**Actions**:
- `reply` — Send a helpful response
- `escalate` — Forward to a human specialist
- `spam` — Mark as unwanted
- `archive` — File as handled

**Grading**: Binary accuracy. +1.0 if `action_type` matches email label, 0.0 otherwise.

**Difficulty**: Easy  
**Expected Baseline Score**: 0.6–0.8

---

### Task 2: Medium — Classification + Response Generation
**Objective**: Classify emails AND generate thoughtful responses when applicable.

**Scoring** (capped at 1.0):
- Email classification accuracy: +0.5
- Response quality (length > 15 chars): +0.3
- Response tone (contains "sorry"): +0.2

**Grading**: Partial credit for partial decisions. Rewards meaningful engagement.

**Difficulty**: Medium  
**Expected Baseline Score**: 0.5–0.7

---

### Task 3: Hard — Full Workflow Automation
**Objective**: Full email workflow — classify, prioritize, handle escalations, generate responses, show consistency.

**Scoring** (grader in `env/grader.py`):
- Classification accuracy: +0.4
- Priority-aware escalation (+0.3 for correct escalation, -0.2 for incorrect)
- Response quality (length, tone, helpfulness): +0.3
- Behavioral consistency (penalty for repetitive actions): -0.3

**Difficulty**: Hard  
**Expected Baseline Score**: 0.4–0.6  
**Challenge**: Requires balancing multiple objectives and avoiding behavioral patterns.

---

## 🔄 Environment API

### Initialization
```python
from env.environment import EmailEnv
from env.tasks import EasyTask, MediumTask, HardTask

env = EmailEnv(task=EasyTask())
```

### Reset
```python
observation = env.reset()
# Returns: Observation(current_email: dict, remaining: int)
```

### Step
```python
action = {"action_type": "reply", "content": "..."}
observation, reward, done, info = env.step(action)
```

**Returns**:
- `observation`: Current email and remaining count
- `reward`: float in [-1.0, 1.0]
- `done`: bool, episode finished
- `info`: dict with metrics (accuracy, total_reward, errors)

### State
```python
observation = env.state()
```

---

## 🎮 Action & Observation Spaces

### Action Space
```json
{
  "action_type": "string (one of: reply, escalate, spam, archive)",
  "content": "string (optional, used only for reply actions)"
}
```

### Observation Space
```json
{
  "current_email": {
    "id": "string",
    "subject": "string",
    "body": "string",
    "sender": "string",
    "label": "string (action_type it should receive)",
    "priority": "string (high, medium, low)"
  },
  "remaining": "int (emails left in episode)"
}
```

### Reward Signal
- **Range**: [-1.0, 1.0]
- **Partial progress**: Rewards partial successes toward task goals
- **Penalties**: Delays, over-escalation, behavioral repetition
- **Shaping**: Includes penalties for undesirable real-world patterns

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)
- OpenAI API key (for baseline inference)

### Local Setup
```bash
# Clone repository
git clone https://github.com/yourusername/email-openenv.git
cd email-openenv

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation
```bash
# Test environment import
python -c "from env.environment import EmailEnv; print('✓ Environment loaded')"

# Run validation
openenv validate
```

---

## 📈 Baseline Inference

### Run Baseline Agent
```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4-turbo-preview"
export API_BASE_URL="https://api.openai.com/v1"

python inference.py
```

### Expected Baseline Scores
| Task   | Model              | Score | Steps |
|--------|-------------------|-------|-------|
| Easy   | gpt-4-turbo       | 0.72  | 18    |
| Medium | gpt-4-turbo       | 0.54  | 18    |
| Hard   | gpt-4-turbo       | 0.41  | 18    |
| Avg    | gpt-4-turbo       | **0.56** | —   |

*Baseline scores are reproducible with OpenAI API. Requires OPENAI_API_KEY.*

---

## 🐳 Docker Deployment

### Build & Run Locally
```bash
docker build -t email-openenv .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY="sk-..." \
  -e MODEL_NAME="gpt-4-turbo-preview" \
  email-openenv
```

### Test API
```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "action": {"action_type": "reply", "content": "..."}}'
```

### Deploy to HuggingFace Spaces
1. Create a new Space at https://huggingface.co/new-space
2. Select "Docker" runtime
3. Push this repository to the Space
4. The Space will automatically build and deploy the environment
5. Access the API at `https://your-space.hf.space`

---

## 📁 Project Structure
```
email-openenv/
├── env/
│   ├── __init__.py
│   ├── environment.py       # EmailEnv class (step/reset/state)
│   ├── models.py            # Observation, Action, Reward (Pydantic)
│   ├── tasks.py             # EasyTask, MediumTask, HardTask
│   └── grader.py            # Hard task grader
├── app.py                   # FastAPI server for HF Spaces
├── inference.py             # Baseline agent + structured logging
├── openenv.yaml             # OpenEnv specification
├── Dockerfile               # Container image
├── requirements.txt         # Dependencies
├── README.md                # This file
└── data/
    └── emails.json          # Dataset (generated/loaded at runtime)
```

---

## 🧪 Validation Checklist

Before submission, ensure:

- [ ] `openenv validate` passes
- [ ] `docker build .` succeeds
- [ ] `docker run` responds to `/reset` on port 7860
- [ ] `inference.py` runs without errors
- [ ] Baseline scores are reproducible
- [ ] All 3 tasks (easy, medium, hard) are implemented
- [ ] Graders return scores in [0.0, 1.0]
- [ ] HuggingFace Space is deployed and responds

Run the validation script:
```bash
bash scripts/validate-submission.sh https://your-space.hf.space
```

---

## 📝 Logging Format

The inference script emits structured JSON logs for automated evaluation:

```json
{"type": "START", "timestamp": "2024-01-15T10:30:00Z", "task": "easy", "environment": "email-openenv", "model": "gpt-4-turbo-preview"}
{"type": "STEP", "step": 1, "action": "{\"action_type\": \"reply\"}", "reward": 0.5, "done": false}
{"type": "STEP", "step": 2, "action": "{\"action_type\": \"archive\"}", "reward": 1.0, "done": false}
{"type": "END", "timestamp": "2024-01-15T10:32:00Z", "success": true, "steps_taken": 18, "final_score": 0.72, "total_reward": 13.8}
```

---

## 🔐 Environment Variables

Required for baseline inference:

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `API_BASE_URL` | LLM API endpoint | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier | `gpt-4-turbo-preview` |
| `HF_TOKEN` | Hugging Face token (alternative to OPENAI_API_KEY) | `hf_...` |

---

## 📚 References

- **OpenEnv Spec**: https://github.com/openenv-community/openenv-core
- **Enron Dataset**: https://www.cs.cmu.edu/~enron/
- **FastAPI**: https://fastapi.tiangolo.com/
- **OpenAI API**: https://platform.openai.com/docs

---

## 📄 License

MIT License — See LICENSE file for details.

---

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

## ✉️ Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

**Last Updated**: January 2024