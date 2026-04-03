import json
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from app.env import EmailTriageEnv
from app.models import EmailTriageAction

load_dotenv()

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("EMAIL_TRIAGE_TASK", "hard")
BENCHMARK = "email_triage_openenv"
CSV_PATH = os.getenv("DATA_PATH", "data/email_triage_synthetic_5000.csv")

if not API_KEY:
    raise ValueError("Missing HF_TOKEN in environment or .env file")

if not API_BASE_URL:
    raise ValueError("Missing API_BASE_URL in environment or .env file")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    joined = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={joined}", flush=True)


ALLOWED_VALUES = {
    "set_category": [
        "technical",
        "feature_request",
        "internal",
        "spam",
        "shipping",
        "security",
        "sales",
        "billing",
    ],
    "set_priority": [
        "low",
        "medium",
        "high",
        "critical",
    ],
    "set_route": [
        "tech_support",
        "product_team",
        "manager",
        "ignore",
        "ops_team",
        "security_team",
        "sales_team",
        "billing_team",
    ],
    "set_disposition": [
        "respond",
        "archive",
        "escalate",
        "mark_spam",
        "request_more_info",
        "resolve",
    ],
    "set_escalation": [True, False],
}


def get_model_action(observation: dict) -> dict:
    current_stage = observation["current_stage"]
    allowed = ALLOWED_VALUES.get(current_stage)

    prompt = f"""
You are an email triage agent.
Return ONLY one valid minified JSON object with keys: action_type, value.

Current stage: {current_stage}
Available actions: {json.dumps(observation["available_actions"], ensure_ascii=False)}

Email:
From: {observation["from_address"]}
Subject: {observation["subject"]}
Body: {observation["body"]}
Customer tier: {observation["customer_tier"]}
Hours waiting: {observation["hours_waiting"]}
Thread history: {observation["thread_history"]}

Partial decision so far:
{json.dumps(observation["partial_decision"], ensure_ascii=False)}

Strict rules:
1. Output exactly one JSON object. No markdown. No explanation.
2. action_type must be exactly "{current_stage}".
3. Do not invent labels.
4. If allowed values are provided, value must be exactly one of them.
5. For "submit", return {{"action_type":"submit","value":null}}.
6. For "set_escalation", value must be true or false.
7. For "draft_response", write a concise professional response.
8. Choose the most operationally correct action, not the most polite wording.

Decision guidance:
- "critical" = security breach, suspicious login, unauthorized access, account compromise, payment breach, severe enterprise outage.
- "high" = urgent issue, but not the most severe security/account-compromise level.
- "medium" = normal support issue requiring action.
- "low" = informational, low urgency, or non-blocking.
- Use disposition "escalate" when the issue needs specialist, manager, or security handling.
- Use disposition "mark_spam" only for junk/spam.
- Use disposition "archive" only when no action is needed.
- Use disposition "request_more_info" when key facts are missing.
- Use disposition "respond" when regular support can answer directly.
- Use disposition "resolve" only when the issue is already solved/closed.

Important patterns:
- Unknown device access, suspicious login, account takeover, credential theft, or security alerts usually imply:
  priority = "critical"
  route = "security_team"
  disposition = "escalate"
  escalation = true
""".strip()

    if allowed is not None:
        prompt += "\n\nAllowed values for this stage:\n" + json.dumps(allowed, ensure_ascii=False)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=180,
        messages=[
            {"role": "system", "content": "You must return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )

    text = response.choices[0].message.content.strip()
    return json.loads(text)


def main():
    env = EmailTriageEnv(csv_path=CSV_PATH, task_name=TASK_NAME, seed=42)
    obs = env.reset().model_dump()

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    rewards: list[float] = []
    steps = 0
    success = False

    try:
        done = False
        while not done:
            model_action = get_model_action(obs)
            action = EmailTriageAction(**model_action)

            next_obs, reward, done, info = env.step(action)
            obs = next_obs.model_dump()

            steps += 1
            rewards.append(reward.reward)

            action_str = json.dumps(model_action, ensure_ascii=False, separators=(",", ":"))
            log_step(
                step=steps,
                action=action_str,
                reward=reward.reward,
                done=done,
                error=obs.get("last_action_error"),
            )

            if done:
                print("FINAL GRADER SCORE:", info.grader_score)
                print("EXPECTED:", info.expected)
                print("PREDICTED:", info.predicted)
                success = info.grader_score >= 0.70

    except Exception as e:
        log_step(
            step=steps + 1,
            action='{"action_type":"error","value":null}',
            reward=0.00,
            done=True,
            error=str(e),
        )
    finally:
        env.close()
        log_end(success=success, steps=steps, rewards=rewards)


if __name__ == "__main__":
    main()