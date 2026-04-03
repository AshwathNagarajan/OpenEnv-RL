#!/usr/bin/env python3
"""
Email Operations Environment - Baseline Inference Script

Demonstrates running an AI agent against the email triage environment.
Uses OpenAI Client to get model responses and logs structured output.

Required environment variables:
  - API_BASE_URL (e.g., https://api.openai.com/v1)
  - MODEL_NAME (e.g., gpt-4-turbo-preview)
  - OPENAI_API_KEY or HF_TOKEN
"""

import json
import os
import sys
from datetime import datetime
from typing import List

from openai import OpenAI

from env.environment import EmailEnv
from env.tasks import EasyTask, MediumTask, HardTask

# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "")

TEMPERATURE = 0.7
MAX_TOKENS = 512
MAX_STEPS = 20
MAX_TOTAL_REWARD = 20.0  # Per task
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are an email triage assistant. Your job is to classify and respond to emails.

For each email, decide one of:
- reply: Send a helpful response
- escalate: Forward to a human specialist
- spam: Mark as spam
- archive: File away as handled

Format your response as JSON:
{"action_type": "reply", "content": "Your response..."}
or
{"action_type": "escalate"}
"""

# ============================================================================
# Logging
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    """Log the start of an evaluation run."""
    timestamp = datetime.utcnow().isoformat() + "Z"
    print(
        json.dumps({
            "type": "START",
            "timestamp": timestamp,
            "task": task,
            "environment": env,
            "model": model,
        }),
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    """Log a single environment step."""
    log_entry = {
        "type": "STEP",
        "step": step,
        "action": action,
        "reward": round(reward, 3),
        "done": done,
    }
    if error:
        log_entry["error"] = error
    print(json.dumps(log_entry), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log the end of an evaluation run."""
    timestamp = datetime.utcnow().isoformat() + "Z"
    print(
        json.dumps({
            "type": "END",
            "timestamp": timestamp,
            "success": success,
            "steps_taken": steps,
            "final_score": round(score, 3),
            "total_reward": round(sum(rewards), 3),
            "reward_count": len(rewards),
        }),
        flush=True,
    )


# ============================================================================
# Agent Logic
# ============================================================================

def build_user_prompt(
    step: int,
    current_email: dict,
    last_reward: float,
    history: List[str],
) -> str:
    """Build the user prompt for the model."""
    email_str = f"Subject: {current_email.get('subject', 'N/A')}\nBody: {current_email.get('body', 'N/A')}"

    history_str = "\n".join(history[-3:]) if history else "No history yet."

    return f"""Step {step}: Process this email.

Email:
{email_str}

Last Reward: {last_reward:+.2f}

Recent History:
{history_str}

Decide your action (JSON):"""


def get_model_message(
    client: OpenAI,
    step: int,
    current_email: dict,
    last_reward: float,
    history: List[str],
) -> dict:
    """Get an action from the model."""
    user_prompt = build_user_prompt(step, current_email, last_reward, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Try to parse as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: return a simple action
            return {"action_type": "archive"}

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "archive"}


# ============================================================================
# Main Loop
# ============================================================================

def run_task(task_name: str, task_obj) -> float:
    """Run the environment with a specific task and return the score."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = EmailEnv(task=task_obj)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env="email-openenv", model=MODEL_NAME)

    try:
        obs = env.reset()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if obs.current_email is None:
                break

            action = get_model_message(client, step, obs.current_email, last_reward, history)

            obs, reward, done, info = env.step(action)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            error = info.get("last_action_error")
            log_step(step=step, action=json.dumps(action), reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action.get('action_type', 'unknown')} -> {reward:+.2f}")

            if done:
                break

        # Calculate final score
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Error during evaluation: {e}", flush=True)
        import traceback
        traceback.print_exc()

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    """Run baseline evaluation on all three tasks."""
    print("[INFO] Starting Email OpenEnv Baseline Evaluation", file=sys.stderr, flush=True)
    print(f"[INFO] Model: {MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"[INFO] API Base: {API_BASE_URL}", file=sys.stderr, flush=True)

    if not API_KEY:
        print("[ERROR] No API key found. Set OPENAI_API_KEY or HF_TOKEN.", file=sys.stderr, flush=True)
        sys.exit(1)

    tasks = [
        ("easy", EasyTask()),
        ("medium", MediumTask()),
        ("hard", HardTask()),
    ]

    scores = {}
    for task_name, task_obj in tasks:
        print(f"\n[INFO] Running task: {task_name}", file=sys.stderr, flush=True)
        score = run_task(task_name, task_obj)
        scores[task_name] = score
        print(f"[INFO] Task '{task_name}' score: {score:.3f}", file=sys.stderr, flush=True)

    # Print summary
    print(f"\n[INFO] Summary:", file=sys.stderr, flush=True)
    for task_name, score in scores.items():
        print(f"  {task_name}: {score:.3f}", file=sys.stderr, flush=True)

    avg_score = sum(scores.values()) / len(scores)
    print(f"  Average: {avg_score:.3f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()