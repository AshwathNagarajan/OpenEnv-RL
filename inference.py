import json
import os
from typing import Optional

from dotenv import load_dotenv

from app.agent import heuristic_action, llm_action
from app.env import EmailTriageEnv
from app.models import EmailTriageAction

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "heuristic-baseline")
BENCHMARK = "email_triage_openenv"
CSV_PATH = os.getenv("DATA_PATH", "data/email_triage_synthetic_5000.csv")
TASK_NAMES = [t.strip() for t in os.getenv("EMAIL_TRIAGE_TASKS", "easy,medium,hard").split(",") if t.strip()]
EPISODES_PER_TASK = int(os.getenv("EPISODES_PER_TASK", "1"))
USE_LLM = bool((os.getenv("HF_TOKEN") or os.getenv("API_KEY")) and os.getenv("API_BASE_URL"))


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


def get_agent_action(observation: dict) -> dict:
    if USE_LLM:
        return llm_action(observation, MODEL_NAME)
    return heuristic_action(observation)


def run_episode(task_name: str, seed: int) -> dict:
    env = EmailTriageEnv(csv_path=CSV_PATH, task_name=task_name, seed=seed)
    obs = env.reset().model_dump()
    log_start(task_name, BENCHMARK, MODEL_NAME)
    rewards: list[float] = []
    steps = 0
    success = False
    grader_score = 0.0

    try:
        done = False
        while not done:
            model_action = get_agent_action(obs)
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
                grader_score = info.grader_score
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

    return {"task": task_name, "success": success, "steps": steps, "rewards": rewards, "grader_score": grader_score}


def main():
    episode_index = 0
    for task_name in TASK_NAMES:
        for _ in range(EPISODES_PER_TASK):
            episode_index += 1
            run_episode(task_name, seed=42 + episode_index)


if __name__ == "__main__":
    main()
