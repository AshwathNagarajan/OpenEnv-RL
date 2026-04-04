from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from app.env import EmailTriageEnv
from app.models import EmailTriageAction

app = FastAPI(title="Email Triage OpenEnv")
BASE_DIR = Path(__file__).resolve().parent.parent
ENV = EmailTriageEnv(csv_path="data/email_triage_synthetic_5000.csv", task_name="easy")


@app.get("/")
def root():
    return {
        "status": "ok",
        "env": "email_triage_openenv",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/state", "/step", "/set_task/{task_name}", "/run"],
    }


@app.get("/run")
def run_inference(
    tasks: str = Query("easy,medium,hard", description="Comma-separated tasks"),
    episodes: int = Query(2, ge=1, le=10, description="Episodes per task"),
):
    env = os.environ.copy()
    env["EMAIL_TRIAGE_TASKS"] = tasks
    env["EPISODES_PER_TASK"] = str(episodes)

    try:
        proc = subprocess.run(
            [sys.executable, "runner.py"],
            cwd=str(BASE_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "runner.py timed out after 300 seconds",
        }


@app.post("/set_task/{task_name}")
def set_task(task_name: str):
    global ENV
    if task_name not in {"easy", "medium", "hard"}:
        raise HTTPException(status_code=400, detail="task_name must be one of: easy, medium, hard")
    ENV = EmailTriageEnv(csv_path="data/email_triage_synthetic_5000.csv", task_name=task_name)
    return {"status": "ok", "task": task_name}


@app.post("/reset")
def reset():
    return ENV.reset().model_dump()


@app.get("/state")
def state():
    return ENV.state().model_dump()


@app.post("/step")
def step(action: EmailTriageAction):
    obs, reward, done, info = ENV.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info.model_dump(),
    }