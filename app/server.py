from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException, Query

from app.env import EmailTriageEnv
from app.models import EmailTriageAction

app = FastAPI(title="Email Triage OpenEnv")
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = "data/email_triage_synthetic_5000.csv"
VALID_TASKS = {"easy", "medium", "hard"}
DEFAULT_SESSION_ID = "default"
_SESSIONS: Dict[str, EmailTriageEnv] = {}


def _create_env(task_name: str = "easy") -> EmailTriageEnv:
    return EmailTriageEnv(csv_path=CSV_PATH, task_name=task_name)


def _get_env(session_id: str = DEFAULT_SESSION_ID) -> EmailTriageEnv:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = _create_env("easy")
    return _SESSIONS[session_id]


@app.get("/")
def root():
    return {
        "status": "ok",
        "env": "email_triage_openenv",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/state", "/step", "/set_task/{task_name}", "/session", "/run"],
        "default_session_id": DEFAULT_SESSION_ID,
    }


@app.post("/session")
def create_session(task_name: str = Query("easy")):
    if task_name not in VALID_TASKS:
        raise HTTPException(status_code=400, detail="task_name must be one of: easy, medium, hard")
    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = _create_env(task_name)
    return {"status": "ok", "session_id": session_id, "task": task_name}


@app.get("/run")
def run_inference(
    tasks: str = Query("easy,medium,hard", description="Comma-separated tasks"),
    episodes: int = Query(2, ge=1, le=10, description="Episodes per task"),
):
    env = os.environ.copy()
    env["EMAIL_TRIAGE_TASKS"] = tasks
    env["EPISODES_PER_TASK"] = str(episodes)

    # Keep the public demo stable even if external LLM credentials are present.
    env.setdefault("FORCE_HEURISTIC", "1")

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
def set_task(task_name: str, session_id: str = Query(DEFAULT_SESSION_ID)):
    if task_name not in VALID_TASKS:
        raise HTTPException(status_code=400, detail="task_name must be one of: easy, medium, hard")
    _SESSIONS[session_id] = _create_env(task_name)
    return {"status": "ok", "task": task_name, "session_id": session_id}


@app.post("/reset")
def reset(session_id: str = Query(DEFAULT_SESSION_ID)):
    return _get_env(session_id).reset().model_dump()


@app.get("/state")
def state(session_id: str = Query(DEFAULT_SESSION_ID)):
    return _get_env(session_id).state().model_dump()


@app.post("/step")
def step(action: EmailTriageAction, session_id: str = Query(DEFAULT_SESSION_ID)):
    obs, reward, done, info = _get_env(session_id).step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info.model_dump(),
    }