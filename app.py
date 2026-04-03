"""
FastAPI server for Email OpenEnv on HuggingFace Spaces.
Provides /reset and /step endpoints for the email triage environment.
"""

import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.environment import EmailEnv
from env.tasks import EasyTask, MediumTask, HardTask
from env.models import Action

app = FastAPI(title="Email OpenEnv", version="2.0")

# Global environment instances (one per task)
environments = {
    "easy": EmailEnv(task=EasyTask()),
    "medium": EmailEnv(task=MediumTask()),
    "hard": EmailEnv(task=HardTask()),
}

# Track active episodes
active_session = {"task": None, "env": None}


class ResetRequest(BaseModel):
    task: str = "easy"


class StepRequest(BaseModel):
    action: dict
    task: str = "easy"


class ResetResponse(BaseModel):
    observation: dict
    task: str


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "email-openenv"}


@app.post("/reset", response_model=ResetResponse)
def reset(request: Optional[ResetRequest] = None):
    """Reset the environment."""
    task = request.task if request else "easy"

    if task not in environments:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")

    env = environments[task]
    obs = env.reset()

    active_session["task"] = task
    active_session["env"] = env

    return ResetResponse(
        observation={
            "current_email": obs.current_email,
            "remaining": obs.remaining,
        },
        task=task,
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Take a step in the environment."""
    if active_session["env"] is None:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")

    env = active_session["env"]

    # Validate action
    try:
        action = Action(**request.action)
        obs, reward, done, info = env.step(action.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {str(e)}")

    return StepResponse(
        observation={
            "current_email": obs.current_email,
            "remaining": obs.remaining,
        },
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/tasks")
def list_tasks():
    """List available tasks."""
    return {
        "tasks": list(environments.keys()),
        "descriptions": {
            "easy": "Email classification only",
            "medium": "Classification + response generation",
            "hard": "Full workflow (classification, priority, response quality)",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
