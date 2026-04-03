from fastapi import FastAPI
from app.env import EmailTriageEnv
from app.models import EmailTriageAction

app = FastAPI(title="Email Triage OpenEnv")

ENV = EmailTriageEnv(csv_path="data/email_triage_synthetic_5000.csv", task_name="easy")


@app.get("/")
def root():
    return {"status": "ok", "env": "email_triage_openenv"}


@app.post("/set_task/{task_name}")
def set_task(task_name: str):
    global ENV
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
