from pydantic import BaseModel
from typing import Optional

class Observation(BaseModel):
    current_email: Optional[dict]
    remaining: int

class Action(BaseModel):
    action_type: str
    content: Optional[str] = None

class Reward(BaseModel):
    score: float
    reason: str