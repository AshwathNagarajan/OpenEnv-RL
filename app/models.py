from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


Category = Literal[
    "technical",
    "feature_request",
    "internal",
    "spam",
    "shipping",
    "security",
    "sales",
    "billing",
]

Priority = Literal["low", "medium", "high", "critical"]

RouteTo = Literal[
    "tech_support",
    "product_team",
    "manager",
    "ignore",
    "ops_team",
    "security_team",
    "sales_team",
    "billing_team",
]

Disposition = Literal[
    "respond",
    "archive",
    "escalate",
    "mark_spam",
    "request_more_info",
    "resolve",
]

ActionType = Literal[
    "set_category",
    "set_priority",
    "set_route",
    "set_disposition",
    "set_escalation",
    "draft_response",
    "submit",
]


class EmailRecord(BaseModel):
    email_id: str
    task_split: Literal["easy", "medium", "hard"]
    from_address: str
    subject: str
    body: str
    received_at: str
    customer_tier: Literal["free", "premium", "enterprise"]
    hours_waiting: int
    thread_history: str = ""
    true_category: Category
    true_priority: Priority
    true_route_to: RouteTo
    true_disposition: Disposition
    acceptable_response: str = ""
    requires_escalation: bool
    is_spam: bool


class TriageDecision(BaseModel):
    category: Optional[Category] = None
    priority: Optional[Priority] = None
    route_to: Optional[RouteTo] = None
    disposition: Optional[Disposition] = None
    escalate: Optional[bool] = None
    drafted_response: Optional[str] = None


class EmailTriageObservation(BaseModel):
    task_name: str
    benchmark: str = "email_triage_openenv"
    step_index: int
    max_steps: int
    current_stage: str
    email_id: str
    from_address: str
    subject: str
    body: str
    customer_tier: str
    hours_waiting: int
    thread_history: str
    available_actions: List[ActionType]
    partial_decision: TriageDecision
    action_history: List[Dict[str, Any]]
    last_action_error: Optional[str] = None


class EmailTriageAction(BaseModel):
    action_type: ActionType
    value: Optional[Any] = None


class EmailTriageReward(BaseModel):
    reward: float = Field(ge=-1.0, le=1.0)
    components: Dict[str, float]
    normalized_score: float = Field(ge=0.0, le=1.0)


class EmailTriageInfo(BaseModel):
    done_reason: Optional[str] = None
    grader_score: float = Field(ge=0.0, le=1.0)
    expected: Dict[str, Any]
    predicted: Dict[str, Any]
