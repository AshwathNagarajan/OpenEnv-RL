from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from app.data import filter_by_task, load_dataset
from app.graders import grade_decision
from app.models import (
    EmailRecord,
    EmailTriageAction,
    EmailTriageInfo,
    EmailTriageObservation,
    EmailTriageReward,
    TriageDecision,
)

STAGES_BY_TASK = {
    "easy": [
        "set_category",
        "set_priority",
        "submit",
    ],
    "medium": [
        "set_category",
        "set_priority",
        "set_route",
        "set_disposition",
        "submit",
    ],
    "hard": [
        "set_category",
        "set_priority",
        "set_route",
        "set_disposition",
        "set_escalation",
        "draft_response",
        "submit",
    ],
}

VALID_VALUES = {
    "set_category": {
        "technical",
        "feature_request",
        "internal",
        "spam",
        "shipping",
        "security",
        "sales",
        "billing",
    },
    "set_priority": {"low", "medium", "high", "critical"},
    "set_route": {
        "tech_support",
        "product_team",
        "manager",
        "ignore",
        "ops_team",
        "security_team",
        "sales_team",
        "billing_team",
    },
    "set_disposition": {
        "respond",
        "archive",
        "escalate",
        "mark_spam",
        "request_more_info",
        "resolve",
    },
}


class EmailTriageEnv:
    def __init__(self, csv_path: str, task_name: str = "easy", seed: int = 42):
        self.csv_path = csv_path
        self.task_name = task_name
        self.seed = seed
        self.rng = random.Random(seed)

        self.all_records: List[EmailRecord] = load_dataset(csv_path)
        self.records: List[EmailRecord] = filter_by_task(self.all_records, task_name)
        if not self.records:
            raise ValueError(f"No records found for task '{task_name}'")

        self.max_steps = len(STAGES_BY_TASK[task_name]) + 2
        self.current_email: Optional[EmailRecord] = None
        self.decision = TriageDecision()
        self.step_index = 0
        self.done = False
        self.last_action_error: Optional[str] = None
        self.action_history: List[Dict[str, Any]] = []
        self.current_stage_index = 0

    def _current_stage(self) -> str:
        stages = STAGES_BY_TASK[self.task_name]
        return stages[min(self.current_stage_index, len(stages) - 1)]

    def _available_actions(self) -> List[str]:
        return [self._current_stage()]

    def reset(self) -> EmailTriageObservation:
        self.current_email = self.rng.choice(self.records)
        self.decision = TriageDecision()
        self.step_index = 0
        self.done = False
        self.last_action_error = None
        self.action_history = []
        self.current_stage_index = 0
        return self.state()

    def state(self) -> EmailTriageObservation:
        if self.current_email is None:
            raise RuntimeError("Environment not reset")

        return EmailTriageObservation(
            task_name=self.task_name,
            step_index=self.step_index,
            max_steps=self.max_steps,
            current_stage=self._current_stage(),
            email_id=self.current_email.email_id,
            from_address=self.current_email.from_address,
            subject=self.current_email.subject,
            body=self.current_email.body,
            customer_tier=self.current_email.customer_tier,
            hours_waiting=self.current_email.hours_waiting,
            thread_history=self.current_email.thread_history,
            available_actions=self._available_actions(),
            partial_decision=self.decision,
            action_history=self.action_history,
            last_action_error=self.last_action_error,
        )

    def _advance_stage(self) -> None:
        self.current_stage_index += 1

    def _draft_response_reward(self, text: str) -> float:
        text = text.strip()
        if len(text) < 12:
            return -0.03

        reward = 0.03
        if any(token in text.lower() for token in ["hello", "hi", "thank you", "thanks"]):
            reward += 0.01
        if self.current_email and self.current_email.from_address.split("@")[0].lower() in text.lower():
            reward += 0.01

        return min(0.08, reward)

    def _dense_reward_for_correct_field(self, field_name: str, value: Any) -> float:
        e = self.current_email
        assert e is not None

        mapping = {
            "set_category": e.true_category,
            "set_priority": e.true_priority,
            "set_route": e.true_route_to,
            "set_disposition": e.true_disposition,
            "set_escalation": e.requires_escalation,
        }

        if mapping.get(field_name) == value:
            return 0.15

        if field_name == "set_route" and self.decision.category is not None:
            valid_route_by_category = {
                "technical": "tech_support",
                "feature_request": "product_team",
                "internal": "manager",
                "spam": "ignore",
                "shipping": "ops_team",
                "security": "security_team",
                "sales": "sales_team",
                "billing": "billing_team",
            }
            if value == valid_route_by_category.get(self.decision.category):
                return 0.05

        if field_name == "set_escalation" and self.decision.priority == "critical" and value is True:
            return 0.05

        if field_name == "set_disposition":
            if self.decision.category == "spam" and value == "mark_spam":
                return 0.05
            if self.decision.category == "security" and value == "escalate":
                return 0.05

        return -0.05

    def _apply_action(self, action: EmailTriageAction) -> float:
        expected_stage = self._current_stage()
        reward = 0.0

        if action.action_type != expected_stage:
            self.last_action_error = f"expected {expected_stage}, got {action.action_type}"
            return -0.10

        if action.action_type in VALID_VALUES:
            if action.value not in VALID_VALUES[action.action_type]:
                self.last_action_error = f"invalid value for {action.action_type}: {action.value}"
                return -0.10

        self.last_action_error = None

        if action.action_type == "set_category":
            self.decision.category = action.value
            reward += self._dense_reward_for_correct_field("set_category", action.value)

        elif action.action_type == "set_priority":
            self.decision.priority = action.value
            reward += self._dense_reward_for_correct_field("set_priority", action.value)

        elif action.action_type == "set_route":
            self.decision.route_to = action.value
            reward += self._dense_reward_for_correct_field("set_route", action.value)

        elif action.action_type == "set_disposition":
            self.decision.disposition = action.value
            reward += self._dense_reward_for_correct_field("set_disposition", action.value)

        elif action.action_type == "set_escalation":
            self.decision.escalate = bool(action.value)
            reward += self._dense_reward_for_correct_field("set_escalation", bool(action.value))

        elif action.action_type == "draft_response":
            text = str(action.value or "").strip()
            self.decision.drafted_response = text
            reward += self._draft_response_reward(text)

        elif action.action_type == "submit":
            if self.task_name == "easy":
                required_ok = all([
                    self.decision.category is not None,
                    self.decision.priority is not None,
                ])
            elif self.task_name == "medium":
                required_ok = all([
                    self.decision.category is not None,
                    self.decision.priority is not None,
                    self.decision.route_to is not None,
                    self.decision.disposition is not None,
                ])
            else:
                required_ok = all([
                    self.decision.category is not None,
                    self.decision.priority is not None,
                    self.decision.route_to is not None,
                    self.decision.disposition is not None,
                    self.decision.escalate is not None,
                ])

            if not required_ok:
                self.last_action_error = "submit before completing required fields"
                return -0.20

            reward += 0.0

        self.action_history.append(
            {
                "step": self.step_index,
                "action_type": action.action_type,
                "value": action.value,
            }
        )

        self._advance_stage()
        return reward

    def step(
        self, action: EmailTriageAction | Dict[str, Any]
    ) -> Tuple[EmailTriageObservation, EmailTriageReward, bool, EmailTriageInfo]:
        if self.done:
            raise RuntimeError("Episode already done")

        if self.current_email is None:
            raise RuntimeError("Call reset() before step()")

        if isinstance(action, dict):
            action = EmailTriageAction(**action)

        self.step_index += 1
        reward_value = self._apply_action(action)

        if self.step_index >= self.max_steps:
            self.done = True
            self.last_action_error = self.last_action_error or "max_steps_exceeded"

        if action.action_type == "submit":
            self.done = True

        grader = grade_decision(self.task_name, self.current_email, self.decision)
        final_score = grader["score"]

        if self.done and action.action_type == "submit":
            reward_value += final_score
            done_reason = "submitted"
        elif self.done:
            reward_value -= 0.10
            done_reason = "terminated"
        else:
            done_reason = None

        reward = EmailTriageReward(
            reward=max(-1.0, min(1.0, round(reward_value, 4))),
            components=grader["components"],
            normalized_score=final_score,
        )

        info = EmailTriageInfo(
            done_reason=done_reason,
            grader_score=final_score,
            expected={
                "category": self.current_email.true_category,
                "priority": self.current_email.true_priority,
                "route_to": self.current_email.true_route_to,
                "disposition": self.current_email.true_disposition,
                "escalate": self.current_email.requires_escalation,
            },
            predicted=self.decision.model_dump(),
        )

        return self.state(), reward, self.done, info

    def close(self) -> None:
        return