from __future__ import annotations

from typing import Dict

from app.models import EmailRecord, TriageDecision


TASK_WEIGHTS = {
    "easy": {
        "category": 0.60,
        "priority": 0.25,
        "spam_bonus": 0.15,
    },
    "medium": {
        "category": 0.30,
        "priority": 0.20,
        "route_to": 0.30,
        "disposition": 0.20,
    },
    "hard": {
        "category": 0.20,
        "priority": 0.15,
        "route_to": 0.20,
        "disposition": 0.15,
        "escalate": 0.15,
        "response_quality": 0.15,
    },
}


def _norm_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def response_quality_score(pred: str | None, gold: str | None) -> float:
    pred_n = _norm_text(pred or "")
    gold_n = _norm_text(gold or "")
    if not pred_n:
        return 0.0
    if pred_n == gold_n:
        return 1.0

    pred_tokens = set(pred_n.split())
    gold_tokens = set(gold_n.split())
    if not gold_tokens:
        return 0.0

    overlap = len(pred_tokens & gold_tokens) / max(1, len(gold_tokens))
    return min(1.0, overlap)


def grade_decision(task_name: str, email: EmailRecord, decision: TriageDecision) -> Dict[str, float]:
    weights = TASK_WEIGHTS[task_name]
    components: Dict[str, float] = {}

    if "category" in weights:
        components["category"] = weights["category"] if decision.category == email.true_category else 0.0

    if "priority" in weights:
        components["priority"] = weights["priority"] if decision.priority == email.true_priority else 0.0

    if "route_to" in weights:
        components["route_to"] = weights["route_to"] if decision.route_to == email.true_route_to else 0.0

    if "disposition" in weights:
        components["disposition"] = weights["disposition"] if decision.disposition == email.true_disposition else 0.0

    if "escalate" in weights:
        components["escalate"] = weights["escalate"] if decision.escalate == email.requires_escalation else 0.0

    if "spam_bonus" in weights:
        spam_ok = (decision.category == "spam") if email.is_spam else (decision.category != "spam")
        components["spam_bonus"] = weights["spam_bonus"] if spam_ok else 0.0

    if "response_quality" in weights:
        components["response_quality"] = weights["response_quality"] * response_quality_score(
            decision.drafted_response, email.acceptable_response
        )

    total = round(sum(components.values()), 4)
    return {
        "score": min(1.0, max(0.0, total)),
        "components": components,
    }
