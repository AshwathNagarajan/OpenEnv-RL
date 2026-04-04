from __future__ import annotations

import json
import os
from typing import Any, Dict

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

ALLOWED_VALUES = {
    "set_category": [
        "technical",
        "feature_request",
        "internal",
        "spam",
        "shipping",
        "security",
        "sales",
        "billing",
    ],
    "set_priority": ["low", "medium", "high", "critical"],
    "set_route": [
        "tech_support",
        "product_team",
        "manager",
        "ignore",
        "ops_team",
        "security_team",
        "sales_team",
        "billing_team",
    ],
    "set_disposition": [
        "respond",
        "archive",
        "escalate",
        "mark_spam",
        "request_more_info",
        "resolve",
    ],
    "set_escalation": [True, False],
}


def _text(obs: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(obs.get("subject", "")),
            str(obs.get("body", "")),
            str(obs.get("thread_history", "")),
        ]
    ).lower()


def _infer_category(obs: Dict[str, Any]) -> str:
    text = _text(obs)
    if any(k in text for k in ["unsubscribe", "buy now", "limited offer", "lottery", "crypto giveaway", "free money", "click here"]):
        return "spam"
    if any(k in text for k in ["suspicious login", "unknown device", "unauthorized", "breach", "compromised", "phishing", "security alert", "stolen", "fraud"]):
        return "security"
    if any(k in text for k in ["invoice", "refund", "charged", "billing", "payment failed", "subscription", "credit card"]):
        return "billing"
    if any(k in text for k in ["pricing", "quote", "demo", "purchase", "enterprise plan", "sales", "contract"]):
        return "sales"
    if any(k in text for k in ["delivery", "shipment", "tracking", "package", "courier", "shipping"]):
        return "shipping"
    if any(k in text for k in ["feature request", "would love", "please add", "enhancement", "roadmap", "integrate"]):
        return "feature_request"
    if any(k in text for k in ["meeting", "internal", "pto", "policy", "team update", "manager review"]):
        return "internal"
    return "technical"


def _infer_priority(obs: Dict[str, Any], category: str) -> str:
    text = _text(obs)
    tier = str(obs.get("customer_tier", "free")).lower()
    hours_waiting = int(obs.get("hours_waiting", 0) or 0)

    if category == "security" and any(k in text for k in ["unauthorized", "compromised", "breach", "suspicious login", "stolen"]):
        return "critical"
    if any(k in text for k in ["production down", "service unavailable", "outage", "cannot access", "blocked", "urgent"]):
        return "high" if category != "security" else "critical"
    if tier == "enterprise" and hours_waiting >= 24:
        return "high"
    if category in {"billing", "technical", "shipping", "sales"}:
        return "medium"
    return "low"


def _infer_route(category: str, priority: str) -> str:
    if category == "security":
        return "security_team"
    if category == "billing":
        return "billing_team"
    if category == "sales":
        return "sales_team"
    if category == "feature_request":
        return "product_team"
    if category == "internal" and priority in {"high", "critical"}:
        return "manager"
    if category == "spam":
        return "ignore"
    if category == "shipping":
        return "ops_team"
    return "tech_support"


def _infer_disposition(category: str, priority: str) -> str:
    if category == "spam":
        return "mark_spam"
    if category == "security":
        return "escalate"
    if category == "internal" and priority == "low":
        return "archive"
    return "respond"


def _infer_escalation(category: str, priority: str) -> bool:
    return category == "security" or priority == "critical"


def _draft_response(obs: Dict[str, Any], category: str, disposition: str) -> str:
    name = str(obs.get("from_address", "customer")).split("@")[0]
    if category == "security":
        return f"Hello {name}, we have escalated your case to our security team immediately and will follow up with next steps as soon as possible."
    if disposition == "mark_spam":
        return "No response needed. Message classified as spam and removed from the active queue."
    return f"Hello {name}, thank you for reaching out. We have reviewed your request and routed it to the appropriate team. We will update you shortly."


def heuristic_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    current_stage = observation["current_stage"]
    partial = observation.get("partial_decision", {}) or {}
    category = partial.get("category") or _infer_category(observation)
    priority = partial.get("priority") or _infer_priority(observation, category)
    route = partial.get("route_to") or _infer_route(category, priority)
    disposition = partial.get("disposition") or _infer_disposition(category, priority)
    escalate = partial.get("escalate")
    if escalate is None:
        escalate = _infer_escalation(category, priority)

    if current_stage == "set_category":
        return {"action_type": current_stage, "value": category}
    if current_stage == "set_priority":
        return {"action_type": current_stage, "value": priority}
    if current_stage == "set_route":
        return {"action_type": current_stage, "value": route}
    if current_stage == "set_disposition":
        return {"action_type": current_stage, "value": disposition}
    if current_stage == "set_escalation":
        return {"action_type": current_stage, "value": escalate}
    if current_stage == "draft_response":
        return {"action_type": current_stage, "value": _draft_response(observation, category, disposition)}
    return {"action_type": "submit", "value": None}


def llm_action(observation: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    api_base_url = os.getenv("API_BASE_URL")
    if not api_key or not api_base_url:
        return heuristic_action(observation)

    if OpenAI is None:
        return heuristic_action(observation)

    client = OpenAI(api_key=api_key, base_url=api_base_url)
    current_stage = observation["current_stage"]
    allowed = ALLOWED_VALUES.get(current_stage)

    prompt = f"""
You are an email triage agent.
Return ONLY one valid minified JSON object with keys: action_type, value.

Current stage: {current_stage}
Available actions: {json.dumps(observation['available_actions'], ensure_ascii=False)}

Email:
From: {observation['from_address']}
Subject: {observation['subject']}
Body: {observation['body']}
Customer tier: {observation['customer_tier']}
Hours waiting: {observation['hours_waiting']}
Thread history: {observation['thread_history']}

Partial decision so far:
{json.dumps(observation['partial_decision'], ensure_ascii=False)}

Strict rules:
1. Output exactly one JSON object. No markdown. No explanation.
2. action_type must be exactly "{current_stage}".
3. Do not invent labels.
4. If allowed values are provided, value must be exactly one of them.
5. For "submit", return {{"action_type":"submit","value":null}}.
6. For "set_escalation", value must be true or false.
7. For "draft_response", write a concise professional response.
8. Choose the most operationally correct action.
""".strip()

    if allowed is not None:
        prompt += "\n\nAllowed values for this stage:\n" + json.dumps(allowed, ensure_ascii=False)

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        max_tokens=180,
        messages=[
            {"role": "system", "content": "You must return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    return json.loads(text)
