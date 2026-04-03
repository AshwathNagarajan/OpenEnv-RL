from __future__ import annotations

import pandas as pd
from typing import List

from app.models import EmailRecord


def load_dataset(csv_path: str) -> List[EmailRecord]:
    df = pd.read_csv(csv_path)

    # Fill missing text fields with empty strings
    text_columns = [
        "email_id",
        "task_split",
        "from_address",
        "subject",
        "body",
        "received_at",
        "customer_tier",
        "thread_history",
        "true_category",
        "true_priority",
        "true_route_to",
        "true_disposition",
        "acceptable_response",
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Fill missing numeric fields
    if "hours_waiting" in df.columns:
        df["hours_waiting"] = df["hours_waiting"].fillna(0).astype(int)

    # Fill missing boolean fields
    if "requires_escalation" in df.columns:
        df["requires_escalation"] = df["requires_escalation"].fillna(False).astype(bool)

    if "is_spam" in df.columns:
        df["is_spam"] = df["is_spam"].fillna(False).astype(bool)

    records = df.to_dict(orient="records")
    return [EmailRecord(**r) for r in records]


def filter_by_task(records: List[EmailRecord], task_name: str) -> List[EmailRecord]:
    return [r for r in records if r.task_split == task_name]