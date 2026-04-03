def smart_agent(email):
    text = (email["subject"] + email["body"]).lower()

    # detect urgency
    if any(word in text for word in ["urgent", "critical", "failure", "down"]):
        return {"action_type": "escalate"}

    # detect spam
    if any(word in text for word in ["win", "free", "offer", "click"]):
        return {"action_type": "spam"}

    # detect informational
    if any(word in text for word in ["meeting", "update", "newsletter"]):
        return {"action_type": "archive"}

    # intelligent reply
    return {
        "action_type": "reply",
        "content": "We sincerely apologize for the inconvenience. Our team will review your request and resolve it as soon as possible."
    }