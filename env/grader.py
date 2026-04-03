def grade_action(email, action, history):
    score = 0.0

    # 1. Classification accuracy
    if action["action_type"] == email["label"]:
        score += 0.4
    else:
        score -= 0.3

    # 2. Priority correctness
    if email["priority"] == "high" and action["action_type"] == "escalate":
        score += 0.3
    elif email["priority"] == "low" and action["action_type"] == "escalate":
        score -= 0.2

    # 3. Response quality
    content = (action.get("content") or "").lower()

    if len(content) > 25:
        score += 0.1

    if any(word in content for word in ["sorry", "apologize"]):
        score += 0.1

    if any(word in content for word in ["resolve", "help", "assist"]):
        score += 0.1

    # 4. Penalty for lazy repetition
    if len(history) > 3:
        last = [h["action"]["action_type"] for h in history[-3:]]
        if len(set(last)) == 1:
            score -= 0.3

    # 5. Bonus for correct difficult decisions
    if email["label"] == "escalate" and email["priority"] == "high":
        if action["action_type"] == "escalate":
            score += 0.2

    return max(min(score, 1.0), -1.0)