from .grader import grade_action

class EasyTask:
    def evaluate(self, email, action, history):
        # only classification matters
        if action["action_type"] == email["label"]:
            return 1.0
        return 0.0


class MediumTask:
    def evaluate(self, email, action, history):
        score = 0

        if action["action_type"] == email["label"]:
            score += 0.5

        content = (action.get("content") or "").lower()

        if len(content) > 15:
            score += 0.3

        if "sorry" in content:
            score += 0.2

        return min(score, 1.0)


class HardTask:
    def evaluate(self, email, action, history):
        return grade_action(email, action, history)