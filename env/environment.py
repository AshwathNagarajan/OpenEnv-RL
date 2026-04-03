import json
from .models import Observation

VALID_ACTIONS = ["reply", "escalate", "spam", "archive"]


class EmailEnv:
    def __init__(self, task):
        self.task = task

    # -----------------------------
    # Load Dataset
    # -----------------------------
    def load_data(self):
        with open("data/emails.json", encoding="utf-8") as f:
            return json.load(f)

    # -----------------------------
    # Reset Environment
    # -----------------------------
    def reset(self):
        self.emails = self.load_data()
        self.index = 0
        self.done = False

        # Tracking
        self.history = []
        self.total_reward = 0.0
        self.last_error = None

        # Metrics
        self.correct_actions = 0
        self.total_steps = 0

        return self.state()

    # -----------------------------
    # Current State
    # -----------------------------
    def state(self):
        if self.index >= len(self.emails):
            return Observation(current_email=None, remaining=0)

        return Observation(
            current_email=self.emails[self.index],
            remaining=len(self.emails) - self.index
        )

    # -----------------------------
    # Validate Action
    # -----------------------------
    def validate_action(self, action):
        if not isinstance(action, dict):
            return False, "Action must be a dictionary"

        if "action_type" not in action:
            return False, "Missing action_type"

        if action["action_type"] not in VALID_ACTIONS:
            return False, "Invalid action_type"

        return True, None

    # -----------------------------
    # Step Function
    # -----------------------------
    def step(self, action):

        if self.done:
            return self.state(), 0.0, True, {
                "error": "Episode already finished"
            }

        email = self.emails[self.index]
        self.total_steps += 1

        # -----------------------------
        # Validate Action
        # -----------------------------
        valid, error = self.validate_action(action)

        if not valid:
            self.last_error = error
            reward = -1.0

            return self.state(), reward, False, {
                "last_action_error": error
            }

        # -----------------------------
        # Compute Reward
        # -----------------------------
        reward = self.task.evaluate(email, action, self.history)

        # -----------------------------
        # Reward Shaping (REAL WORLD)
        # -----------------------------

        # 1. Delay penalty
        if self.index > 10:
            reward -= 0.05

        # 2. Over-escalation penalty
        if action["action_type"] == "escalate":
            escalate_count = sum(
                1 for h in self.history if h["action"]["action_type"] == "escalate"
            )
            if escalate_count > 3:
                reward -= 0.2

        # 3. Repetition penalty
        if len(self.history) > 2:
            last_actions = [h["action"]["action_type"] for h in self.history[-2:]]
            if all(a == action["action_type"] for a in last_actions):
                reward -= 0.2

        # 4. Bonus for correct action
        if action["action_type"] == email["label"]:
            self.correct_actions += 1

        # Clamp reward
        reward = max(min(reward, 1.0), -1.0)

        # -----------------------------
        # Update State
        # -----------------------------
        self.total_reward += reward

        self.history.append({
            "email_id": email["id"],
            "action": action,
            "reward": reward
        })

        self.index += 1

        if self.index >= len(self.emails):
            self.done = True

        # -----------------------------
        # Info Dictionary (IMPORTANT)
        # -----------------------------
        info = {
            "current_index": self.index,
            "total_reward": round(self.total_reward, 2),
            "accuracy": round(self.correct_actions / max(self.total_steps, 1), 2),
            "last_action_error": self.last_error
        }

        return self.state(), reward, self.done, info

    # -----------------------------
    # Close Environment
    # -----------------------------
    def close(self):
        pass