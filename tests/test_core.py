import pytest
import time
import threading
from gi import GeneralIntelligence, Knowledge
from itertools import combinations

# -----------------------------
# ML-Style Additive Knowledge
# -----------------------------
class AdditiveKnowledge(Knowledge):
    def __init__(self, n_features):
        self.n_features = n_features
        self.valid_combinations = []

    def on(self, ctx, gi):
        if "row" not in ctx:
            return None

        row, target, train = ctx.get("row"), ctx.get("target"), ctx.get("train", False)

        if train:
            if not self.valid_combinations:
                all_combs = []
                for r in range(1, self.n_features + 1):
                    all_combs.extend(combinations(range(self.n_features), r))
                self.valid_combinations = [
                    comb for comb in all_combs if sum(row[i] for i in comb) == target
                ]
            else:
                self.valid_combinations = [
                    comb for comb in self.valid_combinations if sum(row[i] for i in comb) == target
                ]
            return None
        else:
            if not self.valid_combinations:
                return None
            comb = self.valid_combinations[0]
            return sum(row[i] for i in comb)


# -----------------------------
# Dialog Knowledge
# -----------------------------
class DialogKnowledge(Knowledge):
    def __init__(self):
        self.history = []

    def on(self, ctx, gi):
        msg = ctx.get("user")
        if msg:
            self.history.append(msg)
            return f"Bot: I heard '{msg}'"


# -----------------------------
# Autonomous Knowledge
#


# -----------------------------
# Autonomous Knowledge
# -----------------------------
class TimerKnowledge(Knowledge):
    def __init__(self):
        self.count = 0

    def on_add(self, knowledge, gi):
        if self is knowledge:
            self.running = True
            self.thread = threading.Thread(target=self.run, args=(gi,), daemon=True)
            self.thread.start()

    def on_remove(self, knowledge, gi):
        if self is knowledge:
            self.running = False

    def run(self, gi):
        while getattr(self, "running", False):
            print("Tick:", self.count)
            self.count += 1
            time.sleep(1)

class TickCtx: pass


# -----------------------------
# Compositional Reasoning Knowledge
# -----------------------------
class AccumulateKnowledge(Knowledge):
    def compose(self, ctx, composer, gi):
        ctx.setdefault("accum", []).append("step")


# -----------------------------
# Tests
# -----------------------------
def test_additive_knowledge_training_and_prediction():
    gi = GeneralIntelligence()
    additive = AdditiveKnowledge(n_features=3)
    gi.learn(additive)

    training_data = [([1, 2, 3], 3), ([0, 3, 1], 3)]
    for row, target in training_data:
        list(gi.on({"row": row, "target": target, "train": True}))

    prediction = next(gi.on({"row": [2, 1, 0], "train": False}))
    assert prediction == sum([2, 1, 0][:2])  # Any valid combination sum


def test_dialog_knowledge_response():
    gi = GeneralIntelligence()
    dialog = DialogKnowledge()
    gi.learn(dialog)

    responses = list(gi.on({"user": "Hello"}))
    assert responses == ["Bot: I heard 'Hello'"]

    responses = list(gi.on({"user": "How are you?"}))
    assert responses == ["Bot: I heard 'How are you?'"]
    assert dialog.history == ["Hello", "How are you?"]


def test_autonomous_knowledge_runs_and_stops():
    gi = GeneralIntelligence()
    timer = TimerKnowledge()
    assert getattr(timer, "count", -1) == 0
    gi.learn(timer)

    # Give it some time to start
    time.sleep(5)
    gi.unlearn(timer)
    assert getattr(timer, "count", -1) == 5


def test_compositional_reasoning():
    gi = GeneralIntelligence()
    acc = AccumulateKnowledge()
    gi.learn(acc)

    def final_composer(ctx):
        return ctx.get("accum")

    result = gi.compose({}, final_composer)
    assert result == ["step"]


if __name__ == "__main__":
    pytest.main()
