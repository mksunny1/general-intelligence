# general-intelligence

**general-intelligence** is a framework for **self-organizing, composable knowledge systems**.

Rather than centering intelligence on algorithms or massive datasets, intelligence emerges from the **structure, interaction, and behavior of knowledge** itself. Each knowledge instance can learn, reason, react to new information, contribute to shared outputs, or even operate autonomously.

This is **not a toy**—it supports ML-style reasoning, prompt-response systems, and autonomous agents, all within the same model.

---

## Core Philosophy

* Intelligence arises from **knowledge, not code**.
* Knowledge is **active, composable, and distributed**.
* The system is **flat**—no central scheduler or hard-coded control.
* Knowledge interacts through **context dictionaries** and **collaborative composition**.

Each `Knowledge` instance decides:

* When to respond to a stimulus (`on`)
* When to react to new knowledge (`on_add`)
* When to react to knowledge removal (`on_remove`)
* How to contribute to shared reasoning (`compose`)

---

## Installation

```bash
pip install general-intelligence
```

---

## Quick Start

```python
from gi import GeneralIntelligence, Knowledge
gi = GeneralIntelligence()


class EchoKnowledge(Knowledge):
    def on(self, ctx, gi):
        if hasattr(ctx, "message"):
            return f"Echo: {ctx.message}"

echo = EchoKnowledge()
gi.learn(echo)

class Context:
    def __init__(self, message):
        self.message = message

print(list(gi.on(Context("Hello"))))  # ['Echo: Hello']

```

All knowledge instances share a **single model** and respond only to contexts relevant to them.

---

## ML-Style Additive Knowledge

Demonstrates how `GeneralIntelligence` can handle **tabular, additive learning**:

```python
from gi import GeneralIntelligence, Knowledge
from itertools import combinations

gi = GeneralIntelligence()

class AdditiveKnowledge(Knowledge):
    def __init__(self, n_features):
        self.n_features = n_features
        self.valid_combinations = []

    def on(self, ctx, gi):
        if hasattr(ctx, "row") and hasattr(ctx, "target"):
            row, target = ctx.row, ctx.target
            # First time: cache all single-element combinations
            if not self.valid_combinations:
                all_combs = []
                for r in range(1, self.n_features + 1):
                    all_combs.extend(combinations(range(self.n_features), r))
                self.valid_combinations = [
                    comb for comb in all_combs if sum(row[i] for i in comb) == target
                ]
            # Keep only combinations that continue to hold
            self.valid_combinations = [
                comb for comb in self.valid_combinations
                if sum(row[i] for i in comb) == target
            ]
        elif hasattr(ctx, "row"):
            row = ctx.row
            for comb in self.valid_combinations:
                return sum(row[i] for i in comb)

additive = AdditiveKnowledge(n_features=3)
gi.learn(additive)

# Training
class TrainCtx:
    def __init__(self, row, target):
        self.row = row
        self.target = target

for row, target in [([1,2,3], 3), ([0,3,1], 3)]:
    list(gi.on(TrainCtx(row, target)))

# Prediction
class PredictCtx:
    def __init__(self, row):
        self.row = row

print(next(gi.on(PredictCtx([2,1,0]))))  # Output: sum of matching combination

```

---

## Dialog / Prompt-Response Knowledge

```python
from gi import GeneralIntelligence, Knowledge
gi = GeneralIntelligence()


class DialogKnowledge(Knowledge):
    def __init__(self):
        self.history = []

    def on(self, ctx, gi):
        if hasattr(ctx, "user"):
            self.history.append(ctx.user)
            return f"Bot: I heard '{ctx.user}'"

dialog = DialogKnowledge()
gi.learn(dialog)

class MsgCtx:
    def __init__(self, user):
        self.user = user

for response in gi.on(MsgCtx("Hello")):
    print(response)  # Bot: I heard 'Hello'

```

---

## Autonomous Knowledge Example

```python
import threading, time
from gi import GeneralIntelligence, Knowledge
gi = GeneralIntelligence()

import threading, time


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
timer = TimerKnowledge()
gi.learn(timer)

# Let autonomous timer run a few ticks
time.sleep(5)
gi.unlearn(timer)  # stop autonomous loop

```

---

## Compositional Reasoning

Knowledge can **modify shared context** and collaborate:

```python
from gi import GeneralIntelligence, Knowledge
gi = GeneralIntelligence()


class AccumulateKnowledge(Knowledge):
    def compose(self, ctx, composer, gi):
        if not hasattr(ctx, "accum"):
            ctx.accum = []
        ctx.accum.append("step")

acc = AccumulateKnowledge()
gi.learn(acc)

def final_composer(ctx):
    return getattr(ctx, "accum", [])

class DummyCtx: pass

print(gi.compose(DummyCtx(), final_composer))  # ['step']

```

Multiple knowledge types—ML-style, dialog, autonomous—can coexist in the same model.

---

## Architectural Principles

| Concept               | Description                                                           |
| --------------------- | --------------------------------------------------------------------- |
| Knowledge as agents   | Each instance is autonomous, reactive, and composable                 |
| Structural similarity | ML-style learning can track relationships across inputs               |
| Emergent reasoning    | Intelligence arises from knowledge interactions, not centralized code |
| Composable context    | `compose` allows knowledge to collaborate in flexible ways            |
| Distributed operation | Knowledge can operate independently, including autonomous loops       |

---

## Use Cases

* **Hierarchical or multimodal reasoning systems**
* **Interactive chatbots or agents**
* **Tabular ML tasks and feature discovery**
* **Autonomous monitoring or simulation agents**
* **Hybrid AI systems combining specialized knowledge modules**

---

## Vision

**GeneralIntelligence** shifts AI from **algorithm-driven to knowledge-driven**.

Knowledge is:

* Composable
* Inspectable
* Autonomous
* Extensible across tasks and domains

A single model can host **diverse knowledge types** that cooperate, compete, or ignore irrelevant contexts.

---

## Next Steps

* Specialized ML-style knowledge modules (numeric, logical, temporal)
* Multi-agent reasoning
* Integration with deep learning perception modules
* Community-built knowledge libraries
* Tutorials demonstrating **cross-cutting knowledge interactions**

---

## License

MIT License
Copyright (c) 2025

