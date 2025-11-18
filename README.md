# **GeneralIntelligence**

### *A composable, multi-knowledge architecture for building real intelligence — not just models.*

---

## ⚡ **What Makes This Library Unique**

### **1. Intelligence as Knowledge, Not Parameters**

You don’t train a giant opaque blob of weights.
You build **explicit knowledge modules** — conceptual units that know when they apply, how they compute, and how to interact with other knowledge.
This turns intelligence into **software** again.

---

### **2. A Flat, Distributed Cognitive Architecture**

No scheduler. No central controller.
Each knowledge class is an autonomous agent:

* It decides when to activate
* It maintains its own memory
* It updates itself through experience
* It collaborates by reading/writing shared context

This makes the system **composable, extensible, and inherently multi-task**.

---

### **3. Multiple Strategies Running in Parallel**

A single GeneralIntelligence model can contain:

* Mathematical hypothesis testers
* Logical relational modules
* Symbolic rules
* Statistical heuristics
* Tree/graph-based reasoning
* Domain-specific knowledge
* Autonomous background agents
* Prompt/dialog knowledge
* Perception plug-ins (e.g., DL model wrappers)

Modules that don’t apply simply **hand off**.
This creates a *parallel hypothesis-testing architecture* where exact solutions are found whenever they exist.

---

### **4. Multi-Strategy ML: Explicit, Testable Hypotheses**

Instead of forcing every dataset into linear models or trees, this architecture allows a single module to test **hundreds or thousands of structured hypotheses**, such as:

* numeric relations
* logical compositions
* hybrid numeric-logical rules
* approximate equalities
* relational constraints
* multi-layer rules discovered via nesting

Each hypothesis tracks its own failures and survives only if within tolerance.
This is **structured conceptual induction**, not blind optimization.

---

### **5. Zero Coupling Between Knowledge Types**

Knowledge modules:

* are self-contained blocks of intelligence
* don’t need to be registered in any config
* don’t break when others change
* can be activated reactively or run autonomously

They interop with the model and other knowledge through lifecycle methods
and shared context enabling powerful cooperation patterns.
This keeps the system **modular, inspectable, and robust**.

---

### **6. Compositional Reasoning Built-In**

Knowledge can participate in compositional flows:

```python
gi.compose(ctx, finalizer)
```

Each module can modify the context during composition, enabling:

* multi-step pipelines
* layered reasoning
* implicit collaboration
* custom “chains of thought”
* tailorable reasoning workflows

This is *structural composition*, not sequential scripting.

---

### **7. Multi-Language by Design**

The architecture uses only:

* classes / objects
* small methods
* shared context objects

Zero reliance on Python-only tricks.
This makes it *trivially portable* to other languages:

* Julia
* R
* Rust
* Go
* C/C++
* TypeScript
* Java/Kotlin
* Swift
* C#
* Zig/Others

The entire ecosystem can be replicated across languages and share conceptual knowledge.

---

### **8. The First General-Purpose “Knowledge Class” Ecosystem**

This library is not just an API.
It defines an *ecosystem pattern* where people can contribute:

* universal hypothesis testers
* symbolic reasoning modules
* numerical/ML hybrids
* perception plug-ins
* planning/goal modules
* domain knowledge packs

This scales intelligence with **contributors rather than compute**.

---

### **9. It’s a Foundation.**

The architecture supports:

* traditional machine learning
* online learning
* multi-subsystem reasoning
* multi-task execution
* hybrid exact + fuzzy learning
* agentic autonomous modules
* continual refinement
* transparent introspection

This is what symbolic AI and deep learning have been trying to achieve.

---

## ⭐ **In Short**

GeneralIntelligence is:

> A modular, distributed, multi-knowledge cognitive engine designed to build real intelligence by composing explicit, testable, autonomous knowledge modules.

It’s small.
It’s simple.
And it’s powerful enough that entire ML workflows, symbolic reasoning processes, dialog systems, and autonomous agents can all live in the *same model* without conflict.

---

# **Examples**


## Simple Addition Knowledge

Demonstrates how `GeneralIntelligence` can learn addition rules in tabular data:

```python
from gi import GeneralIntelligence, Knowledge
from itertools import combinations

gi = GeneralIntelligence()


class AdditionKnowledge(Knowledge):
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


additive = AdditionKnowledge(n_features=3)
gi.learn(additive)


# Training
class TrainCtx:
    def __init__(self, row, target):
        self.row = row
        self.target = target


for row, target in [([1, 2, 3], 3), ([0, 3, 1], 3)]:
    list(gi.on(TrainCtx(row, target)))


# Prediction
class PredictCtx:
    def __init__(self, row):
        self.row = row


print(next(gi.on(PredictCtx([2, 1, 0]))))  # Output: sum of matching combination

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

Understood.
Here is **only the FunctionKnowledge section** — clean, concise, and fully aligned with your architecture (GI + Context + gi.on).

---

## **FunctionKnowledge** (Experimental)

`FunctionKnowledge` is a hypothesis-driven learner that discovers relations between subsets of input features and some RHS value. 
It is plugged into a `GeneralIntelligence` instance, and it participates in learning and prediction through the 
standard `gi.on(Context(...))` interface.

A **hypothesis** has the form:

```
(fn_index, lhs_subset, (rhs_type, rhs_value))
```

Where:

* **fn_index** → index into the provided list of functions
* **lhs_subset** → indices of features used as LHS
* **rhs_type** → `"target" | "feature" | "constant"`
* **rhs_value**

  * None for `"target"`
  * a feature index for `"feature"`
  * literal value for `"constant"`

Each function must support:

```
# training mode
fn(lhs_values, rhs) → bool

# prediction mode
fn(lhs_values) → predicted_rhs | None
```

During training (when `row[target_index]` is not None), FunctionKnowledge checks which hypotheses remain consistent and which fail. Failing hypotheses are allowed to survive up to a configurable tolerance. When the RHS refers to another feature or a constant, a child FunctionKnowledge is spawned to infer deeper dependencies up to `max_depth`.

During prediction (when `row[target_index] is None`), any surviving hypothesis whose function supports inference will yield predicted outputs through GI.

We don't call a `.train()` or `.predict()` on this. We use the same `gi.on()` interface invoking it with an instance of Context exported from the `FunctionKnowledge` module.

---

# **Constructor**

```python
from gi.knowledge.functions  import FunctionKnowledge, Context
FunctionKnowledge(
    functions,          # list of hypothesis functions
    *,
    constants=None,     # constant RHS candidates
    min_lhs=1,
    max_lhs=2,
    tolerance=2,        # how many failures a hypothesis can survive
    max_depth=4,        # nested hypotheses levels
    parent_keys=(),     # prevents cycles in the hierarchy
)
```

---

# **Example Functions**

```python
import statistics as stats

def equals_sum(lhs, rhs=None):
    total = sum(lhs)
    if rhs is None:
        return total
    return total == rhs

def greater_than(lhs, rhs=None):
    m = max(lhs)
    if rhs is None:
        return m
    return m > rhs

def less_than(lhs, rhs=None):
    mn = min(lhs)
    if rhs is None:
        return mn
    return mn < rhs

def equals_median(lhs, rhs=None):
    med = stats.median(lhs)
    if rhs is None:
        return med
    return med == rhs
```

Each function accepts **any number of LHS values** and **RHS of any type**.

---

# **Example: Learning a Sum Rule**

```python
from gi import GeneralIntelligence
from gi.knowledge.functions  import FunctionKnowledge, Context

gi = GeneralIntelligence()

fk = FunctionKnowledge(
    [equals_sum],
    min_lhs=2,
    max_lhs=2,
)

gi.learn(fk)

# Train: third entry is target
gi.on(Context([3, 5, 8], target_index=2))
gi.on(Context([2, 4, 6], target_index=2))
gi.on(Context([10, -1, 9], target_index=2))
```

### Predict

```python
list(gi.on(Context([7, 1, None], target_index=2)))
# → [8]
```

---

# **Example: Mixed Statistical Rules**

```python
from gi import GeneralIntelligence
from gi.knowledge.functions  import FunctionKnowledge, Context
fk = FunctionKnowledge(
    [equals_sum, greater_than, less_than, equals_median],
    constants=[0, 10],
    min_lhs=1,
    max_lhs=3,
)

gi = GeneralIntelligence()
gi.learn(fk)

# Suppose the target is feature 3
train_rows = [
    [2, 3, 5, 5],     # median([2,3,5]) = 3 ≠ 5 → rejected for that hypothesis
    [1, 9, 10, 10],   # max([1,9,10]) = 10 → OK
    [6, 1, 7, 7],     # sum([6,1]) = 7 → OK
]

for r in train_rows:
    gi.on(Context(r, target_index=3))
```

### Predict

```python
list(gi.on(Context([4, 2, 6, None], target_index=3)))
# may yield:
# [6, 7, 10]
```

Multiple hypotheses may fire — GI simply yields them all.

---

# **Example: Feature-to-Feature Reasoning (Child Learners)**

If a hypothesis says:

```
lhs → feature[k]
```

then FK creates a **child FunctionKnowledge** to nest similarly to tree models.

```python
from gi import GeneralIntelligence
from gi.knowledge.functions  import FunctionKnowledge, Context
fk = FunctionKnowledge(
    [equals_sum, equals_median],
    max_depth=3,
)

gi = GeneralIntelligence()
gi.learn(fk)

# Train where target = column 3
gi.on(Context([1, 2, 3, 3], target_index=3))   # median([1,2,3]) = 2 → feature[1]? → child learns it
gi.on(Context([5, 1, 4, 4], target_index=3))   # sum([1,4]) = 5? → etc.
```

### Predict

```python
list(gi.on(Context([7, 1, 2, None], target_index=3)))
# possible cascaded predictions from parent + child hypotheses
```

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

## Use Cases

* **Hierarchical or multimodal reasoning systems**
* **Interactive chatbots or agents**
* **Tabular ML tasks and feature discovery**
* **Autonomous monitoring or simulation agents**
* **Hybrid AI systems combining specialized knowledge modules**

---

## Next Steps

* Specialized knowledge modules
* Community-built knowledge libraries
* Port to other languages
* Tutorials demonstrating **cross-cutting knowledge interactions**

---

## License

MIT License
Copyright (c) 2025

---

