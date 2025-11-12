
# GeneralIntelligence

**GeneralIntelligence** is a minimal but powerful framework for **self-organizing, interactive knowledge systems**.

Instead of relying on fixed algorithms or massive datasets, it treats **knowledge itself as the active core of intelligence**.
Each piece of knowledge can learn, relate, react, and even operate autonomously. The system becomes a living structure of interacting knowledge — not just a database, not just a model.

---

## Core Idea

Traditional AI is algorithm-driven.
This framework is **knowledge-driven**.

* Intelligence here comes from the *structure and interaction of knowledge*.
* Each `Knowledge` instance can recognize patterns, react to new information, and shape context.
* The `GeneralIntelligence` system is just a container — a universe where knowledge organizes itself through relationships.

You can think of it as moving “intelligence” from code into data itself.
Smart data, not smart algorithms.

---

## Example

```python
from gi import GeneralIntelligence, Knowledge
gi = GeneralIntelligence()

# Learn some knowledge
gi.learn(Knowledge([1, 2, 3]))
gi.learn(Knowledge([Knowledge([4, 5]), Knowledge([6, 7])]))

# Identify patterns
matches = list(gi.identify(Knowledge([2, 3, 4]), threshold=5))
print(matches[0])  # ([1, 2, 3], 3)
```

Each `Knowledge` object can contain other `Knowledge` objects, forming nested, compositional structures.
Identification is recursive and structural — the system can find partial or nested matches automatically.

---

## Event-driven Intelligence

Knowledge can declare how it responds to triggers:

```python
from gi import GeneralIntelligence, Knowledge
gi = GeneralIntelligence()
class AlertKnowledge(Knowledge):
    def is_response_for(self, trigger, gi):
        return trigger.get('type') == 'alert'

gi.learn(AlertKnowledge(['⚠️']))
for response in gi.on({'type': 'alert', 'level': 'high'}):
    print(response)
```

This creates **event-driven reasoning** — knowledge responding directly to the world or other knowledge.

---

## Extending Knowledge

Subclass `Knowledge` to define specialized reasoning or modalities.

```python
from gi import Knowledge
class SequenceKnowledge(Knowledge):
    def difference(self, data):
        # Custom sequence distance
        return sum(abs(a - b) for a, b in zip(self.values, data.values))
```

```python
from gi import Knowledge
class PatternKnowledge(Knowledge):
    def __init__(self, values):
        super().__init__(values)
        self.related_patterns = []
    
    def on_knowledge(self, new_knowledge, gi):
        # React when new knowledge is learned
        if isinstance(new_knowledge, PatternKnowledge):
            if self.similar_pattern(new_knowledge):
                self.related_patterns.append(new_knowledge)
                new_knowledge.related_patterns.append(self)
    def similar_pattern(self, new_knowledge):
        return new_knowledge.values == self.values
    
    def compose(self, context, knowledge_class):
        # Influence composition of new knowledge
        context.update(self.contribute_to(context))

    def contribute_to(self, context):
        context['knowledge_to_work_with'].append(self)
```

Knowledge can also:

* react when learned (`on_learned`)
* react to new knowledge (`on_knowledge`)
* contribute to collective reasoning (`compose`)
* operate autonomously (`is_active` + `start`)

---

## Architectural Principles

| Concept                   | Description                                             |
| ------------------------- | ------------------------------------------------------- |
| **Knowledge as agents**   | Each instance can act, relate, and respond              |
| **Structural similarity** | Learning is done through pattern comparison, not labels |
| **Emergent reasoning**    | Intelligence arises from interactions, not code logic   |
| **Composable context**    | Knowledge contributes to shared reasoning contexts      |
| **Autonomous operation**  | Active knowledge can think or evolve independently      |

---

## Use Cases

* Building **hierarchical or multimodal reasoning systems**
* Creating **interactive world models**
* Running **autonomous knowledge agents**
* Integrating with **traditional ML/DL** as specialized knowledge subclasses
* Experimenting with **emergent cognition, continual learning, or reflective AI**

---

## Installation

```bash
pip install general-intelligence
```

*(Package published on PyPI.)*

---

## Quick Start: Minimal Autonomous Knowledge

```python
from gi import GeneralIntelligence, Knowledge
gi = GeneralIntelligence()
class ActiveKnowledge(Knowledge):
    def is_active(self):
        return True
    def start(self, gi):
        print("Thinking on my own...")

gi.learn(ActiveKnowledge("self"))
```

Once learned, the knowledge begins acting independently.

---

## Vision

This project is part of a broader paradigm shift:
**from intelligent algorithms to intelligent knowledge.**

It treats intelligence as a property of structured, interacting knowledge —
a distributed, living network rather than a centralized model.

It’s tiny now, but the system already supports:

* classification,
* prompt-response cycles,
* ongoing internal reasoning,
* and even forms of literal consciousness — all from a simple, composable base.

---

## Next Directions

* Specialized knowledge classes (symbolic, numeric, perceptual, temporal, etc.)
* Multi-agent learning and cooperation
* Integrations with reinforcement learning and deep learning models
* Richer context composition and memory architectures

---

## License

MIT License.
Copyright (c) 2025.
