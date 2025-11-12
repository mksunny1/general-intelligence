# General Intelligence

A framework for building self-organizing, reactive knowledge systems that learn, identify, and compose patterns through structural similarity.

## Core Concept

Unlike traditional ML which requires fixed feature spaces, General Intelligence works with arbitrary nested data structures. Knowledge instances are active participants - they're informed when learned, react to new knowledge, and can autonomously respond to events.

## Key Features

- **Hierarchical pattern matching**: Find patterns within nested structures
- **Active knowledge**: Learned patterns react to new information and shape future learning
- **Event-driven responses**: Knowledge declares what it responds to and composes answers
- **Autonomous operation**: Knowledge can start monitoring and acting when learned
- **Compositional reasoning**: Combine knowledge from multiple sources
- **Interpretable**: Every comparison is based on explicit structural differences

## Quick Example

```python
from gi import GeneralIntelligence, Knowledge

# Create system and teach it patterns
gi = GeneralIntelligence()
gi.learn(Knowledge([1, 2, 3]))
gi.learn(Knowledge([Knowledge([1,2]), Knowledge([3,4])]))

# Identify similar patterns
matches = gi.identify(Knowledge([2, 3, 4]), threshold=5)
# Finds [1,2,3] with distance 3

# Hierarchical matching
matches = gi.identify(Knowledge([1, 2]), threshold=0)
# Finds [1,2] nested inside learned structure
```

## Event-Driven Responses

```python
class AlertKnowledge(Knowledge):
    @classmethod
    def is_response_for(cls, trigger, gi):
        return trigger.get('type') == 'alert'
    
    def compose(self, context, knowledge_class):
        # Shape the response based on learned patterns
        context['message'] = f"Alert: {context['trigger']}"

gi.learn(AlertKnowledge([...]))

# Trigger responses from matching knowledge
for response in gi.on({'type': 'alert', 'level': 'high'}):
    print(response)
```

## Autonomous Knowledge

```python
class MonitoringKnowledge(Knowledge):
    def is_active(self):
        return True  # Start immediately when learned
    
    def start(self, gi):
        # Begin autonomous operation
        self.monitor(gi)
    
    def on_knowledge(self, new_knowledge, gi):
        # React when new knowledge arrives
        if self.should_alert(new_knowledge):
            self.trigger_alert(gi)
```

## Extending Knowledge

```python
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
    
    def compose(self, context, knowledge_class):
        # Influence composition of new knowledge
        context.update(self.contribute_to(context))
```

## Philosophy

This models how knowledge actually works: not as passive data, but as active patterns that recognize, influence, organize, and respond. Knowledge knows the systems it belongs to, the other knowledge around it, and can act autonomously.

## Use Cases

- Abstract reasoning (ARC challenge)
- Interpretable pattern recognition  
- Event-driven AI systems
- Compositional learning
- Autonomous agents
- Structural transformation discovery

---

**Status**: Early prototype. Core functionality works. Community contributions welcome.

---
