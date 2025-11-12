from gi.core import GeneralIntelligence, Knowledge

gi = GeneralIntelligence()

# Basic pattern matching
gi.learn(Knowledge([1, 2, 3]))
print("Exact match:", *gi.identify(Knowledge([1, 2, 3]), threshold=0))
print("Close match:", *gi.identify(Knowledge([2, 3, 4]), threshold=5))
print("No match:", *gi.identify(Knowledge([10, 20, 30]), threshold=5))

# Hierarchical matching
gi.learn(Knowledge([Knowledge([1, 2]), Knowledge([3, 4])]))
print("\nHierarchical match:", *gi.identify(Knowledge([1, 2]), threshold=0))
print("Nested component:", *gi.identify(Knowledge([3, 4]), threshold=0))


# Custom knowledge type - Alphabet encoding
class AlphabetKnowledge(Knowledge):
    def __init__(self, letters):
        # Convert a-z to 1-26
        values = [ord(c.lower()) - ord('a') + 1 for c in letters]
        super().__init__(values)


gi.learn(AlphabetKnowledge("abc"))
print("\nAlphabet match:", *gi.identify(AlphabetKnowledge("abc"), threshold=0))
print("Similar pattern:", *gi.identify(AlphabetKnowledge("bcd"), threshold=5))


# Reactive knowledge
class PatternKnowledge(Knowledge):
    def __init__(self, values):
        super().__init__(values)
        self.related_patterns = []

    def on_knowledge(self, new_knowledge, gi):
        if isinstance(new_knowledge, PatternKnowledge):
            # Simple similarity check for demo
            if len(self.values) == len(new_knowledge.values):
                self.related_patterns.append(new_knowledge)
                print(f"  → Pattern {self.values} connected to {new_knowledge.values}")


print("\nReactive knowledge test:")
gi.learn(PatternKnowledge([1, 2, 3]))
gi.learn(PatternKnowledge([4, 5, 6]))  # Triggers connection
gi.learn(PatternKnowledge([7, 8]))  # Different length, no connection


# Event-driven responses
class AlertKnowledge(Knowledge):
    @classmethod
    def is_response_for(cls, trigger, gi):
        return trigger.get('type') == 'alert'

    def compose(self, context, knowledge_class):
        context['alert_level'] = context['trigger'].get('level', 'unknown')


gi.learn(AlertKnowledge([99]))
print("\nEvent responses:")
for response in gi.on({'type': 'alert', 'level': 'high'}):
    print(f"  Alert response: {response}")

print("\nNon-matching trigger:")
responses = list(gi.on({'type': 'other'}))
print(f"  Responses: {len(responses)}")