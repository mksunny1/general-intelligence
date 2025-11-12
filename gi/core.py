import numpy as np


class GeneralIntelligence:
    """
    A self-organizing knowledge system that learns through structural pattern matching.

    Unlike traditional machine learning which requires fixed feature spaces and large datasets,
    GeneralIntelligence works with arbitrary nested data structures and learns from examples
    by memorizing and comparing structural patterns.

    Knowledge instances are active participants that:
    - React when added to the system (on_learned)
    - Respond to new knowledge being learned (on_knowledge)
    - Compose responses to events (via on() triggers)
    - Operate autonomously if marked as active

    Example:
        >>> gi = GeneralIntelligence()
        >>> gi.learn(Knowledge([1, 2, 3]))
        >>> matches = list(gi.identify(Knowledge([2, 3, 4]), threshold=5))
        >>> print(matches[0])  # ([1, 2, 3], 3)

    Attributes:
        knowledge (list): Collection of learned Knowledge instances

    """
    def __init__(self):
        self.knowledge = []

    def learn(self, new_knowledge):
        """
        Learn a new knowledge instance and integrate it into the system.

        The knowledge is informed it's been learned (on_learned), then all existing
        knowledge is notified about the new arrival (on_knowledge). This allows
        knowledge to form relationships and self-organize.

        Args:
            new_knowledge: A Knowledge instance to learn

        Example:
            >>> gi = GeneralIntelligence()
            >>> gi.learn(Knowledge([1, 2, 3]))
            >>> gi.learn(Knowledge([Knowledge([4, 5]), Knowledge([6, 7])]))
        """
        new_knowledge.on_learned(gi=self)

        # Inform all existing knowledge about new arrival
        for existing in self.knowledge:
            existing.on_knowledge(new_knowledge=new_knowledge, gi=self)
        self.knowledge.append(new_knowledge)

    def identify(self, data, threshold=0):
        """
        Find knowledge matching the given data within a similarity threshold.

        Searches through learned knowledge hierarchically - can match exact data
        or patterns nested within learned structures. Yields all matches within
        the threshold as (knowledge, distance) tuples.

        Args:
            data: Data to identify (typically a Knowledge instance or raw value)
            threshold: Maximum difference distance for a match (default: 0 for exact)

        Yields:
            tuple: (matched_knowledge, distance) for each match found

        Example:
            >>> gi = GeneralIntelligence()
            >>> gi.learn(Knowledge([1, 2, 3]))
            >>> for match, dist in gi.identify(Knowledge([1, 2, 3]), threshold=0):
            ...     print(f"Match: {match}, Distance: {dist}")
        """
        for s_data in self.knowledge:
            if isinstance(s_data, Knowledge):
                potential_data, dist = s_data.identify(data, threshold)
                if potential_data is not None:
                    yield potential_data, dist
            elif abs(s_data - data) <= threshold:
                yield s_data, abs(s_data - data)

    def compose(self, context, knowledge_class):
        """
        Compose new knowledge by letting all learned knowledge shape a context.

        Each knowledge instance can contribute to the context through its compose()
        method, allowing collective intelligence to build complex responses.

        Args:
            context: Dictionary that knowledge instances can modify
            knowledge_class: Class to instantiate with the composed context

        Returns:
            Instance of knowledge_class created from the shaped context

        Example:
            >>> context = {'values': []}
            >>> result = gi.compose(context, Knowledge)
        """
        for knowledge in self.knowledge:
            knowledge.compose(context, knowledge_class)
        return knowledge_class(context)

    def on(self, trigger):
        """
        Trigger event-driven responses from knowledge that declares interest.

        Knowledge instances can implement is_response_for(trigger, gi) to declare
        what events they respond to. Matching knowledge instances compose
        responses collectively.

        Args:
            trigger: Event data (typically a dict with event type and parameters)

        Yields:
            Composed responses from knowledge instances that matched the trigger

        Example:
            >>> class AlertKnowledge(Knowledge):
            ...     def is_response_for(cls, trigger, gi):
            ...         return trigger.get('type') == 'alert'
            >>>
            >>> gi.learn(AlertKnowledge([1]))
            >>> for response in gi.on({'type': 'alert', 'level': 'high'}):
            ...     print(response)
        """
        context = dict(trigger=trigger, responses=[])
        for knowledge in self.knowledge:
            if knowledge.is_response_for(trigger, self):
                resp = self.compose(context, type(knowledge))
                context['responses'].append(resp)
                yield resp


class Knowledge:
    """
    Base class for all knowledge instances in the system.

    Knowledge represents learned patterns that can identify similar patterns,
    react to new information, and compose responses. Unlike passive data,
    Knowledge instances actively participate in learning and reasoning.

    Subclass this to create specialized knowledge types with custom:
    - Difference computation (difference method)
    - Event responses (is_response_for classmethod)
    - Autonomous behavior (is_active, start methods)
    - Relationship formation (on_knowledge method)

    Example:
        >>> class SequenceKnowledge(Knowledge):
        ...     def difference(self, data):
        ...         # Custom comparison for sequences
        ...         return compute_sequence_distance(self.values, data.values)
        >>>
        >>> gi = GeneralIntelligence()
        >>> gi.learn(SequenceKnowledge([1, 2, 3, 4]))

    Attributes:
        values: The data this knowledge represents (numbers and/or nested Knowledge)
    """
    def __init__(self, values=None):
        self.values = values

    def __str__(self):
        return str(self.values)

    def __repr__(self):
        return repr(self.values)

    def difference(self, data):
        """
        Compute structural difference between this knowledge and given data.

        Base implementation does element-wise comparison for same-length sequences.
        Override this for custom comparison logic.

        Args:
            data: Data to compare against (Knowledge instance, list, or value)

        Returns:
            float: Distance measure (0 = identical, inf = incompatible, higher = more different)

        Example:
            >>> k1 = Knowledge([1, 2, 3])
            >>> k2 = Knowledge([2, 3, 4])
            >>> print(k1.difference(k2))  # 3
        """
        data_values = data if isinstance(data, (list, tuple)) else data.values if isinstance(data, Knowledge) else [data]
        if len(data_values) != len(self.values):
            return np.inf
        return sum(difference(v1, v2) for v1, v2 in zip(self.values, data_values))

    def identify(self, data, threshold=0):
        """
        Find this knowledge or nested knowledge matching the data.

        First checks if this knowledge matches. If not, recursively searches
        nested Knowledge instances. Enables hierarchical pattern matching.

        Args:
            data: Data to identify
            threshold: Maximum difference for a match

        Returns:
            tuple: (matched_knowledge, distance) or (None, 0) if no match

        Example:
            >>> nested = Knowledge([Knowledge([1, 2]), Knowledge([3, 4])])
            >>> match, dist = nested.identify(Knowledge([1, 2]))
            >>> print(match)  # [1, 2]
        """
        diff = self.difference(data)
        if diff <= threshold:
            return self, diff
        for value in self.values:
            if isinstance(value, Knowledge):
                potential_result, dist = value.identify(data, threshold)
                if potential_result is not None:
                    return potential_result, dist
        return None, 0

    def compose(self, context, knowledge_class):
        """
        Contribute to composing new knowledge from a context.

        Override this to have your knowledge shape collective responses.
        Modify the context dict to influence the final composed result.

        Args:
            context: Shared context dict that all knowledge can modify
            knowledge_class: The class being composed

        Example:
            >>> def compose(self, context, knowledge_class):
            ...     context['confidence'] = self.compute_confidence()
        """
        pass

    def on_learned(self, gi):
        """
        Called when this knowledge is learned by a GeneralIntelligence system.

        Override to react to being added to a knowledge base. If is_active()
        returns True, start() will be called to begin autonomous operation.

        Args:
            gi: The GeneralIntelligence instance that learned this knowledge

        Example:
            >>> def on_learned(self, gi):
            ...     print(f"Learned by system with {len(gi.knowledge)} total knowledge")
            ...     self.register_with_related_knowledge(gi)
        """
        if self.is_active():
            self.start(gi)

    def on_knowledge(self, new_knowledge, gi):
        """
        Called when new knowledge is learned by a system this knowledge belongs to.

        Override to form relationships, update internal state, or react to
        related knowledge being learned.

        Args:
            new_knowledge: The newly learned Knowledge instance
            gi: The GeneralIntelligence system where this occurred

        Example:
            >>> def on_knowledge(self, new_knowledge, gi):
            ...     if isinstance(new_knowledge, CompatibleKnowledge):
            ...         self.related_knowledge.append(new_knowledge)
        """
        pass

    def is_response_for(self, trigger, gi):
        """
        Declare whether this knowledge type responds to a given trigger.

        Override to make your knowledge react to specific events via gi.on(trigger).

        Args:
            trigger: Event data (typically a dict)
            gi: The GeneralIntelligence instance

        Returns:
            bool: True if this knowledge should compose a response

        Example:
            >>> def is_response_for(cls, trigger, gi):
            ...     return trigger.get('type') == 'pattern_query'
        """
        return False

    def is_active(self):
        """
        Whether this knowledge should operate autonomously.

        Returns:
            bool: True to have start() called when learned

        Example:
            >>> def is_active(self):
            ...     return True  # Begin monitoring when learned
        """
        return False

    def start(self, gi):
        """
        Begin autonomous operation (called if is_active() returns True).

        Override to implement self-directed behavior like monitoring,
        periodic updates, or proactive reasoning.

        Args:
            gi: The GeneralIntelligence instance this knowledge belongs to

        Example:
            >>> def start(self, gi):
            ...     self.monitor_thread = threading.Thread(target=self.monitor, args=(gi,))
            ...     self.monitor_thread.start()
        """
        pass


def difference(value1, value2):
    if isinstance(value1, Knowledge):
        return value1.difference(value2)
    elif isinstance(value2, Knowledge):
        return value2.difference(value1)
    else:
        return abs(value1 - value2)

