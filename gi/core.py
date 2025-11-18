from typing import Union, Any


class GeneralIntelligence:
    """
    A self-organizing knowledge system that grows by accumulating
    independently authored Knowledge instances.

    GeneralIntelligence imposes **no hierarchy and no control center**.
    Each knowledge instance is autonomous and responsible for:

        - reacting to new knowledge (on_knowledge)
        - reacting to stimuli (on)
        - optionally running independently (start/stop)
        - shaping shared output (compose)

    The system is fully distributed: all knowledge interacts through
    shared context dictionaries and through explicit collaboration rules
    each subclass defines for itself.

    Example:
        >>> gi = GeneralIntelligence()
        >>> class Counter(Knowledge):
        ...     def __init__(self):
        ...         self.count = 0
        ...     def on(self, ctx, gi):
        ...         inc = ctx["data"].get("increment")
        ...         if inc:
        ...             self.count += inc
        ...             return self   # yield directly
        >>> c = Counter()
        >>> gi.learn(c)
        >>> for _ in gi.on({"increment": 2}):
        ...     pass
        >>> c.count
        2

    Attributes:
        knowledge (list): The ordered list of Knowledge instances.
                          Earlier entries have higher priority.
    """

    def __init__(self):
        self.knowledge = []

    def learn(self, knowledge):
        """
        Add a new knowledge instance to the system.

        Steps:
            1. Append to the internal list.
            2. Call on_knowledge(new_knowledge, self) on *all* knowledge,
               including the new instance itself.
            3. Start autonomous behavior (start()).

        Args:
            knowledge: A Knowledge instance.

        Returns:
            self
        """
        self.knowledge.append(knowledge)
        for existing in self.knowledge:
            existing.on_add(knowledge, self)
        return self

    def move(self, knowledge, new_index):
        """
        Reorder a knowledge instance.

        Knowledge classes may call this to adjust their own priority.
        """
        self.knowledge.remove(knowledge)
        self.knowledge.insert(new_index, knowledge)

    def unlearn(self, knowledge):
        """
        Remove a knowledge instance and stop its autonomous behavior.
        """
        for existing in self.knowledge:
            existing.on_remove(knowledge, self)
        self.knowledge.remove(knowledge)

    def compose(self, context, composer):
        """
        Run the collaborative composition pipeline.

        Each knowledge instance may modify the context before the final
        composer is called.

        Args:
            context: Shared dictionary.
            composer: Callable taking (context) -> output.

        Returns:
            Any output produced by composer(context).
        """
        for k in self.knowledge:
            k.compose(context, composer, self)
        return composer(context)

    def on(self, context):
        """
        Stimulate the system.

        Knowledge.on() may return:
            - None        → ignore stimulus
            - non-callable → yield directly
            - callable     → treat as composer and return composed result

        Args:
            context: Arbitrary input.

        Yields:
            Responses from knowledge modules.
        """

        for knowledge in self.knowledge:
            result = knowledge.on(context, self)

            if result is not None:
                response = (
                    self.compose(context, result)
                    if callable(result)
                    else result
                )
                yield response


def on(gi: GeneralIntelligence, context: Any):
    return list(gi.on(context))


class Knowledge:
    """
    Base class for all knowledge modules.

    A Knowledge instance is a self-contained behavioral unit. There is
    no required structure—each subclass decides its own internal data,
    parameters, protocols, and interactions.

    Override these methods to implement behavior:

        on_knowledge(new_knowledge, gi)
            Called every time ANY knowledge is added, including itself.

        on(context, gi)
            Handle stimuli.
            Return:
                None          → ignore
                non-callable   → yield directly
                callable       → treated as composer

        compose(context, composer, gi)
            Optional hook to shape the shared context during composition.

        start(gi)
            Begin autonomous behavior (threads, loops, timers, etc.)

        stop(gi)
            End autonomous behavior when removed.

    Example:
        >>> class Echo(Knowledge):
        ...     def on(self, ctx, gi):
        ...         msg = ctx["data"].get("echo")
        ...         if msg:
        ...             self.last = msg
        ...             return msg
        >>> gi = GeneralIntelligence()
        >>> ek = gi.learn(Echo())
        >>> list(gi.on({"echo": "hello"}))
        ['hello']
    """

    def on_add(self, knowledge, gi):
        pass

    def on_remove(self, knowledge, gi):
        pass

    def on(self, context, gi):
        pass

    def compose(self, context, composer, gi):
        pass
