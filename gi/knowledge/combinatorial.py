from itertools import combinations
from gi import Knowledge


class Context:
    def __init__(self, row, target_index=-1):
        self.row = row
        self.target_index = target_index


class CombinatorialKnowledge(Knowledge):
    """
    A hypothesis-based combinatorial learner.

    Each hypothesis is uniquely defined by:
        (fn_index, lhs_subset, (rhs_type, rhs_value))

    where rhs_type ∈ {"target", "feature", "constant"}.

    • Constants come from the constructor and are NEVER added as row features.
    • Failures are tracked PER hypothesis; exceeding tolerance deletes it.
    • Children are created for non-target RHS matches, using max_depth - 1.
    • Prediction returns ALL matches (caller may aggregate externally).
    """

    def __init__(
        self,
        functions,
        *,
        constants=None,
        min_lhs=1,
        max_lhs=2,
        tolerance=5,
        max_depth=1,
        parent_key=None,
    ):
        self.functions = functions
        self.constants = constants or []
        self.min_lhs = min_lhs
        self.max_lhs = max_lhs
        self.tolerance = tolerance
        self.max_depth = max_depth
        self.parent_key = parent_key  # used only to prevent identical self-nesting

        # Mapping: hyp_key → {"fail": int, "child": None or CombinatorialKnowledge}
        self.hypotheses = {}

    # ----------------------------------------------------------------------
    # Utility: produce all RHS candidates (target, feature, constant)
    # ----------------------------------------------------------------------
    def _rhs_candidates(self, row, target_index):
        # Target RHS
        yield ("target", None)

        # Feature RHS
        for i, v in enumerate(row):
            if i != target_index:
                yield ("feature", i)

        # Constant RHS
        for c in self.constants:
            yield ("constant", c)

    # ----------------------------------------------------------------------
    # Utility: enumerate all possible hypotheses for a given row
    # ----------------------------------------------------------------------
    def _enumerate_hypotheses(self, row, target_index):
        n = len(row)

        indices = range(n)

        for subset_size in range(self.min_lhs, self.max_lhs + 1):
            for lhs_subset in combinations(indices, subset_size):
                lhs_subset = tuple(lhs_subset)
                lhs_values = tuple(row[i] for i in lhs_subset)

                for fn_index, fn in enumerate(self.functions):
                    for rhs_type, rhs_val in self._rhs_candidates(row, target_index):
                        key = (fn_index, lhs_subset, (rhs_type, rhs_val))
                        yield key, fn, lhs_subset, lhs_values, rhs_type, rhs_val

    # ----------------------------------------------------------------------
    # Training on a single example
    # ----------------------------------------------------------------------
    def on(self, ctx, gi):
        """
        Add a training example.
        row: list of feature values + target at row[target_index]
        """
        if not isinstance(ctx, Context):
            return None
        row = ctx.row
        target_index = ctx.target_index

        new_hypotheses = {}

        for (
            key, fn, lhs_subset, lhs_values, rhs_type, rhs_val
        ) in self._enumerate_hypotheses(row, target_index):

            # Evaluate function
            if rhs_type == "target":
                # Training-time rule: fn(lhs_values, target_value) → True/False
                y = row[target_index]
                try:
                    passed = fn(lhs_values, y)
                except Exception:
                    passed = False
            elif rhs_type == "feature":
                try:
                    passed = fn(lhs_values, row[rhs_val])
                except Exception:
                    passed = False
            else:  # constant
                try:
                    passed = fn(lhs_values, rhs_val)
                except Exception:
                    passed = False

            # If passed, keep it or insert it
            if passed:
                h = self.hypotheses.get(key)
                if h is None:
                    h = {"fail": 0, "child": None}
                else:
                    # reset fail? No: fail persists, but success doesn't change it.
                    pass
                new_hypotheses[key] = h

        # All old but not renewed hypotheses "fail"
        for key, h in self.hypotheses.items():
            if key not in new_hypotheses:
                h["fail"] += 1
                if h["fail"] <= self.tolerance:
                    new_hypotheses[key] = h
                # else: drop entirely

        self.hypotheses = new_hypotheses

        # After updating, build children where needed
        self._update_children(row, target_index)
        return None

    # ----------------------------------------------------------------------
    # Build/update child learners for feature & constant hypotheses
    # ----------------------------------------------------------------------
    def _update_children(self, row, target_index):
        if self.max_depth <= 0:
            return  # no children allowed

        for key, h in self.hypotheses.items():
            fn_index, lhs_subset, (rhs_type, rhs_val) = key
            if rhs_type == "target":
                continue  # target rules do NOT nest

            # Prevent self-nesting loops using exact same key
            if key == self.parent_key:
                continue

            # Evaluate again (safe; row is tiny)
            lhs_values = tuple(row[i] for i in lhs_subset)
            fn = self.functions[fn_index]
            if rhs_type == "feature":
                rhs_value = row[rhs_val]
            else:
                rhs_value = rhs_val

            try:
                passed = fn(lhs_values, rhs_value)
            except Exception:
                passed = False

            if not passed:
                continue  # child triggers only on success

            # Create child if absent
            if h["child"] is None:
                h["child"] = CombinatorialKnowledge(
                    self.functions,
                    constants=self.constants,
                    min_lhs=self.min_lhs,
                    max_lhs=self.max_lhs,
                    tolerance=self.tolerance,
                    max_depth=self.max_depth - 1,       # <-- you required this
                    parent_key=key,
                )

            # Feed same example to child
            h["child"].on(Context(row, target_index))

    # ----------------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------------
    def predict(self, row):
        """
        Returns ALL predictions from matching hypotheses and children.
        Caller aggregates externally.
        """
        out = []

        for key, h in self.hypotheses.items():
            fn_index, lhs_subset, (rhs_type, rhs_val) = key
            fn = self.functions[fn_index]

            lhs_values = tuple(row[i] for i in lhs_subset)

            if rhs_type == "target":
                # prediction: fn(lhs_values) → yhat or None
                try:
                    yhat = fn(lhs_values)
                except Exception:
                    yhat = None
                if yhat is not None:
                    out.append(yhat)

            else:
                # feature or constant rule: test relation and recurse to child
                if rhs_type == "feature":
                    rhs_value = row[rhs_val]
                else:
                    rhs_value = rhs_val

                try:
                    passed = fn(lhs_values, rhs_value)
                except Exception:
                    passed = False

                if passed and h["child"] is not None:
                    out.extend(h["child"].predict(row))

        return out
