from itertools import combinations
from gi import Knowledge, GeneralIntelligence


class Context:
    def __init__(self, row, target_index=-1):
        self.row = row
        self.target_index = target_index


def on(gi: GeneralIntelligence, context: Context):
    result = []
    for result1 in gi.on(context):
        for result2 in result1:
            result.append(result2)
    return result


class FunctionKnowledge(Knowledge):
    """
    A hypothesis-driven learner.

    A hypothesis is a tuple:
        (fn_index, lhs_subset, (rhs_type, rhs_value))

    Where the function signature is:

        • Training:    fn(lhs_values, rhs) → bool
        • Prediction:  fn(lhs_values)      → predicted_rhs
                       fn(lhs_values, rhs) → bool

    If RHS is omitted, fn returns the inferred RHS.
    If RHS is supplied, fn returns True/False.

    RHS types:
        - target   → row[target_index]
        - feature  → row[rhs_value]
        - constant → rhs_value
    """

    def __init__(
        self,
        functions,
        *,
        constants=None,
        min_lhs=1,
        max_lhs=2,
        tolerance=2,
        max_depth=4,
        parent_keys=(),
    ):
        self.functions = functions
        self.constants = constants or []
        self.min_lhs = min_lhs
        self.max_lhs = max_lhs
        self.tolerance = tolerance
        self.max_depth = max_depth
        self.parent_keys = parent_keys

        # hyp_key → {"fail": int, "child": FunctionKnowledge or None}
        self.hypotheses = {}

    # ----------------------------------------------------------------------
    # RHS candidates generator
    # ----------------------------------------------------------------------
    def _rhs_candidates(self, row, target_index):
        yield "target", None
        for i, _ in enumerate(row):
            if i != target_index:
                yield "feature", i
        for c in self.constants:
            yield "constant", c

    # ----------------------------------------------------------------------
    # enumerate hypotheses
    # ----------------------------------------------------------------------
    def _enumerate(self, row, target_index):
        n = len(row)

        for size in range(self.min_lhs, self.max_lhs + 1):
            for lhs_subset in combinations(range(n), size):
                lhs_subset = tuple(lhs_subset)
                lhs_values = tuple(row[i] for i in lhs_subset)

                for fn_index, fn in enumerate(self.functions):
                    for rhs_type, rhs_val in self._rhs_candidates(row, target_index):
                        key = (fn_index, lhs_subset, (rhs_type, rhs_val))
                        yield key, fn, lhs_subset, lhs_values, rhs_type, rhs_val

    # ----------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------
    def on(self, ctx, gi):
        # print(repr(ctx))
        if not isinstance(ctx, Context):
            return None
        # print(self.hypotheses)
        row = ctx.row
        target_index = ctx.target_index

        # print(row, target_index)

        # row[target_index] = None = prediction mode
        if row[target_index] is None:
            return self.predict(row, gi)

        new_h = {}
        # print(row, target_index)

        for key, fn, lhs_subset, lhs_values, rhs_type, rhs_val in self._enumerate(
            row, target_index
        ):
            # resolve RHS
            if rhs_type == "target":
                rhs = row[target_index]
            elif rhs_type == "feature":
                rhs = row[rhs_val]
            else:
                rhs = rhs_val

            # Training: fn(lhs_values, rhs) → bool
            try:
                ok = fn(lhs_values, rhs)
            except Exception:
                ok = False

            # print(lhs_values, rhs, ok)
            if ok:
                # print('ok!')
                h = self.hypotheses.get(key, {"fail": 0, "child": None})
                new_h[key] = h

        # Old hypotheses failing
        for key, h in self.hypotheses.items():
            if key not in new_h:
                h["fail"] += 1
                if h["fail"] <= self.tolerance:
                    new_h[key] = h

        self.hypotheses = new_h

        # child learners for non-target
        self._children_update(row, target_index, gi)
        return None

    # ----------------------------------------------------------------------
    # Child update
    # ----------------------------------------------------------------------
    def _children_update(self, row, target_index, gi):
        if self.max_depth <= 0:
            return

        for key, h in self.hypotheses.items():
            fn_index, lhs_subset, (rhs_type, rhs_val) = key

            if rhs_type == "target":
                continue
            if key in self.parent_keys:
                continue

            if h["child"] is None:
                h["child"] = FunctionKnowledge(
                    self.functions,
                    constants=self.constants,
                    min_lhs=self.min_lhs,
                    max_lhs=self.max_lhs,
                    tolerance=self.tolerance,
                    max_depth=self.max_depth - 1,
                    parent_keys=(*self.parent_keys, key),
                )

            h["child"].on(Context(row, target_index), gi)

    # ----------------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------------
    def predict(self, row, gi):
        for key, h in self.hypotheses.items():
            fn_index, lhs_subset, (rhs_type, rhs_val) = key
            fn = self.functions[fn_index]
            lhs_values = tuple(row[i] for i in lhs_subset)

            if rhs_type == "target":
                # prediction: fn(lhs_values) → yhat
                try:
                    yhat = fn(lhs_values)
                except Exception:
                    yhat = None
                if yhat is not None:
                    yield yhat

            else:
                # feature / constant hypothesis:
                rhs = row[rhs_val] if rhs_type == "feature" else rhs_val
                try:
                    ok = fn(lhs_values, rhs)
                except Exception:
                    ok = False

                if ok and h["child"] is not None:
                    for output in h["child"].predict(row, gi):
                        yield output

