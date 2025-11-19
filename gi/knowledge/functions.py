from itertools import combinations, permutations
from gi import Knowledge, GeneralIntelligence


class Context:
    def __init__(self, row, target_index=-1, lhs_cache=None):
        self.row = row
        self.target_index = target_index
        # lhs_cache is a list of (values_list, compare_fn) tuples
        self.lhs_cache = lhs_cache


def on(gi: GeneralIntelligence, context: Context):
    result = []
    for result1 in gi.on(context):
        for result2 in result1:
            result.append(result2)
    return result


# ============================================================================
# UTILITY FUNCTIONS FOR CREATING ROW-LEVEL COMPUTE FUNCTIONS
# ============================================================================

def make_combinations(compute_fn, min_size, max_size):
    """
    Wrapper that creates a row-level function from a subset-level compute function.

    Args:
        compute_fn: Function that takes a subset tuple and returns a single value
        min_size: Minimum subset size
        max_size: Maximum subset size

    Returns:
        Function that takes a row and returns list of computed values for all combinations

    Example:
        compute_sum = lambda subset: sum(subset)
        row_sum = make_combinations(compute_sum, min_size=2, max_size=3)
        # row_sum([1, 2, 3]) returns [sum of all 2-combos, sum of all 3-combos]
    """

    def row_compute(row):
        results = []
        for size in range(min_size, max_size + 1):
            for combo in combinations(range(len(row)), size):
                subset_values = tuple(row[i] for i in combo)
                try:
                    results.append(compute_fn(subset_values))
                except Exception:
                    results.append(None)
        return results

    return row_compute


def make_permutations(compute_fn, min_size, max_size):
    """
    Wrapper that creates a row-level function from a subset-level compute function.
    Uses permutations instead of combinations (order matters).

    Args:
        compute_fn: Function that takes a subset tuple and returns a single value
        min_size: Minimum subset size
        max_size: Maximum subset size

    Returns:
        Function that takes a row and returns list of computed values for all permutations

    Example:
        compute_diff = lambda subset: subset[0] - subset[1]
        row_diff = make_permutations(compute_diff, min_size=2, max_size=2)
        # row_diff([5, 3]) returns [5-3, 3-5] = [2, -2]
    """

    def row_compute(row):
        results = []
        for size in range(min_size, max_size + 1):
            for perm in permutations(range(len(row)), size):
                subset_values = tuple(row[i] for i in perm)
                try:
                    results.append(compute_fn(subset_values))
                except Exception:
                    results.append(None)
        return results

    return row_compute


# ============================================================================
# MAIN KNOWLEDGE CLASS
# ============================================================================

class FunctionKnowledge(Knowledge):
    """
    A hypothesis-driven learner.

    A hypothesis is a tuple:
        (fn_index, value_index, (rhs_type, rhs_value))

    Where:
        • functions = [(row_compute_fn, compare_fn), ...]
        • row_compute_fn(row) → [value1, value2, ...] (list of values for all subsets)
        • compare_fn(computed_value, rhs) → bool (training)

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
            tolerance=2,
            max_depth=4,
            parent_keys=(),
            lhs_cache=None,
    ):
        self.functions = functions  # List of (row_compute_fn, compare_fn) tuples
        self.constants = constants or []
        self.tolerance = tolerance
        self.max_depth = max_depth
        self.parent_keys = parent_keys

        # hyp_key → {"fail": int, "child": FunctionKnowledge or None}
        self.hypotheses = {}
        self.count = 0

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
    # Build LHS cache - compute everything once
    # ----------------------------------------------------------------------
    def _build_lhs_cache(self, row):
        """Build cache of (values_list, compare_fn) tuples - one per function"""
        cache = []

        for row_compute_fn, compare_fn in self.functions:
            try:
                values_list = row_compute_fn(row)
            except Exception:
                values_list = []

            cache.append((values_list, compare_fn))

        return cache

    # ----------------------------------------------------------------------
    # enumerate hypotheses using LHS cache
    # ----------------------------------------------------------------------
    def _enumerate(self, row, target_index, lhs_cache):
        for fn_index, (values_list, compare_fn) in enumerate(lhs_cache):
            for value_index, computed_value in enumerate(values_list):
                if computed_value is None:
                    continue

                for rhs_type, rhs_val in self._rhs_candidates(row, target_index):
                    key = (fn_index, value_index, (rhs_type, rhs_val))
                    yield key, compare_fn, computed_value

    # ----------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------
    def on(self, ctx, gi):
        if not isinstance(ctx, Context):
            return None

        row = ctx.row
        target_index = ctx.target_index

        # Build lhs_cache if not provided (only at top level)
        if ctx.lhs_cache is None:
            lhs_cache = self._build_lhs_cache(row)
        else:
            lhs_cache = ctx.lhs_cache

        # (row[target_index] == None) => prediction mode
        if row[target_index] is None:
            return self.predict(row, lhs_cache, gi)

        new_h = {}
        combs = self._enumerate(row, target_index, lhs_cache)
        # print(len(combs), self.max_depth, self.count)
        self.count += 1

        for key, compare_fn, computed_lhs in combs:
            rhs_type, rhs_val = key[2]

            # resolve RHS
            if rhs_type == "target":
                rhs = row[target_index]
            elif rhs_type == "feature":
                rhs = row[rhs_val]
            else:
                rhs = rhs_val

            # Training: compare_fn(computed_lhs, rhs) → bool
            try:
                ok = compare_fn(computed_lhs, rhs)
            except Exception:
                ok = False

            if ok:
                h = self.hypotheses.get(key, {"fail": 0, "child": None})
                new_h[key] = h

        # Old hypotheses failing
        for key, h in self.hypotheses.items():
            if key not in new_h:
                h["fail"] += 1
                if h["fail"] <= self.tolerance:
                    new_h[key] = h

        self.hypotheses = new_h

        # child learners for non-target - pass the SAME lhs_cache
        self._children_update(row, target_index, lhs_cache, gi)
        return None

    # ----------------------------------------------------------------------
    # Child update
    # ----------------------------------------------------------------------
    def _children_update(self, row, target_index, lhs_cache, gi):
        if self.max_depth <= 0:
            return

        for key, h in self.hypotheses.items():
            fn_index, value_index, (rhs_type, rhs_val) = key

            if rhs_type == "target":
                continue
            if key in self.parent_keys:
                continue

            if h["child"] is None:
                h["child"] = FunctionKnowledge(
                    self.functions,
                    constants=self.constants,
                    tolerance=self.tolerance,
                    max_depth=self.max_depth - 1,
                    parent_keys=(*self.parent_keys, key)
                )

            # Pass the SAME lhs_cache to child
            h["child"].on(Context(row, target_index, lhs_cache), gi)

    # ----------------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------------
    def predict(self, row, lhs_cache, gi):
        for fn_index, (values_list, compare_fn) in enumerate(lhs_cache):
            for value_index, computed_value in enumerate(values_list):
                if computed_value is None:
                    continue

                for rhs_type, rhs_val in self._rhs_candidates(row, len(row) - 1):
                    key = (fn_index, value_index, (rhs_type, rhs_val))

                    # Check if this hypothesis exists
                    if key not in self.hypotheses:
                        continue

                    h = self.hypotheses[key]

                    if rhs_type == "target":
                        # prediction: use computed_value as the prediction
                        yield computed_value

                    else:
                        # feature / constant hypothesis:
                        rhs = row[rhs_val] if rhs_type == "feature" else rhs_val
                        try:
                            ok = compare_fn(computed_value, rhs)
                        except Exception:
                            ok = False

                        if ok and h["child"] is not None:
                            # Pass the SAME lhs_cache to child
                            for output in h["child"].predict(row, lhs_cache, gi):
                                yield output
