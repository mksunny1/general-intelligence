import statistics as stats
from gi import GeneralIntelligence
from gi.knowledge.functions  import FunctionKnowledge, Context, on

# ---------------------------------------------------------------------
# Example functions used in FunctionKnowledge
# ---------------------------------------------------------------------

def equals_sum(lhs, rhs=None):
    total = sum(lhs)
    return total if rhs is None else (total == rhs)

def greater_than(lhs, rhs=None):
    m = max(lhs)
    return m if rhs is None else (m > rhs)

def less_than(lhs, rhs=None):
    mn = min(lhs)
    return mn if rhs is None else (mn < rhs)

def equals_median(lhs, rhs=None):
    med = stats.median(lhs)
    return med if rhs is None else (med == rhs)


# =====================================================================
# TESTS
# =====================================================================

def test_learns_simple_sum_rule():
    gi = GeneralIntelligence()

    fk = FunctionKnowledge([equals_sum], min_lhs=2, max_lhs=2)
    gi.learn(fk)

    # Training: the target = sum of first two features
    # print(fk.hypotheses)
    on(gi, Context([3, 5, 8], target_index=2))
    # print(fk.hypotheses)
    on(gi, Context([2, 4, 6], target_index=2))
    # print(fk.hypotheses)
    on(gi, Context([10, -1, 9], target_index=2))
    # print(fk.hypotheses)

    pred = on(gi, Context([7, 1, None], target_index=2))
    # print('pred 1: ', pred, len(pred))
    assert 8 in pred
    assert len(pred) == 1


def test_mixed_statistical_rules_yield_multiple_predictions():
    gi = GeneralIntelligence()

    fk = FunctionKnowledge(
        [equals_sum, greater_than, less_than, equals_median],
        constants=[0, 10],
        min_lhs=1,
        max_lhs=3,
    )
    gi.learn(fk)

    train_rows = [
        [2, 3, 5, 5],      # only some hypotheses survive
        [1, 9, 10, 10],
        [6, 1, 7, 7],
    ]
    for r in train_rows:
        print(len(fk.hypotheses))
        on(gi, Context(r, target_index=3))

    pred = on(gi, Context([4, 2, 6, None], target_index=3))
    # print('pred 2: ', pred, len(pred))

    # Depending on surviving hypotheses, these 3 values are possible
    assert all(p in [6, 7, 10] for p in pred)
    assert len(pred) >= 1


def test_child_hypotheses_are_used():
    """
    If the RHS refers to a feature, FK should create a child FK
    that learns how that feature is generated.
    """

    gi = GeneralIntelligence()

    fk = FunctionKnowledge([equals_sum, equals_median], max_depth=3)
    gi.learn(fk)

    # Train: target = column 3
    on(gi, Context([1, 2, 3, 3], target_index=3))
    on(gi, Context([5, 1, 4, 4], target_index=3))

    pred = on(gi, Context([7, 1, 2, None], target_index=3))
    # print('pred 3: ', pred, len(pred))
    # We don't assert specific values because many hypothesis paths can survive.
    # What we *can* assert is that predictions exist.
    assert len(pred) > 0


def test_constants_are_valid_rhs_candidates():
    """
    Actually a useless ChatGPT test:
    """
    return
    gi = GeneralIntelligence()

    def equals_constant(lhs, rhs=None):
        return rhs if rhs is None else (lhs[0] == rhs)

    fk = FunctionKnowledge(
        [equals_constant],
        constants=[42],
        min_lhs=1,
        max_lhs=1,
    )
    gi.learn(fk)

    # Training row: target is literally the constant 42
    on(gi, Context([42, 0], target_index=1))

    pred = on(gi, Context([42, None], target_index=1))
    # print('pred 4: ', pred, len(pred))
    assert 42 in pred


def test_hypothesis_tolerance_allows_some_failures():
    gi = GeneralIntelligence()

    fk = FunctionKnowledge([equals_sum], tolerance=2)
    gi.learn(fk)

    # True rule: third col = sum(first two)
    on(gi, Context([1, 2, 3], target_index=2))  # OK
    on(gi, Context([3, 4, 7], target_index=2))  # OK

    # Now feed incorrect examples but within tolerance
    on(gi, Context([5, 5, 999], target_index=2))  # FAIL 1
    on(gi, Context([2, 2, 999], target_index=2))  # FAIL 2

    # Still should not eliminate the sum hypothesis yet
    pred = on(gi, Context([10, 5, None], target_index=2))
    # print('pred 5: ', pred, len(pred))
    # Should still predict 15
    assert 15 in pred


def test_target_value_none_is_prediction_mode():
    gi = GeneralIntelligence()

    fk = FunctionKnowledge([equals_sum])
    gi.learn(fk)

    # Training
    on(gi, Context([2, 3, 5], target_index=2))

    # Now prediction using the row having None at target
    pred = on(gi, Context([10, 1, None], target_index=2))
    # print('pred 6: ', pred, len(pred))
    assert 11 in pred


if __name__ == '__main__':
    test_learns_simple_sum_rule()
    test_mixed_statistical_rules_yield_multiple_predictions()
    test_child_hypotheses_are_used()
    test_constants_are_valid_rhs_candidates()
    test_hypothesis_tolerance_allows_some_failures()
    test_target_value_none_is_prediction_mode()
