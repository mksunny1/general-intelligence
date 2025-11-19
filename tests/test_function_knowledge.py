import statistics as stats
from gi import GeneralIntelligence
from gi.knowledge.functions import FunctionKnowledge, Context, on, make_combinations, make_permutations


# ---------------------------------------------------------------------
# Helper functions for building row-level compute functions
# ---------------------------------------------------------------------

def compare_equals(computed_lhs, rhs):
    return computed_lhs == rhs


def compare_greater(computed_lhs, rhs):
    return computed_lhs > rhs


def compare_less(computed_lhs, rhs):
    return computed_lhs < rhs


# Create row-level functions using the utility wrappers
equals_sum = (make_combinations(lambda s: sum(s), min_size=2, max_size=2), compare_equals)
greater_than = (make_combinations(lambda s: max(s), min_size=1, max_size=3), compare_greater)
less_than = (make_combinations(lambda s: min(s), min_size=1, max_size=3), compare_less)
equals_median = (make_combinations(lambda s: stats.median(s), min_size=1, max_size=3), compare_equals)
equals_difference = (make_permutations(lambda s: s[0] - s[1], min_size=2, max_size=2), compare_equals)


# =====================================================================
# TESTS
# =====================================================================

def test_learns_simple_sum_rule():
    gi = GeneralIntelligence()

    fk = FunctionKnowledge([equals_sum])
    gi.learn(fk)

    # Training: the target = sum of first two features
    on(gi, Context([3, 5, 8], target_index=2))
    on(gi, Context([2, 4, 6], target_index=2))
    on(gi, Context([10, -1, 9], target_index=2))

    pred = on(gi, Context([7, 1, None], target_index=2))
    assert 8 in pred
    print("✓ test_learns_simple_sum_rule passed")


def test_mixed_statistical_rules_yield_multiple_predictions():
    gi = GeneralIntelligence()

    fk = FunctionKnowledge(
        [equals_sum, greater_than, less_than, equals_median],
        constants=[0, 10]
    )
    gi.learn(fk)

    train_rows = [
        [2, 3, 5, 5],
        [1, 9, 10, 10],
        [6, 1, 7, 7],
    ]
    for r in train_rows:
        print(len(fk.hypotheses))
        on(gi, Context(r, target_index=3))

    pred = on(gi, Context([4, 2, 6, None], target_index=3))
    assert all(p in [6, 7, 10] for p in pred)
    assert len(pred) >= 1
    print("✓ test_mixed_statistical_rules_yield_multiple_predictions passed")


def test_child_hypotheses_are_used():
    gi = GeneralIntelligence()

    fk = FunctionKnowledge([equals_sum, equals_median], max_depth=3)
    gi.learn(fk)

    on(gi, Context([1, 2, 3, 3], target_index=3))
    on(gi, Context([5, 1, 4, 4], target_index=3))

    pred = on(gi, Context([7, 1, 2, None], target_index=3))
    assert len(pred) > 0
    print("✓ test_child_hypotheses_are_used passed")


def test_hypothesis_tolerance_allows_some_failures():
    gi = GeneralIntelligence()

    fk = FunctionKnowledge([equals_sum], tolerance=2)
    gi.learn(fk)

    on(gi, Context([1, 2, 3], target_index=2))
    on(gi, Context([3, 4, 7], target_index=2))
    on(gi, Context([5, 5, 999], target_index=2))
    on(gi, Context([2, 2, 999], target_index=2))

    pred = on(gi, Context([10, 5, None], target_index=2))
    assert 15 in pred
    print("✓ test_hypothesis_tolerance_allows_some_failures passed")


def test_target_value_none_is_prediction_mode():
    gi = GeneralIntelligence()

    fk = FunctionKnowledge([equals_sum])
    gi.learn(fk)

    on(gi, Context([2, 3, 5], target_index=2))
    pred = on(gi, Context([10, 1, None], target_index=2))
    assert 11 in pred
    print("✓ test_target_value_none_is_prediction_mode passed")


def test_multiple_values_mode():
    """Test that permutations mode spreads LHS values"""
    gi = GeneralIntelligence()

    fk = FunctionKnowledge([equals_difference])
    gi.learn(fk)

    # Train: target = difference (order matters!)
    # 5 - 2 = 3
    on(gi, Context([5, 2, 3], target_index=2))
    # 8 - 3 = 5
    on(gi, Context([8, 3, 5], target_index=2))

    # Predict: 10 - 3 = 7
    pred = on(gi, Context([10, 3, None], target_index=2))
    assert 7 in pred
    print("✓ test_multiple_values_mode passed")


if __name__ == '__main__':
    test_learns_simple_sum_rule()
    # test_mixed_statistical_rules_yield_multiple_predictions()
    test_child_hypotheses_are_used()
    test_hypothesis_tolerance_allows_some_failures()
    test_target_value_none_is_prediction_mode()
    test_multiple_values_mode()

    print("\nAll tests passed! ✓")