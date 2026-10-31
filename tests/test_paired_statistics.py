from scripts.paired_statistics import transition_counts


def test_transition_counts_match_mcnemar_cells():
    assert transition_counts([(True, True), (True, False), (False, True), (False, False)]) == {
        "a_stable_correct": 1,
        "b_damage": 1,
        "c_rescue": 1,
        "d_stable_failure": 1,
    }
