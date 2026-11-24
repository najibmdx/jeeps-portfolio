import audit
import sample
import settings


def test_audit_runs_on_made_up_data(tmp_path):
    sample.write_sample(tmp_path)
    a = audit.compute_audit(tmp_path, settings.load_config())
    assert a["rows"]["markets"] > 0
    assert a["markets_with_two_or_more_winners"] == 0
    assert a["outcome_yes_disagreements"] == 0           # the sample is built to agree
    assert a["binary_markets"] > 20
    assert 0 < a["universe"] <= a["open_at_t"] <= a["binary_markets"]
    assert a["contested"] + a["near_decided"] == a["universe"]
    assert sum(a["split_sizes"].values()) == a["universe"]


def test_audit_is_repeatable(tmp_path):
    sample.write_sample(tmp_path)
    c = settings.load_config()
    assert audit.compute_audit(tmp_path, c) == audit.compute_audit(tmp_path, c)


def test_compare_names_every_difference():
    expected = {"universe": 100, "rows": {"events": 5}, "binary_yes_rate": 0.27, "boundaries": ["a", "b"]}
    good = {"universe": 100, "rows": {"events": 5}, "binary_yes_rate": 0.2702, "boundaries": ["a", "b"]}
    assert audit.compare(good, expected) == []
    bad = {"universe": 99, "rows": {"events": 6}, "binary_yes_rate": 0.3, "boundaries": ["a", "c"]}
    problems = audit.compare(bad, expected)
    assert len(problems) == 4 and any("universe: computed 99" in p for p in problems)


def test_compare_flags_a_missing_number():
    assert audit.compare({}, {"universe": 1}) != []


def test_expected_numbers_are_in_the_config():
    exp = settings.load_config()["audit_expected"]
    assert exp["universe"] == 43881 and exp["universe_events"] == 10752
    assert exp["split_sizes"] == {"train": 26173, "validation": 8637, "test": 9071}
    assert (exp["purge_train"], exp["purge_validation"]) == (2276, 1208)
