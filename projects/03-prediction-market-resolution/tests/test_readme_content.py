from pathlib import Path

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def test_no_claim_sentence():
    assert "No claim about trading use, returns or profit." in README


def test_data_credit():
    assert "Akey" in README and "CC-BY-4.0" in README


def test_no_promises():
    low = README.lower()
    for phrase in ("profitable", "guaranteed", "make money", "beat the market", "alpha"):
        assert phrase not in low


def test_readme_status_matches_the_stage():
    assert "Stage 5" in README and "there are no results to report" in README and "baselines scored on validation" in README
