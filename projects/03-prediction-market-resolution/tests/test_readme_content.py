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
    assert "Stage 9" in README and "no final results to report" in README and "untuned model ladder" in README and "python src/models.py" in README
    assert "python src/tune.py" in README and "python src/freeze.py" in README and "frozen_choice.yaml" in README
    assert "python src/ablate.py" in README and "cannot change the frozen choice" in README
    assert "python src/final_test.py --confirm-open-test" in README and "a second run is refused" in README
