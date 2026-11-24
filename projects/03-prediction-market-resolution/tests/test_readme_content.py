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
