"""Guards on what README.md must keep saying. These do not check numbers (check_readme.py does, on the real outputs)."""

import re

import pytest
import yaml

import settings

README = (settings.PROJECT_DIR / "README.md").read_text(encoding="utf-8")
CONFIG = yaml.safe_load((settings.PROJECT_DIR / "config.yaml").read_text(encoding="utf-8"))


def squash(text):
    return re.sub(r"\s+", " ", text)


@pytest.mark.parametrize("phrase", [
    "risk flag, not a trade signal",
    "no claim about trading, returns or profit",
    "A learning project",
    "CC BY-NC 4.0",
    "arXiv 2602.13480",
    "## Numbers that could mislead",
    "## Assumptions and limits",
    "## AI tools used",
    "CI cannot check the numbers below",
    "The validation pile was used for choices",
    "cannot separate",
])
def test_the_readme_keeps_its_honesty_statements(phrase):
    assert phrase in squash(README)


@pytest.mark.parametrize("name", ["decision_log.md", "model_note.md", "recommendation.md", "learning_log.md", "prompt_chat_history.md"])
def test_the_readme_links_every_own_words_file(name):
    documents = README.split("## Documents")[1].split("## AI tools used")[0]
    assert f"- `{name}`:" in documents


def test_the_hashes_and_sizes_in_the_readme_are_the_ones_in_the_config():
    for key in ("feature", "label"):
        entry = CONFIG["data"][key]
        assert entry["sha256"] in README
        assert f"{entry['bytes']:,}" in README


def test_the_readme_makes_no_profit_or_return_promise():
    lowered = README.lower()
    for bad in ("will profit", "guaranteed", "make money", "beat the market", "alpha"):
        assert bad not in lowered


def test_every_command_the_readme_names_exists():
    for script in ("get_data.py", "pipeline.py", "final_test.py", "check_readme.py"):
        assert f"src/{script}" in README
        assert (settings.PROJECT_DIR / "src" / script).exists()
    assert (settings.PROJECT_DIR / "run.sh").exists()
