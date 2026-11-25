import shutil

import pipeline
import sample
import settings


def test_sample_run_succeeds(capsys):
    assert pipeline.main(["--sample"]) == 0
    out = capsys.readouterr().out
    assert "SAMPLE MODE" in out
    assert "The test split was not opened by this run." in out


def test_sample_run_writes_nothing_into_the_project(tmp_path):
    before = sorted(p.name for p in settings.PROJECT_DIR.iterdir())
    pipeline.main(["--sample"])
    assert sorted(p.name for p in settings.PROJECT_DIR.iterdir()) == before


def test_missing_file_gives_exit_2(tmp_path, capsys):
    assert pipeline.main(["--data-dir", str(tmp_path)]) == 2
    assert "Missing data files" in capsys.readouterr().out


def test_wrong_size_gives_exit_1(tmp_path, capsys):
    sample.write_sample(tmp_path)  # right layout, but not the real sizes
    assert pipeline.main(["--data-dir", str(tmp_path)]) == 1
    assert "expected 7581274" in capsys.readouterr().out


def test_wrong_layout_gives_exit_1(tmp_path, capsys):
    sample.write_sample(tmp_path)
    shutil.copy(tmp_path / "markets.parquet", tmp_path / "events.parquet")
    assert pipeline.main(["--data-dir", str(tmp_path)]) == 1
    assert "PROBLEM events" in capsys.readouterr().out


def test_gitignore_keeps_data_and_outputs_out():
    """Data and results stay out of the repo; the one exception is the report of the one-time test (Stage 10).
    tests/test_check_readme.py asks git itself which paths are ignored."""
    lines = (settings.PROJECT_DIR / ".gitignore").read_text().splitlines()
    for line in ("data/", ".venv/", "outputs/*", "!outputs/final_test_k7.json"):
        assert line in lines
    assert lines.index("outputs/*") < lines.index("!outputs/final_test_k7.json")   # the exception must come after the rule it exempts
    assert [ln for ln in lines if ln.startswith("!")] == ["!outputs/", "!outputs/final_test_k7.json"]   # and it is the only one


def test_sample_run_builds_the_dataset(capsys):
    assert pipeline.main(["--sample"]) == 0
    out = capsys.readouterr().out
    assert "Built the dataset for k=7" in out
