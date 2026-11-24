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
