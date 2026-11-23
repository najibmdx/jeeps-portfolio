"""Tests for the data script. No internet is used: the download is replaced by a fake."""

import copy
import hashlib

import pytest

import get_data
import settings

CONTENT = b"mint_address,min_ratio\nSAMPLE0000,0.5\n"


@pytest.fixture
def config():
    """The real config, but with the label's size and hash set to a tiny made-up file."""
    cfg = copy.deepcopy(settings.load_config())
    cfg["data"]["label"]["bytes"] = len(CONTENT)
    cfg["data"]["label"]["sha256"] = hashlib.sha256(CONTENT).hexdigest()
    cfg["data"]["feature"]["bytes"] = len(CONTENT)
    cfg["data"]["feature"]["sha256"] = hashlib.sha256(CONTENT).hexdigest()
    return cfg


def fake_download(content):
    calls = []

    def download(url, dest):
        calls.append(url)
        dest.write_bytes(content)

    download.calls = calls
    return download


def test_sha256_matches_the_known_value_for_abc(tmp_path):
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert get_data.sha256_of(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_check_file_names_every_problem(tmp_path):
    sha = hashlib.sha256(CONTENT).hexdigest()
    path = tmp_path / "f"
    assert get_data.check_file(path, len(CONTENT), sha) == "missing"
    path.write_bytes(CONTENT + b"x")
    assert get_data.check_file(path, len(CONTENT), sha) == "wrong_size"
    path.write_bytes(b"y" * len(CONTENT))
    assert get_data.check_file(path, len(CONTENT), sha) == "wrong_hash"
    path.write_bytes(CONTENT)
    assert get_data.check_file(path, len(CONTENT), sha) == "ok"


def test_label_is_downloaded_when_missing(tmp_path, config):
    download = fake_download(CONTENT)
    ok, message = get_data.fetch_label(config, tmp_path, download)
    assert ok and "downloaded" in message
    assert (tmp_path / "label.csv").read_bytes() == CONTENT
    assert not (tmp_path / "label.csv.part").exists()
    assert download.calls == [config["data"]["label"]["url"]]


def test_a_good_existing_label_is_not_downloaded_again(tmp_path, config):
    (tmp_path / "label.csv").write_bytes(CONTENT)
    download = fake_download(b"something else")
    ok, _ = get_data.fetch_label(config, tmp_path, download)
    assert ok and download.calls == []
    assert (tmp_path / "label.csv").read_bytes() == CONTENT


def test_a_wrong_existing_label_is_never_overwritten(tmp_path, config):
    (tmp_path / "label.csv").write_bytes(b"my own edited copy")
    download = fake_download(CONTENT)
    ok, message = get_data.fetch_label(config, tmp_path, download)
    assert not ok and "will not overwrite" in message
    assert download.calls == []
    assert (tmp_path / "label.csv").read_bytes() == b"my own edited copy"


def test_a_download_with_the_wrong_hash_is_thrown_away(tmp_path, config):
    download = fake_download(b"z" * len(CONTENT))
    ok, message = get_data.fetch_label(config, tmp_path, download)
    assert not ok and "thrown away" in message
    assert not (tmp_path / "label.csv").exists()
    assert not (tmp_path / "label.csv.part").exists()


def test_a_network_failure_leaves_nothing_behind(tmp_path, config):
    def broken(url, dest):
        dest.write_bytes(b"half")
        raise OSError("connection reset")

    ok, message = get_data.fetch_label(config, tmp_path, broken)
    assert not ok and "failed" in message
    assert list(tmp_path.iterdir()) == []


def test_feature_file_missing_tells_him_what_to_do(tmp_path, config):
    ok, message = get_data.check_feature(config, tmp_path)
    assert not ok and "not found" in message and "feature.pkl" in message and "MELT" in message


def test_feature_file_with_wrong_hash_says_do_not_load_it(tmp_path, config):
    (tmp_path / "feature.pkl").write_bytes(b"q" * len(CONTENT))
    ok, message = get_data.check_feature(config, tmp_path)
    assert not ok and "Do NOT load it" in message


def test_the_script_never_downloads_the_feature_file(tmp_path, config):
    download = fake_download(CONTENT)
    get_data.main([], config=config, data_dir=tmp_path, download=download)
    assert config["data"]["label"]["url"] in download.calls
    assert len(download.calls) == 1            # only the label
    assert not (tmp_path / "feature.pkl").exists()


def test_main_returns_0_only_when_both_files_are_right(tmp_path, config):
    (tmp_path / "feature.pkl").write_bytes(CONTENT)
    assert get_data.main([], config=config, data_dir=tmp_path, download=fake_download(CONTENT)) == 0


def test_main_returns_1_when_the_feature_file_is_missing(tmp_path, config):
    assert get_data.main([], config=config, data_dir=tmp_path, download=fake_download(CONTENT)) == 1


def test_check_only_does_not_download(tmp_path, config):
    download = fake_download(CONTENT)
    code = get_data.main(["--check-only"], config=config, data_dir=tmp_path, download=download)
    assert code == 1 and download.calls == []
    assert not (tmp_path / "label.csv").exists()
