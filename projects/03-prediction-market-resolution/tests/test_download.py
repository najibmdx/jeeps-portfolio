import functools
import hashlib
import http.server
import threading

import pytest

import download
import settings


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "served"
    root.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_good_download(server, tmp_path):
    root, url = server
    data = b"x" * 3_000_000
    (root / "f.bin").write_bytes(data)
    dest = tmp_path / "out" / "f.bin"
    download.download_file(f"{url}/f.bin", dest, len(data), hashlib.sha256(data).hexdigest(), say=lambda *_: None)
    assert dest.read_bytes() == data
    assert not dest.with_name("f.bin.part").exists()


def test_wrong_hash_is_refused_and_nothing_is_kept(server, tmp_path):
    root, url = server
    data = b"x" * 1000
    (root / "f.bin").write_bytes(data)
    dest = tmp_path / "f.bin"
    with pytest.raises(ValueError, match="sha256"):
        download.download_file(f"{url}/f.bin", dest, len(data), "0" * 64, say=lambda *_: None)
    assert not dest.exists() and not dest.with_name("f.bin.part").exists()


def test_wrong_size_is_refused(server, tmp_path):
    root, url = server
    (root / "f.bin").write_bytes(b"x" * 1000)
    with pytest.raises(ValueError, match="bytes"):
        download.download_file(f"{url}/f.bin", tmp_path / "f.bin", 999, None, say=lambda *_: None)


def test_file_problem(tmp_path):
    p = tmp_path / "a.bin"
    assert download.file_problem(p, 3, None) == "not there"
    p.write_bytes(b"abc")
    assert download.file_problem(p, 3, hashlib.sha256(b"abc").hexdigest()) is None
    assert "expected 4" in download.file_problem(p, 4, None)
    assert "sha256" in download.file_problem(p, 3, "1" * 64)


def test_main_skips_files_that_already_pass(tmp_path, capsys, monkeypatch):
    config = settings.load_config()
    monkeypatch.setattr(download, "file_problem", lambda *a, **k: None)
    assert download.main(["--data-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.count("already there") == 4
