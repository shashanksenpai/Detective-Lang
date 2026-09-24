"""The static mount must serve the UI pages and nothing else (BACKLOG S-1).

Uses a throwaway directory laid out like the project root - pages next to a
database, sources, an upload folder and a .git folder - so the test says
exactly what may and may not be reachable without importing server.py (which
pulls in the whole ML stack).
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ui_static import UIStaticFiles, is_ui_page


@pytest.fixture()
def client(tmp_path):
    (tmp_path / "cases.html").write_text("<h1>cases</h1>")
    (tmp_path / "Workspace.HTML").write_text("<h1>workspace</h1>")
    (tmp_path / "detective.db").write_bytes(b"SQLite format 3\x00 private chats")
    (tmp_path / "server.py").write_text("SECRET = 'source'")
    (tmp_path / "sample_chat.txt").write_text("1/1/25, 9:00 AM - A: hi")
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / ".hidden.html").write_text("hidden")
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / "chat.txt").write_text("private")
    (tmp_path / "uploads" / "nested.html").write_text("nested page")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]")
    app = FastAPI()
    app.mount("/", UIStaticFiles(directory=str(tmp_path), html=True), name="static")
    return TestClient(app)


@pytest.mark.parametrize("url", ["/cases.html", "/Workspace.HTML"])
def test_ui_pages_are_served(client, url):
    resp = client.get(url)
    assert resp.status_code == 200
    assert resp.text.startswith("<h1>")


@pytest.mark.parametrize("url", [
    "/detective.db",            # every imported chat
    "/server.py",               # source
    "/sample_chat.txt",
    "/data.json",
    "/uploads/chat.txt",        # uploaded exports
    "/uploads/nested.html",     # even an .html file, if it is not top-level
    "/uploads/",
    "/.git/config",             # repository history
    "/.hidden.html",
    "/",                        # there is no index page
    "/cases.html/../detective.db",
    "/%2e%2e/detective.db",
    "/..%2fdetective.db",
    "/detective.db%00.html",
])
def test_everything_else_is_refused(client, url):
    resp = client.get(url)
    assert resp.status_code == 404
    assert "private chats" not in resp.text
    assert "SECRET" not in resp.text


@pytest.mark.parametrize("path, expected", [
    ("cases.html", True),
    ("Investigation_Board.HTML", True),
    ("", False),
    (".", False),
    ("..", False),
    (".hidden.html", False),
    ("detective.db", False),
    ("cases.html.bak", False),
    ("sub/cases.html", False),
    ("sub\\cases.html", False),
    ("../cases.html", False),
])
def test_is_ui_page(path, expected):
    assert is_ui_page(path) is expected
