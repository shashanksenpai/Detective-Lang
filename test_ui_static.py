"""The static mount must serve the UI pages and static/ assets, and nothing else
(BACKLOG S-1). Uses a throwaway directory laid out like the project root - pages
next to a database, sources, an upload folder and a .git folder - so the test says
exactly what may and may not be reachable without importing server.py (which
pulls in the whole ML stack).
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ui_static import UIStaticFiles, is_public_path, is_ui_asset, is_ui_page


@pytest.fixture()
def client(tmp_path):
    (tmp_path / "cases.html").write_text("<h1>cases</h1>")
    (tmp_path / "Workspace.HTML").write_text("<h1>workspace</h1>")
    (tmp_path / "detective.db").write_bytes(b"SQLite format 3\x00 private chats")
    (tmp_path / "server.py").write_text("SECRET = 'source'")
    (tmp_path / "sample_chat.txt").write_text("1/1/25, 9:00 AM - A: hi")
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "stray.css").write_text("body{}")                 # assets live in static/, not the root
    (tmp_path / ".hidden.html").write_text("hidden")
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / "chat.txt").write_text("private")
    (tmp_path / "uploads" / "nested.html").write_text("nested page")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]")
    static = tmp_path / "static"
    static.mkdir()
    (static / "theme.css").write_text(":root{--x:1}")
    (static / "shell.js").write_text("void 0;")
    (static / "logo.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    (static / "secret.py").write_text("SECRET = 'source'")
    (static / "data.json").write_text("{}")
    (static / ".hidden.css").write_text("hidden")
    (static / "sub").mkdir()
    (static / "sub" / "deep.css").write_text("deep")
    app = FastAPI()
    app.mount("/", UIStaticFiles(directory=str(tmp_path), html=True), name="static")
    return TestClient(app)


@pytest.mark.parametrize("url", ["/cases.html", "/Workspace.HTML"])
def test_ui_pages_are_served(client, url):
    resp = client.get(url)
    assert resp.status_code == 200
    assert resp.text.startswith("<h1>")


@pytest.mark.parametrize("url, content_type", [
    ("/static/theme.css", "text/css"),
    ("/static/shell.js", "javascript"),        # text/javascript; a browser may refuse a script served as text/plain
    ("/static/logo.svg", "image/svg+xml"),
])
def test_static_assets_are_served_with_the_right_type(client, url, content_type):
    resp = client.get(url)
    assert resp.status_code == 200
    assert content_type in resp.headers["content-type"]


@pytest.mark.parametrize("url", [
    "/detective.db",            # every imported chat
    "/server.py",               # source
    "/sample_chat.txt",
    "/data.json",
    "/stray.css",               # an asset extension is not enough outside static/
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
    # inside static/ only the listed asset types, one level deep, visible names
    "/static/",
    "/static/secret.py",
    "/static/data.json",
    "/static/sub/deep.css",
    "/static/.hidden.css",
    "/static/../detective.db",
    "/static/%2e%2e/detective.db",
    "/static/theme.css/../../detective.db",
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
    ("static/theme.css", False),         # an asset is not a page
])
def test_is_ui_page(path, expected):
    assert is_ui_page(path) is expected


@pytest.mark.parametrize("path, expected", [
    ("static/theme.css", True),
    ("static\\theme.css", True),         # Windows-normalised
    ("static/shell.JS", True),
    ("static/logo.svg", True),
    ("static/font.woff2", True),
    ("theme.css", False),
    ("static", False),
    ("static/", False),
    ("static/.hidden.css", False),
    ("static/data.json", False),
    ("static/run.py", False),
    ("static/sub/deep.css", False),
    ("static\\sub\\deep.css", False),
    ("static/../theme.css", False),
    ("Static/theme.css", False),         # the directory name is exact
    ("uploads/theme.css", False),
])
def test_is_ui_asset(path, expected):
    assert is_ui_asset(path) is expected


def test_a_public_path_is_a_page_or_an_asset_and_nothing_else():
    assert is_public_path("cases.html") and is_public_path("static/theme.css")
    assert not is_public_path("detective.db") and not is_public_path("static/x.py")
