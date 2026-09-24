"""Static hosting for the UI only (BACKLOG S-1).

server.py used to mount the whole project directory as static files, so a
running server handed out detective.db (every imported chat), uploads/, the
Python source and .git/ to anything that could reach the port. The UI is a
handful of top-level .html pages plus a few shared assets in static/
(theme.css, shell.js, ...); this serves exactly that and refuses everything
else with a 404.

Allowed, and nothing more:
  * a bare top-level *.html file                      cases.html
  * one file directly inside static/ with a UI asset extension
                                                       static/theme.css
Hidden names (.git, .env), any other extension (detective.db, *.py, *.json,
*.txt, ...), deeper paths (static/x/y.css, uploads/...) and traversal are 404.

Light on purpose (no ML imports): test_ui_static.py exercises it in well under
a second.
"""
import mimetypes
import os

from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles

ASSET_DIR = "static"
ASSET_EXTENSIONS = (".css", ".js", ".svg", ".woff2")

# Windows resolves types from the registry, where .js is sometimes text/plain and a
# browser may then refuse to run the script: pin the ones the UI depends on.
for _ext, _type in ((".js", "text/javascript"), (".css", "text/css"),
                    (".svg", "image/svg+xml"), (".woff2", "font/woff2")):
    mimetypes.add_type(_type, _ext)


def is_ui_page(path: str) -> bool:
    """True for a bare top-level *.html file name, e.g. "cases.html".

    `path` is what StaticFiles hands over after normalising the URL, so it is
    separator-normalised for the OS: a nested or traversing path ("sub/x.html",
    "../x.html") has a basename different from itself and is refused, as is
    anything hidden ("." prefix, which covers .git/) or not an .html file
    (detective.db, uploads/, *.py, *.txt, *.json, ...).
    """
    return (
        bool(path)
        and os.path.basename(path) == path
        and not path.startswith(".")
        and path.lower().endswith(".html")
    )


def is_ui_asset(path: str) -> bool:
    """True for exactly `static/<name>` where <name> is a visible file with one of
    ASSET_EXTENSIONS. Both separators are handled (Windows normalises to a backslash);
    any other depth, a hidden name or another extension is refused."""
    parts = path.replace("\\", "/").split("/")
    return (
        len(parts) == 2
        and parts[0] == ASSET_DIR
        and bool(parts[1])
        and not parts[1].startswith(".")
        and parts[1].lower().endswith(ASSET_EXTENSIONS)
    )


def is_public_path(path: str) -> bool:
    return is_ui_page(path) or is_ui_asset(path)


class UIStaticFiles(StaticFiles):
    """StaticFiles that only ever serves the UI pages and static/ assets (see is_public_path)."""

    async def get_response(self, path, scope):
        if not is_public_path(path):
            raise HTTPException(status_code=404)
        return await super().get_response(path, scope)
