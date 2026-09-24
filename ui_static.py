"""Static hosting for the UI pages only (BACKLOG S-1).

server.py used to mount the whole project directory as static files, so a
running server handed out detective.db (every imported chat), uploads/, the
Python source and .git/ to anything that could reach the port. The UI is a
handful of self-contained top-level .html pages (no local css/js/images), so
this serves exactly that and refuses everything else with a 404.

Light on purpose (no ML imports): test_ui_static.py exercises it in well under
a second.
"""
import os

from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles


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


class UIStaticFiles(StaticFiles):
    """StaticFiles that only ever serves top-level .html pages (see is_ui_page)."""

    async def get_response(self, path, scope):
        if not is_ui_page(path):
            raise HTTPException(status_code=404)
        return await super().get_response(path, scope)
