"""Structural guard for the Phase 7 UI: every page is on the shared theme and workstation bar,
and every asset a page references exists and is one the static policy will actually serve.
A typo in a path or an asset the policy refuses would silently break a page, and nothing else
in the suite loads the pages. Light (no ML imports). Run: pytest test_ui_pages.py
"""
import re
from pathlib import Path

import pytest

from ui_static import is_ui_asset

ROOT = Path(__file__).parent
PAGES = sorted(p.name for p in ROOT.glob("*.html"))
EXPECTED = {"cases.html", "detective_lang.html", "person.html", "combined_dossier.html",
            "merge_review.html", "investigation_board.html", "workspace.html"}


def _read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_the_pages_are_the_seven_the_ui_has():
    assert set(PAGES) == EXPECTED, f"unexpected page set: {sorted(set(PAGES) ^ EXPECTED)}"


@pytest.mark.parametrize("page", PAGES)
def test_every_page_uses_the_shared_theme_and_shell(page):
    src = _read(page)
    assert 'href="static/theme.css"' in src, "not on the shared theme"
    assert 'src="static/shell.js"' in src, "no workstation bar"
    assert 'name="viewport"' in src, "no viewport meta: it will not be responsive on a phone"
    assert '<main class="wb-main' in src, "content is not inside the workstation layout"
    # the shell script must run before the page's own script, which may call into it
    assert src.index("static/shell.js") < src.index("const API_BASE"), "shell.js must load before the page script"


@pytest.mark.parametrize("page", PAGES)
def test_no_page_carries_the_old_design_or_a_private_copy_of_the_tokens(page):
    src = _read(page)
    assert "Special Elite" not in src, "the old noir typeface is superseded"
    assert "--panel-raised" not in src and "--hairline" not in src, "old token names: use the theme's"
    assert "@import url" not in src, "fonts are loaded once, by static/theme.css"


@pytest.mark.parametrize("page", PAGES)
def test_every_asset_a_page_references_exists_and_is_served(page):
    refs = re.findall(r'(?:href|src)="(static/[^"]+)"', _read(page))
    assert refs, "no static assets referenced"
    for ref in refs:
        assert (ROOT / ref).is_file(), f"{page} references {ref}, which does not exist"
        assert is_ui_asset(ref), f"{page} references {ref}, which the static policy refuses to serve"


def test_the_dossier_pages_share_one_renderer():
    """person.html and combined_dossier.html once carried near-identical copies of the rendering (and of
    its XSS bug). They must use static/dossier.js and must not grow a private renderer again."""
    for page in ("person.html", "combined_dossier.html"):
        src = _read(page)
        assert 'src="static/dossier.js"' in src and 'href="static/dossier.css"' in src
        assert "Dossier.renderProfile(" in src
        for private in ("function renderSentiment", "function renderTraits", "function renderVocab", "function statRows"):
            assert private not in src, f"{page} has its own {private}: use static/dossier.js"


@pytest.mark.parametrize("page", PAGES)
def test_pages_that_render_names_define_or_share_an_escape_helper(page):
    """Sender names, source labels and case names come from uploaded exports. A page that builds markup
    with innerHTML must have esc() (or the shared Dossier.esc) available - the escaping itself is verified
    in a real browser with hostile names, see BACKLOG S-3."""
    src = _read(page)
    if "innerHTML" in src:
        assert ("function esc(" in src) or ("Dossier.esc" in src) or ('src="static/dossier.js"' in src), \
            f"{page} sets innerHTML but has no escape helper"
