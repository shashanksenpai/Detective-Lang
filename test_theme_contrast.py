"""Design-token guard for static/theme.css: near-black surfaces with neon accents is
exactly the style where dim technical labels quietly become unreadable, so the contrast
of every colour used as text is checked against every surface (WCAG 2.x ratios).
Light (no ML imports). Run: pytest test_theme_contrast.py
"""
import re
from pathlib import Path

import pytest

CSS = Path(__file__).with_name("static").joinpath("theme.css").read_text(encoding="utf-8")
TOKENS = dict(re.findall(r"--([a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{6})\b", CSS))
SURFACES = ["bg-0", "bg-1", "bg-2", "bg-3"]


def _luminance(hex_colour):
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_the_tokens_the_theme_promises_exist():
    for name in SURFACES + ["text", "text-dim", "text-faint", "cyan", "blue", "amber", "red", "green"]:
        assert name in TOKENS, f"--{name} is missing from :root"


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("colour, minimum", [
    ("text", 7.0), ("text-dim", 4.5), ("text-faint", 4.5),
    ("cyan", 4.5), ("blue", 4.5), ("amber", 4.5), ("red", 4.5), ("green", 4.5),
])
def test_text_colours_are_legible_on_every_surface(colour, minimum, surface):
    ratio = contrast(TOKENS[colour], TOKENS[surface])
    assert ratio >= minimum, f"--{colour} on --{surface} is {ratio:.2f}:1, needs {minimum}:1"


def test_surfaces_are_charcoal_not_pure_black():
    for name in SURFACES:
        assert TOKENS[name].lower() != "#000000", f"--{name} is pure black"
        assert _luminance(TOKENS[name]) > 0.002


def test_dim_accents_are_never_used_as_text():
    """The -dim variants are for borders and glows: they do not meet text contrast. Only a
    decorative ::before/::after glyph may use one."""
    violations = []
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", CSS):
        if re.search(r"(?<![-\w])color\s*:\s*var\(--(?:cyan|blue|amber|red|green)-dim\)", body) and "::" not in selector:
            violations.append(selector.strip())
    assert violations == [], f"-dim colour used as text in: {violations}"


def test_the_meter_fill_needs_no_child_element():
    """Every template renders <span class="meter" style="--v:..."></span> with nothing inside. The fill
    once lived on `.meter > i`, which no template creates, so every meter showed an empty track."""
    assert not re.search(r"\.meter[^{]*>\s*i\b", CSS), "the meter fill must not depend on an <i> child"
    fill = re.search(r"\.meter::before\s*\{([^}]*)\}", CSS)
    assert fill, "no .meter::before rule draws the fill"
    assert "var(--v" in fill.group(1) and "width" in fill.group(1), "the fill must be sized by --v"
    for variant in ("cyan", "amber"):
        assert re.search(rf"\.meter\.{variant}::before", CSS), f".meter.{variant} has no fill colour"


def test_red_is_reserved_for_suspicious_indicators():
    """Colour contract: red means suspicious / high-risk only. The theme may define red
    tokens and a `.red` tag/dot, but no generic component (button, note, table, tab) may use it."""
    generic = []
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", CSS):
        uses_red = re.search(r"var\(--red(-dim|-bg)?\)", body)
        if uses_red and not re.search(r"\.red\b", selector):
            generic.append(selector.strip())
    assert generic == [], f"red used outside an explicit .red state: {generic}"
