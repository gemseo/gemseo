# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""A mkdocs hook rendering an estimated reading time on each page.

The reading time is opt-in: it is displayed only on pages that declare
``reading_time: true`` in their frontmatter. On every other page (the default),
the reading time is neither computed nor exposed.

The hook renders a badge under the first heading of every opted-in page and
writes ``assets/reading_times.json`` (a ``page url -> minutes`` map) at the end
of the build. The documentation home page fetches that map to display the same
computed times in its learning-path listing.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.pages import Page

# --- Reading-time protocol ---------------------------------------------------
# The estimation is deliberately simple and self-contained so it is easy to
# tune: change WORDS_PER_MINUTE, or adjust what estimate_reading_time counts.

WORDS_PER_MINUTE = 200
"""The average reading speed used to convert a word count into minutes."""

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")


def estimate_reading_time(markdown: str) -> int:
    """Estimate the reading time of a markdown document.

    Fenced code blocks and inline HTML tags are dropped before counting words.

    Args:
        markdown: The markdown content of the page.

    Returns:
        The estimated reading time in minutes (at least one).
    """
    text = _FENCED_CODE.sub(" ", markdown)
    text = _HTML_TAG.sub(" ", text)
    word_count = len(text.split())
    return max(1, math.ceil(word_count / WORDS_PER_MINUTE))


# --- mkdocs hook -------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6} .*)$", re.MULTILINE)

# Maps ``page.url`` to the estimated reading time in minutes, filled while the
# pages are rendered and dumped once the build is complete.
_registry: dict[str, int] = {}


def _badge(minutes: int) -> str:
    """Build the markdown snippet displaying the reading time.

    Args:
        minutes: The estimated reading time in minutes.

    Returns:
        The markdown snippet to insert into the page.
    """
    return f'<p class="reading-time" markdown>:material-clock-outline: {minutes} min read</p>'


def on_page_markdown(markdown: str, page: Page, **kwargs: Any) -> str:
    """Compute the reading time and insert its badge into the page.

    Args:
        markdown: The markdown content of the page.
        page: The page being rendered.
        **kwargs: The remaining event arguments passed by mkdocs (unused).

    Returns:
        The markdown content, with a reading-time badge when opted in.
    """
    if page.meta.get("reading_time", False) is not True:
        return markdown

    minutes = estimate_reading_time(markdown)
    _registry[page.url] = minutes

    badge = _badge(minutes)
    # Search the heading on a copy with fenced code blocks blanked out
    # (keeping the character offsets aligned)
    # so a `#` line inside a leading code block
    # is not mistaken for the first heading.
    masked = _FENCED_CODE.sub(lambda m: " " * len(m.group()), markdown)
    match = _HEADING.search(masked)
    if match is None:
        return f"{badge}\n\n{markdown}"

    end = match.end()
    return f"{markdown[:end]}\n\n{badge}\n{markdown[end:]}"


def on_post_build(config: MkDocsConfig) -> None:
    """Write the collected reading times next to the other build assets.

    Args:
        config: The mkdocs configuration.
    """
    output_path = Path(config["site_dir"]) / "assets" / "reading_times.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(_registry, output_file, ensure_ascii=False)
