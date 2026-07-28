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
"""A mkdocs hook rendering a complexity badge on each page.

The complexity level is opt-in and never inferred: it is displayed only on pages
that declare a valid ``complexity`` in their frontmatter (one of ``beginner``,
``intermediate`` or ``advanced``). On every other page (the default), or when the
declared value is not one of those levels, no badge is rendered.

The badge is inserted under the first heading, next to the reading-time badge
rendered by ``docs/hooks/reading_time.py``. The two hooks share the same
insertion point; because both badges are ``display: inline-flex`` they render on
the same line.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.pages import Page

# The complexity levels accepted in the frontmatter, matching the vocabulary
# already used by docs/_scripts/learning_paths.py for the learning paths.
_LEVELS = ("beginner", "intermediate", "advanced")

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6} .*)$", re.MULTILINE)

# Maps ``page.url`` to the declared complexity level, filled while the pages are
# rendered and dumped once the build is complete. The documentation home page
# fetches it to show the same level in its learning-path listing.
_registry: dict[str, str] = {}


def _badge(level: str) -> str:
    """Build the markdown snippet displaying the complexity level.

    Args:
        level: The complexity level, one of ``_LEVELS``.

    Returns:
        The markdown snippet to insert into the page.
    """
    return (
        f'<p class="complexity" markdown>:material-school-outline: {level.title()}</p>'
    )


def on_page_markdown(markdown: str, page: Page, **kwargs: Any) -> str:
    """Insert the complexity badge into the page when opted in.

    Args:
        markdown: The markdown content of the page.
        page: The page being rendered.
        **kwargs: The remaining event arguments passed by mkdocs (unused).

    Returns:
        The markdown content, with a complexity badge when a valid level is set.
    """
    level = str(page.meta.get("complexity", "")).strip().lower()
    if level not in _LEVELS:
        return markdown

    _registry[page.url] = level

    badge = _badge(level)
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
    """Write the collected complexity levels next to the other build assets.

    Args:
        config: The mkdocs configuration.
    """
    output_path = Path(config["site_dir"]) / "assets" / "complexities.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(_registry, output_file, ensure_ascii=False)
