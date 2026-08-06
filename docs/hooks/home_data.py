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
"""A mkdocs hook inlining the home learning-path data into the home page.

`docs/assets/js/home.js` renders the "What do you want to do?" panel from three
build-time assets: the learning paths themselves
(`assets/learning_paths.json`, written by `docs/_scripts/learning_paths.py`),
the reading times (`assets/reading_times.json`, written by
`docs/hooks/reading_time.py`) and the complexity levels
(`assets/complexities.json`, written by `docs/hooks/complexity.py`).

Fetching them from the browser costs one network round trip each before anything
can be displayed. This hook merges the three into a single payload and embeds it
in the built home page, so the panel renders without any extra request. The
fetches remain in `home.js` as a fallback, hence a failure here only makes the
first display of the panel slower, never broken.

The three source assets are only complete once the build is over: the reading
times and the complexity levels are collected while the pages are rendered. This
is why the payload is injected into the *built* home page in `on_post_build`
rather than exposed to `docs/index.md` at rendering time.

This hook must therefore be declared **after** `docs/hooks/complexity.py` and
`docs/hooks/reading_time.py` in the `hooks` configuration: mkdocs fires an
event on the hooks in declaration order, and this one consumes what those two
write.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig

_LOGGER = logging.getLogger("mkdocs.hooks.home_data")

# The identifier of the injected element; `home.js` looks it up by this id.
_ELEMENT_ID = "gemseo-lp-data"

# The payload is injected right before this tag, the last stable anchor of the
# page: the `minify` plugin runs on `on_post_page`, i.e. before this hook, and
# keeps it.
_ANCHOR = "</body>"


def _load_json(path: Path, *, required: bool) -> dict[str, Any]:
    """Load a JSON asset written earlier in the build.

    Args:
        path: The path to the JSON file.
        required: Whether a missing file is an error;
            when `False`, a missing file yields an empty mapping.

    Returns:
        The parsed content, empty when the file is missing and not required.

    Raises:
        FileNotFoundError: When the file does not exist and is required.
    """
    if not path.is_file():
        if required:
            msg = f"{path} does not exist."
            raise FileNotFoundError(msg)
        _LOGGER.info("%s does not exist; home page data will lack it.", path)
        return {}

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _serialize(payload: dict) -> str:
    """Serialize the payload for embedding in an HTML `script` element.

    Args:
        payload: The data to embed.

    Returns:
        The JSON representation, with `<` escaped so that a `</script>`
        sequence in the data cannot close the element early.
    """
    return json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")


def on_post_build(config: MkDocsConfig) -> None:
    """Embed the learning-path data into the built home page.

    Args:
        config: The mkdocs configuration.

    Raises:
        FileNotFoundError: When the learning paths asset does not exist.
    """
    site_dir = Path(config["site_dir"])
    assets_dir = site_dir / "assets"

    # The learning paths are the only mandatory part: without them there is no
    # panel to render at all, so a missing file is a build error rather than a
    # silent fallback to fetching.
    paths = _load_json(assets_dir / "learning_paths.json", required=True)
    payload = {
        "goals": paths.get("goals", []),
        "times": _load_json(assets_dir / "reading_times.json", required=False),
        "levels": _load_json(assets_dir / "complexities.json", required=False),
    }

    home_page = site_dir / "index.html"
    if not home_page.is_file():
        _LOGGER.info("%s does not exist; home page data not inlined.", home_page)
        return

    html = home_page.read_text(encoding="utf-8")
    if _ANCHOR not in html:
        _LOGGER.info("%s has no %s; home page data not inlined.", home_page, _ANCHOR)
        return

    element = (
        f'<script type="application/json" id="{_ELEMENT_ID}">'
        f"{_serialize(payload)}</script>"
    )
    home_page.write_text(
        html.replace(_ANCHOR, f"{element}{_ANCHOR}", 1), encoding="utf-8"
    )
