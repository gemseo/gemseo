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
"""Assemble the documentation home learning paths into a single JSON asset.

Each goal is described by a ``docs/learning_paths/<goal>.yml`` file and the
display order lives in ``_config.yml``. This script reads them at build time,
validates the content and writes ``assets/learning_paths.json``, which the home
page fetches and renders.

The ``title`` of a resource or a prerequisite is optional: when it is omitted,
it defaults to the first heading of the referenced page, read from the page
source by [_read_page_title][], minus the type prefix a few of those headings
carry for the listing they appear in. That heading is deliberately preferred
over the label the page carries in the mkdocs ``nav``, which differs on a few
pages: the heading is what a reader sees at the top of the page itself. Set
``title`` explicitly only to override it. It is mandatory for an entry whose
``path`` is an external ``http(s)`` URL, as there is no page source to read.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mkdocs_gen_files
import yaml

_TYPES = frozenset({"explanation", "tutorial", "howto", "reference"})

_REQUIRED_GOAL_KEYS = ("id", "code", "title", "audience", "blurb", "resources")

_SOURCE_DIR = Path("docs/learning_paths")
_DOCS_DIR = _SOURCE_DIR.parent
_OUTPUT_PATH = "assets/learning_paths.json"

# The gallery pages are written by the gallery plugin, possibly after this
# script runs, so a `generated/examples/<rest>/` path is resolved against the
# example source it is built from instead: the `docs/examples/<rest>.py` script
# of a single example, or the `README.md` of a gallery section.
_GALLERY_PREFIX = "generated/examples/"
_EXAMPLES_DIR = _DOCS_DIR / "examples"

# How many snippet includes [_read_page_title][] follows before giving up.
_MAX_SNIPPET_DEPTH = 3

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# The module docstring of an example script, whose first heading is the title
# the gallery gives to the generated page.
_PY_DOCSTRING = re.compile(r'"""(.*?)"""', re.DOTALL)
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
# The trailing attribute list of a heading, e.g. `# Title { #the-anchor }`.
_HEADING_ATTRIBUTES = re.compile(r"\s*\{[^}]*\}\s*$")
# The type prefix a heading carries for the listing it appears in: the gallery
# index for an example, the algorithms section for a generated table. The card
# of the home page is already headed by the type of its group, so the prefix
# would only be said twice, on a line that has no room for it.
_TITLE_PREFIX = re.compile(r"^(?:Tutorial\s*-\s*|Available:\s*)")
# A pymdownx.snippets include, resolved from the project root as that extension
# is configured without a `base_path`. The pages of the `algorithms/` section
# hold nothing but such an include, so their title lives in the included file,
# which docs/_scripts/algos/script.py writes before this script runs.
_SNIPPET = re.compile(r'^--8<--\s+"([^"]+)"\s*$', re.MULTILINE)
_PROJECT_DIR = _DOCS_DIR.parent


def _load_yaml(path: Path) -> dict:
    """Load a YAML file into a dictionary.

    Args:
        path: The path to the YAML file.

    Returns:
        The parsed content.

    Raises:
        TypeError: When the file is empty or does not hold a mapping.
    """
    with path.open(encoding="utf-8") as file:
        content = yaml.safe_load(file)
    if not isinstance(content, dict):
        msg = f"{path}: expected a YAML mapping, got {type(content).__name__}."
        raise TypeError(msg)
    return content


def _is_external(path: str) -> bool:
    """Tell whether a path points outside the documentation site.

    Args:
        path: The `path` value of a resource or a prerequisite.

    Returns:
        Whether the path is an external URL.
    """
    return path.startswith(("http://", "https://"))


def _resolve_source(path: str) -> Path | None:
    """Find the source file a documentation path is built from.

    Args:
        path: The `path` value of a resource or a prerequisite.

    Returns:
        The source file, or `None` for an external URL
        and when no source file matches.
    """
    if _is_external(path):
        return None

    stripped = path.strip("/")
    if stripped.startswith(_GALLERY_PREFIX):
        rest = stripped[len(_GALLERY_PREFIX) :]
        candidates = (_EXAMPLES_DIR / f"{rest}.py", _EXAMPLES_DIR / rest / "README.md")
    else:
        candidates = (_DOCS_DIR / f"{stripped}.md", _DOCS_DIR / stripped / "index.md")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def _check_path(path: str, source: Path) -> Path | None:
    """Check that a path points to a page the documentation actually builds.

    External URLs are skipped: they are not site files.

    Args:
        path: The `path` value of a resource or a prerequisite.
        source: The file the path was read from, used in error messages.

    Returns:
        The source file the page is built from,
        or `None` for an external URL.

    Raises:
        ValueError: When the path matches no documentation page.
    """
    file = _resolve_source(path)
    if file is None and not _is_external(path):
        msg = f"{source}: path '{path}' matches no documentation page."
        raise ValueError(msg)

    return file


def _read_page_title(file: Path, depth: int = 0) -> str | None:
    """Read the title of a documentation page from its source file.

    The title is the first level-one heading, with its trailing attribute list
    and its type prefix stripped. In an example script, that heading is
    searched in the module docstring only, so that the license comment cannot
    be mistaken for it. A page holding no heading of its own but including a
    snippet is followed into that snippet.

    Args:
        file: The source file of the page, as returned by [_resolve_source][].
        depth: The number of snippet includes already followed,
            bounding the recursion.

    Returns:
        The title, or `None` when neither the file nor the snippets it includes
        declare a level-one heading.
    """
    text = file.read_text(encoding="utf-8")

    if file.suffix == ".py":
        docstring = _PY_DOCSTRING.search(text)
        if docstring is None:
            return None
        text = docstring.group(1)
    else:
        text = _HTML_COMMENT.sub(" ", text)

    text = _FENCED_CODE.sub(" ", text)
    heading = _H1.search(text)
    if heading is not None:
        title = _HEADING_ATTRIBUTES.sub("", heading.group(1))
        return _TITLE_PREFIX.sub("", title).strip() or None

    if depth >= _MAX_SNIPPET_DEPTH:
        return None

    for match in _SNIPPET.finditer(text):
        included = _PROJECT_DIR / match.group(1)
        if not included.is_file():
            continue
        title = _read_page_title(included, depth + 1)
        if title is not None:
            return title

    return None


def _fill_title(entry: dict, source: Path, file: Path | None) -> None:
    """Default the title of a resource or a prerequisite to its page title.

    An entry that already carries a `title` is left untouched: an explicit
    title overrides the one of the referenced page.

    Args:
        entry: The resource or prerequisite definition, modified in place.
        source: The file the entry was read from, used in error messages.
        file: The source file of the referenced page,
            as returned by [_check_path][],
            or `None` when the entry points to an external URL.

    Raises:
        ValueError: When the title is omitted
            and the entry points to an external URL
            or cannot be read from the referenced page.
    """
    if entry.get("title"):
        return

    path = entry["path"]
    if file is None:
        msg = (
            f"{source}: no 'title' given for the external URL '{path}'; "
            f"'title' is mandatory when 'path' points outside the documentation."
        )
        raise ValueError(msg)

    title = _read_page_title(file)
    if title is None:
        msg = (
            f"{source}: no 'title' given for path '{path}' and none could be read "
            f"from the page it points to; set 'title' explicitly."
        )
        raise ValueError(msg)

    entry["title"] = title


def _validate_goal(goal: dict, goal_id: str, source: Path) -> None:
    """Validate a single goal definition.

    Args:
        goal: The parsed goal definition.
        goal_id: The identifier from `_config.yml` (also the file stem);
            the goal's own `id` field must match it.
        source: The file the goal was read from, used in error messages.

    Raises:
        ValueError: When a required key is missing,
            an enum value is invalid,
            a path matches no documentation page,
            a title is neither given nor readable from the page it points to,
            or
            the goal's `id` does not match `goal_id`.
    """
    for key in _REQUIRED_GOAL_KEYS:
        if key not in goal:
            msg = f"{source}: missing required key '{key}'."
            raise ValueError(msg)

    if goal["id"] != goal_id:
        msg = (
            f"{source}: goal id '{goal['id']}' does not match '{goal_id}' "
            f"(from _config.yml order / file name)."
        )
        raise ValueError(msg)

    for prerequisite in goal.get("prerequisites") or ():
        if "path" not in prerequisite:
            msg = f"{source}: prerequisite missing required key 'path'."
            raise ValueError(msg)
        file = _check_path(prerequisite["path"], source)
        _fill_title(prerequisite, source, file)

    for resource in goal["resources"]:
        for key in ("type", "desc", "path"):
            if key not in resource:
                msg = f"{source}: resource missing required key '{key}'."
                raise ValueError(msg)
        if resource["type"] not in _TYPES:
            msg = f"{source}: invalid type '{resource['type']}' (expected {sorted(_TYPES)})."
            raise ValueError(msg)
        file = _check_path(resource["path"], source)
        _fill_title(resource, source, file)


def _build_payload() -> dict:
    """Build the JSON payload from the learning path files.

    Returns:
        A dictionary with the ordered goals.

    Raises:
        FileNotFoundError: When a goal listed in the order has no YAML file.
        TypeError: When a YAML file is empty or does not hold a mapping.
        ValueError: When a goal definition is invalid.
    """
    config = _load_yaml(_SOURCE_DIR / "_config.yml")

    goals = []
    for goal_id in config["order"]:
        source = _SOURCE_DIR / f"{goal_id}.yml"
        if not source.is_file():
            msg = f"{_SOURCE_DIR}/_config.yml lists '{goal_id}' but {source} does not exist."
            raise FileNotFoundError(msg)
        goal = _load_yaml(source)
        _validate_goal(goal, goal_id, source)
        goals.append(goal)

    return {"goals": goals}


with mkdocs_gen_files.open(_OUTPUT_PATH, "w") as output_file:
    json.dump(_build_payload(), output_file, ensure_ascii=False)
