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
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT_PATH = Path(__file__, "..", "..").resolve()

_DOC_PATH = _ROOT_PATH / "docs"

_SRC_PATH = _ROOT_PATH / "src"

_GENERATED_PATH = _DOC_PATH / "generated"
"""The mkdocs-gallery build directory, mirroring the sources in ``docs/examples``."""

MARKUP_TO_REGEX = {
    "malformed admonition, e.g. '!!!note' instead of '!!! note'": re.compile(
        r"^\s*#?\s*!!!\w", re.MULTILINE
    ),
    "Markdown footnote definition, e.g. '[^1]: Author, Title'": re.compile(
        r"^\s*#?\s*\[\^[^\]\n]+\]:", re.MULTILINE
    ),
    "reST citation definition, e.g. '.. [1] Author, Title'": re.compile(
        r"^\s*\.\. \[\d+\]", re.MULTILINE
    ),
    "reST citation reference, e.g. '[1]_'": re.compile(r"\[\d+\]_"),
    "reST hyperlink target, e.g. '.. _Name: https://www.gemseo.org'": re.compile(
        r"^\s*\.\. _[^:\n]+:", re.MULTILINE
    ),
    "reST hyperlink reference, e.g. '`Name`_'": re.compile(
        r"`[A-Za-z][^`\n]*`_(?![A-Za-z0-9_])"
    ),
}
"""The markups that must not be used, mapped to their regular expressions.

The documentation is rendered from Markdown:
the reST markups are leftovers from the Sphinx era and are rendered verbatim,
while a Markdown footnote definition is only rendered in the document defining it,
and so cannot be used in a docstring,
which mkdocstrings renders separately from the page including it.
The references must be cited from the ``docs/references.bib`` file
in the case of a documentation page
and rendered as a ``!!! quote "References"`` admonition
in the case of a docstring.
"""


_ADMONITION_REGEX = re.compile(r"^\s*(?:#\s*)?!!!\s")
"""The regular expression to find the first line of an admonition."""

_BLANK_LINE_REGEX = re.compile(r"\s*#?\s*")
"""The regular expression that a blank line matches, in Markdown or in a comment."""

_CELL_SEPARATOR = "# %%"
"""The separator between two cells of a gallery script, starting a new block."""


def get_file_paths() -> list[Path]:
    """Return the paths to the files whose markup must be checked.

    Returns:
        The paths to the documentation sources and to the Python modules.
    """
    paths = [
        path
        for pattern in ("*.md", "*.py")
        for path in _DOC_PATH.rglob(pattern)
        if _GENERATED_PATH not in path.parents
    ]
    paths.extend(_SRC_PATH.rglob("*.py"))
    return sorted(paths)


FILE_PATHS = get_file_paths()


@pytest.mark.parametrize("markup", MARKUP_TO_REGEX)
def test_markup_is_not_used(markup: str) -> None:
    """Check that a markup that the documentation cannot render is not used.

    Args:
        markup: The description of the markup.
    """
    regex = MARKUP_TO_REGEX[markup]
    paths = [
        str(path.relative_to(_ROOT_PATH))
        for path in FILE_PATHS
        if regex.search(path.read_text(encoding="utf-8"))
    ]
    assert not paths, f"{markup} in {paths}"


def test_admonition_is_preceded_by_a_blank_line() -> None:
    """Check that a blank line precedes an admonition.

    Otherwise, python-markdown appends the admonition to the preceding paragraph
    and renders its marker verbatim.
    """
    locations = []
    for path in FILE_PATHS:
        lines = path.read_text(encoding="utf-8").splitlines()
        locations.extend(
            f"{path.relative_to(_ROOT_PATH)}:{index + 1}"
            for index, line in enumerate(lines)
            if index
            and _ADMONITION_REGEX.match(line)
            and not _BLANK_LINE_REGEX.fullmatch(lines[index - 1])
            and lines[index - 1].strip() != _CELL_SEPARATOR
        )
    assert not locations, f"admonition without a preceding blank line in {locations}"
