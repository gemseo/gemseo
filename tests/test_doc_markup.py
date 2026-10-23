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

_root_path = Path(__file__, "..", "..").resolve()

_doc_path = _root_path / "docs"

_src_path = _root_path / "src"

_generated_path = _doc_path / "generated"
"""The mkdocs-gallery build directory, mirroring the sources in ``docs/examples``."""

markup_to_regex = {
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


_admonition_regex = re.compile(r"^\s*(?:#\s*)?!!!\s")
"""The regular expression to find the first line of an admonition."""

_blank_line_regex = re.compile(r"\s*#?\s*")
"""The regular expression that a blank line matches, in Markdown or in a comment."""

_cell_separator = "# %%"
"""The separator between two cells of a gallery script, starting a new block."""


def get_file_paths() -> list[Path]:
    """Return the paths to the files whose markup must be checked.

    Returns:
        The paths to the documentation sources and to the Python modules.
    """
    paths = [
        path
        for pattern in ("*.md", "*.py")
        for path in _doc_path.rglob(pattern)
        if _generated_path not in path.parents
    ]
    paths.extend(_src_path.rglob("*.py"))
    return sorted(paths)


file_paths = get_file_paths()


@pytest.mark.parametrize("markup", markup_to_regex)
def test_markup_is_not_used(markup: str) -> None:
    """Check that a markup that the documentation cannot render is not used.

    Args:
        markup: The description of the markup.
    """
    regex = markup_to_regex[markup]
    paths = [
        str(path.relative_to(_root_path))
        for path in file_paths
        if regex.search(path.read_text(encoding="utf-8"))
    ]
    assert not paths, f"{markup} in {paths}"


def test_admonition_is_preceded_by_a_blank_line() -> None:
    """Check that a blank line precedes an admonition.

    Otherwise, python-markdown appends the admonition to the preceding paragraph
    and renders its marker verbatim.
    """
    locations = []
    for path in file_paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        locations.extend(
            f"{path.relative_to(_root_path)}:{index + 1}"
            for index, line in enumerate(lines)
            if index
            and _admonition_regex.match(line)
            and not _blank_line_regex.fullmatch(lines[index - 1])
            and lines[index - 1].strip() != _cell_separator
        )
    assert not locations, f"admonition without a preceding blank line in {locations}"
