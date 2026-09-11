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
from collections import Counter
from pathlib import Path

import pytest

doc_path = Path(__file__, "..", "..", "docs").resolve()

bib_path = doc_path / "references.bib"

_generated_path = doc_path / "generated"
"""The mkdocs-gallery build directory, mirroring the sources in ``docs/examples``."""

_entry_regex = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)
"""The regular expression to find the keys of the BibTeX entries."""

_citation_block_regex = re.compile(r"\[([^\]\[\n]*@[^\]\[\n]*)\]")
"""The regular expression to find the citation blocks in a documentation page.

A block can cite several references
and prefix or suffix a key,
e.g. ``[see @foo, p. 1; @bar]``,
as mkdocs-bibtex splits it on the semicolons
(see ``CITATION_BLOCK_REGEX`` in ``mkdocs_bibtex.citation``).
"""

_email_regex = re.compile(r"[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}")
"""The regular expression that an email address matches.

mkdocs-bibtex ignores the parts of a citation block including an email address,
e.g. ``[contact@gemseo.org]``.
"""

_citation_key_regex = re.compile(r"@([^\s,;]+)")
"""The regular expression to find the key of a citation.

The key pattern is deliberately loose,
so that a key that mkdocs-bibtex cannot parse is caught by the tests
instead of being silently ignored.
"""

_key_regex = re.compile(r"^[\w-]+$")
"""The regular expression that a citation key must match.

mkdocs-bibtex parses the key of a bracketed citation with ``[\\w-]+``
(see ``CITATION_REGEX`` in ``mkdocs_bibtex.citation``),
so a key including any other character, e.g. a colon, cannot be cited.
"""


def get_bib_keys() -> list[str]:
    """Return the keys of the BibTeX entries.

    Returns:
        The keys of the BibTeX entries, in the order of the file.
    """
    return _entry_regex.findall(bib_path.read_text(encoding="utf-8"))


def get_citation_keys(text: str) -> set[str]:
    """Return the keys of the references cited in a text.

    Args:
        text: The text of a documentation source.

    Returns:
        The keys of the cited references.
    """
    return {
        key
        for block in _citation_block_regex.findall(text)
        for citation in block.split(";")
        if not _email_regex.search(citation)
        for key in _citation_key_regex.findall(citation)
    }


def get_page_paths() -> list[Path]:
    """Return the paths to the documentation sources that can cite a reference.

    Returns:
        The paths to the Markdown pages and to the gallery scripts.
    """
    return sorted(
        path
        for pattern in ("*.md", "*.py")
        for path in doc_path.rglob(pattern)
        if _generated_path not in path.parents
    )


page_paths = get_page_paths()


def test_bib_keys_are_unique() -> None:
    """Check that no BibTeX entry is defined twice."""
    duplicated_keys = [
        key for key, count in Counter(get_bib_keys()).items() if count > 1
    ]
    assert not duplicated_keys


@pytest.mark.parametrize("key", get_bib_keys())
def test_bib_key_is_citable(key: str) -> None:
    """Check that a BibTeX key can be used in a citation.

    Args:
        key: The key of a BibTeX entry.
    """
    assert _key_regex.match(key)


@pytest.mark.parametrize(
    "page_path",
    page_paths,
    ids=(str(path.relative_to(doc_path)) for path in page_paths),
)
def test_citations_are_defined(page_path: Path) -> None:
    """Check that the citations of a documentation source are defined.

    Args:
        page_path: The path to a Markdown page or a gallery script.
    """
    keys = get_citation_keys(page_path.read_text(encoding="utf-8"))
    assert not keys.difference(get_bib_keys())
