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
"""Split the changelog into one page per minor version.

The file `CHANGELOG.md`, written by towncrier, stays the single source of the
changelog. At build time, this script splits it into the pages of the
directory [_output_dir][]:

- `index.md`: the introduction of `CHANGELOG.md`, the latest release and the
  nested list of all the pages and of their versions,
- `X.Y.md`: the versions `X.Y.*` (including the pre-releases such as `X.Y.0rc0`),
  for each minor version `X.Y` from [_first_individual_minor][] on,
- `older.md`: all the versions before [_first_individual_minor][],
- `SUMMARY.md`: the navigation, read by the plugin literate-nav.

The sections of the versions are copied in the order of `CHANGELOG.md`,
which must be the newest first, the only change being that the label starting
a top-level list item, e.g. `- **Added** ...`, is converted into a badge.
The script raises an error when `CHANGELOG.md` does not have the expected
structure instead of silently skipping what it does not understand.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from itertools import groupby
from operator import attrgetter
from pathlib import Path
from typing import Final

import mkdocs_gen_files
from markdown.extensions.toc import slugify

project_dir = Path(os.getenv("MKDOCS_CONFIG_DIR", "."))

_changelog_path: Final[Path] = project_dir / "CHANGELOG.md"
"""The path to the changelog written by towncrier."""

_edit_path: Final[str] = "CHANGELOG.md"
"""The path to the edited file of the generated pages, relative to the project."""

_output_dir: Final[str] = "changelog"
"""The directory of the generated pages, relative to the documentation directory."""

_intro_heading: Final[str] = "# Changelog"
"""The heading of the changelog title, before its introduction."""

_marker: Final[str] = "<!-- towncrier release notes start -->"
"""The comment ending the introduction and starting the release notes."""

_version_heading_pattern: Final[re.Pattern[str]] = re.compile(
    r"## Version (?P<version>(?P<major>\d+)\.(?P<minor>\d+)\.\d+\S*) \((?P<date>.+)\)"
)
"""The pattern of the heading of the section of a version, e.g. `## Version 7.0.0rc0 (2026-10-21)`."""  # noqa: E501

_first_individual_minor: Final[tuple[int, int]] = (6, 0)
"""The oldest minor version having its own page; the older versions share a page."""

_older_file_name: Final[str] = "older.md"
"""The name of the file of the page of the versions older than the first minor one."""

_older_title: Final[str] = "Older versions"
"""The title of the page of the versions older than the first minor one."""

_slugify_separator: Final[str] = "-"
"""The separator of the slugs of the headings, that of the extension `toc`."""

_labels: Final[tuple[str, ...]] = (
    "Breaking change",
    "Removed",
    "Deprecated",
    "Added",
    "Changed",
    "Fixed",
    "Security",
)
"""The labels starting the items of the release notes, e.g. `- **Added** ...`."""

_label_pattern: Final[re.Pattern[str]] = re.compile(
    r"^(?P<marker>[-*]) \*\*(?P<label>[^*\n]+)\*\* ", re.MULTILINE
)
"""The pattern of the beginning of a top-level list item starting with a bold text."""

_removed_are_breaking_sentence: Final[str] = (
    'The items labelled <span class="changelog-label changelog-label--removed">'
    "Removed</span> are breaking changes, "
    "as are those labelled "
    '<span class="changelog-label changelog-label--breaking-change">'
    "Breaking change</span>."
)
"""The sentence of the landing page stating that the removals are breaking changes."""


@dataclass(frozen=True)
class _Version:
    """The section of the changelog about a version."""

    name: str
    """The name of the version, e.g. `7.0.0rc0`."""

    minor: tuple[int, int]
    """The major and minor numbers of the version."""

    heading: str
    """The text of the heading of the section, e.g. `Version 7.0.0rc0 (2026-10-21)`."""

    text: str
    """The text of the section, starting with its heading, verbatim."""


@dataclass(frozen=True)
class _Page:
    """A page of the changelog."""

    file_name: str
    """The name of the file of the page."""

    title: str
    """The title of the page."""

    versions: tuple[_Version, ...]
    """The versions of the page, the newest first."""


def _split_changelog(text: str) -> tuple[str, str]:
    """Split the changelog into its introduction and its release notes.

    Args:
        text: The text of the changelog.

    Returns:
        The introduction, i.e. the text between the title of the changelog
        and the towncrier marker, and the release notes, i.e. the text after the marker.

    Raises:
        ValueError: When the title or the marker is missing.
    """
    title_match = re.search(rf"^{re.escape(_intro_heading)}$", text, re.MULTILINE)
    if title_match is None:
        msg = f"The heading '{_intro_heading}' is missing from {_changelog_path}."
        raise ValueError(msg)

    marker_index = text.find(_marker, title_match.end())
    if marker_index == -1:
        msg = f"The marker '{_marker}' is missing from {_changelog_path}."
        raise ValueError(msg)

    return (
        text[title_match.end() : marker_index].strip(),
        text[marker_index + len(_marker) :],
    )


def _parse_versions(release_notes: str) -> list[_Version]:
    """Parse the sections of the versions.

    Args:
        release_notes: The text of the changelog after the towncrier marker.

    Returns:
        The versions in the order of the text.

    Raises:
        ValueError: When the text has no version,
            when it has a text before the first version,
            or when a level-2 heading is not the heading of a version.
    """
    lines = release_notes.splitlines(keepends=True)
    heading_indices = [i for i, line in enumerate(lines) if line.startswith("## ")]
    if not heading_indices:
        msg = f"No heading '## Version X.Y.Z (date)' found in {_changelog_path}."
        raise ValueError(msg)

    if "".join(lines[: heading_indices[0]]).strip():
        msg = (
            f"{_changelog_path} has a text between the marker '{_marker}' "
            "and the first version."
        )
        raise ValueError(msg)

    versions = []
    for start, end in zip(
        heading_indices, [*heading_indices[1:], len(lines)], strict=True
    ):
        heading = lines[start].rstrip("\n")
        match = _version_heading_pattern.fullmatch(heading)
        if match is None:
            msg = (
                f"The heading '{heading}' of {_changelog_path} "
                f"does not match the pattern {_version_heading_pattern.pattern!r}."
            )
            raise ValueError(msg)

        versions.append(
            _Version(
                name=match["version"],
                minor=(int(match["major"]), int(match["minor"])),
                heading=heading.removeprefix("## "),
                text="".join(lines[start:end]),
            )
        )

    return versions


def _group_versions(versions: list[_Version]) -> list[_Page]:
    """Group the versions into pages.

    Args:
        versions: The versions, the newest first.

    Returns:
        The pages, the newest first, the page of the older versions being the last.
    """
    minor_to_versions: dict[tuple[int, int], list[_Version]] = {}
    older_versions = []
    for version in versions:
        if version.minor >= _first_individual_minor:
            minor_to_versions.setdefault(version.minor, []).append(version)
        else:
            older_versions.append(version)

    pages = [
        _Page(f"{major}.{minor}.md", f"Version {major}.{minor}", tuple(page_versions))
        for (major, minor), page_versions in minor_to_versions.items()
    ]
    if older_versions:
        pages.append(_Page(_older_file_name, _older_title, tuple(older_versions)))

    return pages


def _convert_labels(text: str) -> str:
    """Convert the labels starting the top-level list items into badges.

    The bold text elsewhere is left untouched.

    Args:
        text: The text of a section of the changelog.

    Returns:
        The text where the leading `**Label** ` of a top-level list item
        is replaced by a `<span>` styled by the CSS class
        `changelog-label changelog-label--<slug>`.

    Raises:
        ValueError: When a top-level list item starts with an unknown label.
    """

    def convert(match: re.Match[str]) -> str:
        label = match["label"]
        if label not in _labels:
            msg = (
                f"The label '{label}' of an item of {_changelog_path} is unknown; "
                f"the labels are {', '.join(_labels)}."
            )
            raise ValueError(msg)

        slug = _slugify(label)
        return (
            f'{match["marker"]} <span class="changelog-label changelog-label--{slug}">'
            f"{label}</span> "
        )

    return _label_pattern.sub(convert, text)


def _render_page(page: _Page) -> str:
    """Render the Markdown source of a page.

    Args:
        page: The page.

    Returns:
        The Markdown source of the page: its title followed by its versions,
        the labels of their items being converted into badges.
    """
    return (
        f"# {page.title}\n\n"
        + "".join(
            _convert_labels(version.text.rstrip("\n")) + "\n\n"
            for version in page.versions
        ).rstrip("\n")
        + "\n"
    )


def _slugify(heading: str) -> str:
    """Compute the anchor of a heading, as the extension `toc` does.

    Args:
        heading: The text of the heading.

    Returns:
        The identifier of the heading in the rendered page.
    """
    return slugify(heading, _slugify_separator)


def _render_index(intro: str, pages: list[_Page]) -> str:
    """Render the Markdown source of the landing page.

    Args:
        intro: The introduction of the changelog.
        pages: The pages, the newest first.

    Returns:
        The Markdown source of the landing page:
        the introduction
        and the list of the minor versions linking to their sections,
        the older ones being in a collapsed block.
    """
    items = []
    for page in pages:
        if page.file_name == _older_file_name:
            items.extend(("", f'??? note "[{page.title}]({page.file_name})"', ""))
            # Python-Markdown requires 4 spaces to indent the content of a block.
            for (major, minor), versions in groupby(
                page.versions, key=attrgetter("minor")
            ):
                anchor = _slugify(next(versions).heading)
                items.append(
                    f"    - [Version {major}.{minor}]({page.file_name}#{anchor})"
                )
        else:
            items.append(f"- [{page.title}]({page.file_name})")

    items.append("")

    return "\n".join([
        "# Changelog",
        "",
        intro,
        "",
        _removed_are_breaking_sentence,
        "",
        *items,
    ])


def _render_summary(pages: list[_Page]) -> str:
    """Render the navigation read by the plugin literate-nav.

    Args:
        pages: The pages, the newest first.

    Returns:
        The navigation: the landing page, the upgrading guide,
        then the pages, the newest first.
    """
    return "".join(
        f"- [{title}]({file_name})\n"
        for title, file_name in [
            ("Overview", "index.md"),
            ("Upgrading", "../software/upgrading.md"),
            *((page.title, page.file_name) for page in pages),
        ]
    )


def _write_page(file_name: str, content: str) -> None:
    """Write a generated page of the changelog.

    Args:
        file_name: The name of the file of the page.
        content: The Markdown source of the page.
    """
    path = f"{_output_dir}/{file_name}"
    with mkdocs_gen_files.open(path, "w") as page_file:
        page_file.write(content)

    if file_name != "SUMMARY.md":
        mkdocs_gen_files.set_edit_path(path, _edit_path)


_intro, _release_notes = _split_changelog(_changelog_path.read_text(encoding="utf-8"))
_pages = _group_versions(_parse_versions(_release_notes))

for _page in _pages:
    _write_page(_page.file_name, _render_page(_page))

_write_page("index.md", _render_index(_intro, _pages))
_write_page("SUMMARY.md", _render_summary(_pages))
