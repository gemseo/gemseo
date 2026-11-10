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
"""Generate the documentation files that Zensical cannot generate itself.

Zensical supports neither MkDocs hooks nor the mkdocs-gallery and gen-files plugins,
so the prebuild of docs-tools does their work before the build
and this script wraps it with what is specific to GEMSEO.
Run it from the root of the repository before building or serving the documentation:

    uv run --group doc --extra all python docs/_scripts/prebuild.py

Pass ``--no-gallery`` to skip the examples.

It

- calls the prebuild of docs-tools, which runs the mkdocs-gallery examples
  into ``docs/generated/examples``, copies the gallery stylesheets
  into ``docs/css/gallery`` and writes ``_docs/generated.yml``,
  inherited by ``mkdocs-fast.yml``, with the navigation of ``nav.yml``
  where the gallery directories are expanded,
- writes the data of the home page (learning paths, reading times and complexity levels)
  into ``docs/assets``,
- writes the credits into ``_docs/credits.md``, replacing the page of docs-tools,
- splits ``CHANGELOG.md`` into the pages of ``docs/changelog``,
- writes the list of the global configuration options
  into ``_docs/global_configuration.md``,
- completes ``_docs/generated.yml``
  with the directories of the navigation with a ``SUMMARY.md`` expanded,
  with the footer columns of ``docs/footer.yml``
  and with the reading times displayed by the reading-time badges
  and the data of the home page.

An example failing unexpectedly makes the script fail.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import runpy
from pathlib import Path
from typing import Any

import yaml
from docs_tools.prebuild import prebuild

root_dir = Path.cwd()
"""The root directory of the repository."""

docs_dir = root_dir / "docs"
"""The directory of the documentation sources."""

generated_dir = root_dir / "_docs"
"""The directory of the generated files that are not documentation pages."""

complexity_levels = ("beginner", "intermediate", "advanced")
"""The complexity levels accepted in the front matter of a page."""

words_per_minute = 200
"""The average reading speed used to convert a word count into minutes."""

_fenced_code = re.compile(r"```.*?```", re.DOTALL)
_html_tag = re.compile(r"<[^>]+>")
_front_matter = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_summary_link = re.compile(
    r"^- \[(?P<title>[^\]]+)\]\((?P<target>[^)]+)\)\s*$", re.MULTILINE
)


def reset_gemseo_configuration() -> None:
    """Reset the global configuration of GEMSEO."""
    from gemseo.util.global_configuration import _configuration

    _configuration.__init__()


def read_front_matter(markdown: str) -> dict[str, Any]:
    """Read the front matter of a Markdown document.

    Args:
        markdown: The Markdown document.

    Returns:
        The front matter, empty when the document has none.
    """
    match = _front_matter.match(markdown)
    if match is None:
        return {}

    return yaml.safe_load(match.group(1)) or {}


def estimate_reading_time(markdown: str) -> int:
    """Estimate the reading time of a Markdown document.

    Fenced code blocks and inline HTML tags are dropped before counting words.

    Args:
        markdown: The Markdown content of the page.

    Returns:
        The estimated reading time in minutes (at least one).
    """
    text = _fenced_code.sub(" ", markdown)
    text = _html_tag.sub(" ", text)
    return max(1, math.ceil(len(text.split()) / words_per_minute))


def get_page_url(path: Path) -> str:
    """Return the URL of a page relative to the site root.

    Args:
        path: The path of the page source relative to the documentation directory.

    Returns:
        The URL of the page, e.g. ``user_guide/concepts/design_space/``.
    """
    parts = path.with_suffix("").parts
    if parts[-1] == "index":
        parts = parts[:-1]

    return "".join(f"{part}/" for part in parts)


def compute_badge_data() -> tuple[dict[str, int], dict[str, str]]:
    """Compute the reading times and complexity levels declared by the pages.

    Returns:
        The reading time in minutes and the complexity level of the pages
        opting in, indexed by page URL.
    """
    reading_times = {}
    complexities = {}
    for path in sorted(docs_dir.rglob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        front_matter = read_front_matter(markdown)
        if not front_matter:
            continue

        url = get_page_url(path.relative_to(docs_dir))
        if front_matter.get("reading_time", False) is True:
            reading_times[url] = estimate_reading_time(
                _front_matter.sub("", markdown, count=1)
            )

        level = str(front_matter.get("complexity", "")).strip().lower()
        if level in complexity_levels:
            complexities[url] = level

    return reading_times, complexities


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write data to a JSON file.

    Args:
        data: The data.
        path: The path of the JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def expand_summaries(nav: list[Any], directory: str = "") -> list[Any]:
    """Expand the directories of a navigation listing their pages in a SUMMARY.md.

    Zensical does not support the literate-nav plugin,
    which read the navigation of a directory from its ``SUMMARY.md``,
    a list of links ``- [Title](target)`` where a target ending with ``/``
    is a subdirectory with its own ``SUMMARY.md``.

    Args:
        nav: The navigation.
        directory: The directory the targets of the navigation are relative to,
            relative to the documentation directory and ending with ``/``
            unless empty.

    Returns:
        The navigation where each directory with a ``SUMMARY.md`` is replaced
        by the navigation of that file.
    """
    new_nav = []
    for item in nav:
        if isinstance(item, dict):
            ((title, value),) = item.items()
            if isinstance(value, list):
                value = expand_summaries(value, directory)
            elif isinstance(value, str):
                value = expand_summary(f"{directory}{value}")

            item = {title: value}

        new_nav.append(item)

    return new_nav


def expand_summary(target: str) -> str | list[Any]:
    """Expand a navigation target if it is a directory with a SUMMARY.md.

    Args:
        target: The navigation target, relative to the documentation directory.

    Returns:
        The navigation read from the ``SUMMARY.md`` of the directory,
        or the target itself when it is not such a directory.
    """
    summary_path = docs_dir / target / "SUMMARY.md"
    if not target.endswith("/") or not summary_path.is_file():
        return target

    nav = [
        {match["title"]: match["target"]}
        for match in _summary_link.finditer(summary_path.read_text(encoding="utf-8"))
    ]
    return expand_summaries(nav, target)


def write_generated_config(
    reading_times: dict[str, int], complexities: dict[str, str]
) -> None:
    """Complete the configuration generated by docs-tools.

    The directories of the navigation with a ``SUMMARY.md`` are expanded
    and the footer, the reading times and the data of the home page are added.

    Args:
        reading_times: The reading times in minutes indexed by page URL.
        complexities: The complexity levels indexed by page URL.
    """
    path = generated_dir / "generated.yml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["nav"] = expand_summaries(config["nav"])
    footer = yaml.safe_load((docs_dir / "footer.yml").read_text(encoding="utf-8"))
    config["extra"] = {
        "footer": footer,
        "reading_times": reading_times,
        # Embedded in the home page to render the learning paths without requests.
        "home_data": {
            "goals": json.loads(
                (docs_dir / "assets" / "learning_paths.json").read_text(
                    encoding="utf-8"
                )
            )["goals"],
            "times": reading_times,
            "levels": complexities,
        },
    }
    path.write_text(
        "# Generated by docs/_scripts/prebuild.py, do not edit.\n"
        + yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    """Generate the documentation files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-gallery",
        action="store_true",
        help="Do not run the examples of the gallery.",
    )
    arguments = parser.parse_args()
    try:
        prebuild(root_dir, run_examples=not arguments.no_gallery)
    finally:
        # The examples may leave the process in a deleted temporary directory
        # and a stale GEMSEO configuration.
        os.chdir(root_dir)
        reset_gemseo_configuration()

    scripts_dir = docs_dir / "_scripts"
    runpy.run_path(str(scripts_dir / "learning_paths.py"))
    # GEMSEO keeps its own credits page on purpose:
    # this overwrites the one written by docs-tools.
    runpy.run_path(str(scripts_dir / "credits.py"))
    # It writes docs/changelog/SUMMARY.md, which the navigation expands.
    runpy.run_path(str(scripts_dir / "changelog.py"))
    runpy.run_path(str(scripts_dir / "global_configuration.py"))
    reading_times, complexities = compute_badge_data()
    assets_dir = docs_dir / "assets"
    write_json(reading_times, assets_dir / "reading_times.json")
    write_json(complexities, assets_dir / "complexities.json")
    write_generated_config(reading_times, complexities)


if __name__ == "__main__":
    main()
