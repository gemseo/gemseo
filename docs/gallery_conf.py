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
"""Configuration of mkdocs-gallery, loaded by the docs-tools prebuild.

The docs-tools prebuild patches mkdocs-gallery (Python 3.14 docstring parser,
example introductions, thumbnails) and adds its style resetter
before loading this file, so this file only holds what is specific to GEMSEO.

All the directories that must be run are gathered into 2 categories:
*tutorials* and *how-tos*.

A third category has been added (*bulk*) to gather all the examples that have not been modified yet.
"""

from __future__ import annotations

from pathlib import Path

from mkdocs_gallery.gen_gallery import DEFAULT_GALLERY_CONF
from mkdocs_gallery.sorting import _SortKey

file_dir_path = Path(__file__).parent
example_dir_name = "examples"

# TODO: find a way to put this into _docs
examples_dir = file_dir_path / example_dir_name

_leaky_examples = frozenset({"plot_howto_directory_manager.py"})
"""Examples that leak global state and must run after every other example."""


def _has_leaky_example(subdir: Path) -> bool:
    return any((subdir / name).is_file() for name in _leaky_examples)


examples_subdirs = []
for category_name in ["bulk", "howtos", "tutorials"]:
    directory_path = examples_dir / category_name
    examples_subdirs += [
        subdir
        for subdir in directory_path.iterdir()
        if subdir.is_dir() and (subdir / "README.md").is_file()
    ]

# Push subsections containing leaky examples to the end so they do not pollute
# later subsections via global state.
examples_subdirs.sort(key=_has_leaky_example)


def _patch_gallery():
    # To get the "reset_modules" to work,
    # we have to hard code _reset_dict similarly to what
    # is already built in it.
    import sys

    from mkdocs_gallery.scrapers import _reset_dict

    sys.path.append(str(file_dir_path / "_scripts"))
    import gallery_configuration
    import gallery_logging

    _reset_dict["gallery_configuration.reset_configuration"] = (
        gallery_configuration.reset_configuration
    )
    _reset_dict["gallery_logging.reset_logging"] = gallery_logging.reset_logging


_patch_gallery()

examples_dir_relative = [
    str(subdir.relative_to(file_dir_path)) for subdir in examples_subdirs
]


def insert_generated_in_path(path: Path) -> Path:
    """Insert the `generated` directory just after the `docs` directory.

    Args:
        path: The path within the `generated` directory must be added.

    Returns:
        The path containing the `generated` directory.
    """
    parts = list(path.parts)
    idx = parts.index("docs") + 1
    parts.insert(idx, "generated")
    return Path(*parts)


class _LeakyLastSortKey(_SortKey):
    """Sort examples by file name, pushing leaky examples to the end."""

    def __call__(self, file: Path) -> tuple[bool, str]:
        return file.name in _leaky_examples, file.name


conf = {
    "examples_dirs": examples_subdirs,
    "gallery_dirs": [insert_generated_in_path(subdir) for subdir in examples_subdirs],
    # As a precaution, keep the already defined reset modules.
    # The docs-tools prebuild appends its own style resetter after these ones.
    # `gallery_configuration.reset_configuration` must come before
    # `gallery_logging.reset_logging`, which routes the GEMSEO logs to the gallery.
    "reset_modules": DEFAULT_GALLERY_CONF["reset_modules"]
    + (
        "gallery_configuration.reset_configuration",
        "gallery_logging.reset_logging",
    ),
    "within_subsection_order": _LeakyLastSortKey,
    "default_thumb_file": str(
        file_dir_path / "assets" / "images" / "gemseo" / "monogram.png"
    ),
}
