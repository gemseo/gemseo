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

from pathlib import Path

from mkdocs_gallery.gen_gallery import DEFAULT_GALLERY_CONF

file_dir_path = Path(__file__).parent
example_dir_name = "examples"

# TODO: find a way to put this into _docs
gallery_dir = file_dir_path / "generated" / example_dir_name
examples_dir = file_dir_path / example_dir_name
examples_subdirs = [
    subdir.name
    for subdir in examples_dir.iterdir()
    if (examples_dir / subdir).is_dir()
    and (examples_dir / subdir / "README.md").is_file()
]


def _patch_gallery():
    # To get the "reset_modules" to work,
    # we have to hard code _reset_dict similarly to what
    # is already built in it.
    import sys

    from mkdocs_gallery.scrapers import _reset_dict

    sys.path.append(str(file_dir_path / "_scripts"))
    import gallery_logging

    _reset_dict["gallery_logging.reset_logging"] = gallery_logging.reset_logging


_patch_gallery()


conf = {
    f"{example_dir_name}_dirs": [examples_dir / subdir for subdir in examples_subdirs],
    "gallery_dirs": [gallery_dir / subdir for subdir in examples_subdirs],
    # As a precaution, keep the already defined reset modules.
    "reset_modules": DEFAULT_GALLERY_CONF["reset_modules"]
    + ("gallery_logging.reset_logging",),
}
