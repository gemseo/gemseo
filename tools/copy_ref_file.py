#! env python

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
"""Script to copy reference files from a source to a destination."""

from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copy


def copy_files(source_rep: Path, dest_rep: Path, extension: str):
    """Copy files with prescribed extension from source to dest.

    The function is recursive through the source directory
    but only the files are copied into the destination directory,
    not their parent directories.

    Args:
        source_rep: The source directory.
        dest_rep: The destination directory.
        extension: The extension of files selected.
    """
    for f in source_rep.iterdir():
        if f.is_dir():
            copy_files(f, dest_rep, extension)
        elif f.suffix == extension:
            copy(f, dest_rep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy reference files from a source to a destination"
    )
    parser.add_argument(
        "source",
        type=Path,
        help="The source directory where files must be copied (explore the tree)",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="The destination directory where files must be pasted.",
    )
    parser.add_argument(
        "extension",
        type=str,
        help="Extension of files to be copied (with '.', e.g. '.pdf')",
    )

    args = parser.parse_args()

    copy_files(args.source, args.destination, args.extension)
