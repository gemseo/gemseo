#! /usr/bin/env python
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
"""Extract the requirements from setup.cfg to requirements .in files."""
from __future__ import annotations

import argparse
from configparser import ConfigParser
from pathlib import Path
from typing import Iterable
from typing import Sequence


def extract_requirments(
    output_file_path: str,
    add_install_requires: bool,
    extras_require_keys: Iterable[str],
) -> int:
    """Main routine.

    Args:
        output_file_path: The path to the output file.
        add_install_requires: Whether to consider the install_requires section.
        extras_require_keys: The extras_requires keys to consider.

    Returns:
        The exit status.
    """
    setup_cfg = ConfigParser()
    setup_cfg.read("setup.cfg")

    requirements = ""

    if add_install_requires:
        try:
            requirements += setup_cfg["options"]["install_requires"]
        except KeyError:
            print("install_requires section cannot be found")  # noqa: T001
            return 1

    for key in extras_require_keys:
        try:
            requirements += setup_cfg["options.extras_require"][key]
        except KeyError:
            print(f"extras_require name {key} cannot be found")  # noqa: T001
            return 1

    # Make sure there is no empty line before and a single newline after.
    Path(output_file_path).write_text(requirements.strip() + "\n")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: The CLI arguments.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_file_path", help="Path to the output file")
    parser.add_argument(
        "--add-install-requires",
        default=False,
        action="store_true",
        help="Whether to process the install_requires section",
    )
    parser.add_argument(
        "--add-extras-require-key",
        default=[],
        action="append",
        metavar="NAME",
        help="Name of the extras_require to process, can be used multiple times",
    )

    args = parser.parse_args(argv)

    return extract_requirments(
        args.output_file_path, args.add_install_requires, args.add_extras_require_key
    )


if __name__ == "__main__":
    raise SystemExit(main())
