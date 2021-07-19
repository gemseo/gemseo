#! /usr/bin/env python

# -*- coding: utf-8 -*-
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

import argparse
import sys
from configparser import ConfigParser
from pathlib import Path
from typing import Iterable


def main(
    output_file_path: str,
    add_install_requires: bool,
    extras_require_keys: Iterable[str],
) -> None:
    """Main routine.

    Args:
        output_file_path: Path to the output file.
        add_install_requires: If True, process the install_requires section.
        extras_require_keys: Extras_requires keys to process.
    """
    # read setup.cfg
    setup_cfg = ConfigParser()
    setup_cfg.read("setup.cfg")

    requirements = []

    if add_install_requires:
        try:
            requirements += [setup_cfg["options"]["install_requires"]]
        except KeyError:
            sys.exit("install_requires section cannot be found")

    for key in extras_require_keys:
        try:
            requirements += [setup_cfg["options.extras_require"][key]]
        except KeyError:
            sys.exit(f"extras_require name {key} cannot be found")

    # dump the requirements file
    with Path(output_file_path).open("w") as req_file:
        for req in requirements:
            if not req:
                continue
            req_file.write(f"{req.strip()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_file_path", help="Path to the output file")
    parser.add_argument(
        "--add-install-requires",
        default=False,
        action="store_true",
        help="Whether to process the install_requires section",
    )
    parser.add_argument(
        "--extras-require-key",
        default=[],
        action="append",
        metavar="NAME",
        help="Name of the extras_require to process, can be used multiple times",
    )

    args = parser.parse_args()

    main(args.output_file_path, args.add_install_requires, args.extras_require_key)
