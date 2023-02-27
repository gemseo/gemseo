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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""CLI for |g| study."""
from __future__ import annotations

import argparse
from ast import literal_eval
from os import getcwd
from pathlib import Path

from gemseo.utils.study_analysis import StudyAnalysis


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "A tool to generate a N2 chart and an XDSM diagram "
            "from an Excel description file."
        )
    )
    parser.add_argument(
        "study_file",
        help="The path of the XLS file that describes the study.",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--out_dir",
        help="The path of the directory to save the files.",
        type=str,
        default=getcwd(),
    )

    parser.add_argument(
        "-x", "--xdsm", help="Whether to generate a XDSM.", action="store_true"
    )

    parser.add_argument(
        "-p",
        "--save_pdf",
        help="Whether to save the XDSM as a PDF file.",
        action="store_true",
    )

    parser.add_argument(
        "-s", "--fig_size", help="The size of the N2 figure, as a tuple (x,y)", type=str
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    directory_path = Path(args.out_dir)
    directory_path.mkdir(exist_ok=True)
    study = StudyAnalysis(args.study_file)
    if args.fig_size is None:
        study.generate_n2(directory_path / "n2.pdf")
    else:
        study.generate_n2(
            directory_path / "n2.pdf", fig_size=literal_eval(args.fig_size)
        )

    if args.xdsm:
        study.generate_xdsm(directory_path, save_pdf=args.save_pdf, show_html=True)
