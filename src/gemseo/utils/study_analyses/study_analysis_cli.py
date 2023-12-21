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
from pathlib import Path
from typing import Final

from gemseo.utils.study_analyses.coupling_study_analysis import CouplingStudyAnalysis
from gemseo.utils.study_analyses.mdo_study_analysis import MDOStudyAnalysis

STUDY_ANALYSIS_TYPES: Final[dict[str, CouplingStudyAnalysis]] = {
    "coupling": CouplingStudyAnalysis,
    "mdo": MDOStudyAnalysis,
}
"""The types of study analyses."""


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
        "-t",
        "--study-type",
        help="The type of the study.",
        choices=STUDY_ANALYSIS_TYPES.keys(),
        default="mdo",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        help="The path of the directory to save the files.",
        type=str,
        default=Path().cwd(),
    )
    parser.add_argument(
        "-x",
        "--xdsm",
        help="Whether to generate a XDSM; compatible only with the study type 'mdo'.",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--save-pdf",
        help="Whether to save the XDSM as a PDF file.",
        action="store_true",
    )
    parser.add_argument(
        "--height",
        help="The height of the N2 figure in inches.",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--width",
        help="The width of the N2 figure in inches.",
        type=float,
        default=10.0,
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    directory_path = Path(args.out_dir)
    directory_path.mkdir(exist_ok=True)
    study = STUDY_ANALYSIS_TYPES[args.study_type](args.study_file)
    study.generate_n2(directory_path / "n2.pdf", fig_size=(args.height, args.width))
    file_name = "{}_coupling_graph.pdf"
    study.generate_coupling_graph(directory_path / file_name.format("full"))
    study.generate_coupling_graph(directory_path / file_name.format("condensed"), False)
    if args.xdsm:
        if args.study_type == "mdo":
            study.generate_xdsm(directory_path, save_pdf=args.save_pdf, show_html=True)
        else:
            raise ValueError(
                "The option 'xdsm' is compatible only with the study type 'mdo'."
            )
