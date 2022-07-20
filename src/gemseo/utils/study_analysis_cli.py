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
from os import mkdir
from os.path import exists
from os.path import join

from gemseo.utils.study_analysis import StudyAnalysis


def parse_args():
    """Parse CLI arguments."""
    descr = (
        "A tool to generate a N2 and XDSM diagrams " + "from an excel description file."
    )
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument(
        "study_file", help="XLS file that describes the study", type=str
    )

    parser.add_argument(
        "-o",
        "--out_dir",
        help="Output directory for the generated files",
        type=str,
        default=getcwd(),
    )

    parser.add_argument(
        "-x", "--xdsm", help="If True, generates the XDSM file", action="store_true"
    )

    parser.add_argument(
        "-l",
        "--latex_output",
        help="If True, generates the XDSM in PDF" + "and Latex",
        action="store_true",
    )

    parser.add_argument(
        "-s", "--fig_size", help="Size of the N2 figure, tuple (x,y)", type=str
    )
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    out_dir = args.out_dir
    study_file = args.study_file

    latex_output = args.latex_output
    study = StudyAnalysis(study_file)

    if not exists(out_dir):
        mkdir(out_dir)

    if args.fig_size is not None:
        fig_size = literal_eval(args.fig_size)
        study.generate_n2(join(out_dir, "n2.pdf"), fig_size=fig_size)
    else:
        study.generate_n2(join(out_dir, "n2.pdf"))

    if args.xdsm:
        study.generate_xdsm(out_dir, latex_output=latex_output, open_browser=True)
