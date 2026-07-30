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
from unittest import mock

import pytest

from gemseo.util.study_analysis.mdo_study_analysis import MDOStudyAnalysis
from gemseo.util.study_analysis.study_analysis_cli import main
from gemseo.util.study_analysis.study_analysis_cli import parse_args
from gemseo.util.testing.helper import assert_exception

INPUT_DIR = Path(__file__).parent / "study_inputs"
STUDY_FILE = INPUT_DIR / "disciplines_spec.xlsx"
STUDY_FILE_WITHOUT_SCENARIO = INPUT_DIR / "disciplines_spec_without_scenario.xlsx"


@pytest.fixture
def set_argv(monkeypatch):
    """A callable setting the CLI arguments."""

    def set_(*argv: str) -> None:
        """Set the CLI arguments.

        Args:
            *argv: The CLI arguments, without the program name.
        """
        monkeypatch.setattr("sys.argv", ["gemseo-study", *argv])

    return set_


def test_parse_args_default(set_argv, tmp_wd) -> None:
    """Verify the default CLI arguments."""
    set_argv(str(STUDY_FILE))
    args = parse_args()
    assert args.study_file == str(STUDY_FILE)
    assert args.study_type == "mdo"
    assert Path(args.out_dir) == tmp_wd
    assert args.xdsm is False
    assert args.save_pdf is False
    assert args.height == 15.0
    assert args.width == 10.0


def test_parse_args_custom(set_argv) -> None:
    """Verify the CLI arguments when all of them are passed."""
    set_argv(
        str(STUDY_FILE),
        "-t",
        "coupling",
        "-o",
        "out",
        "-x",
        "-p",
        "--height",
        "1.5",
        "--width",
        "2.5",
    )
    args = parse_args()
    assert args.study_type == "coupling"
    assert args.out_dir == "out"
    assert args.xdsm is True
    assert args.save_pdf is True
    assert args.height == 1.5
    assert args.width == 2.5


def test_main(set_argv, tmp_wd) -> None:
    """Verify that the N2 chart and the coupling graphs are generated."""
    set_argv(str(STUDY_FILE), "-o", "out")
    main()
    out_dir = tmp_wd / "out"
    assert (out_dir / "n2.pdf").exists()
    assert (out_dir / "full_coupling_graph.pdf").exists()
    assert (out_dir / "condensed_coupling_graph.pdf").exists()


def test_main_xdsm(set_argv, tmp_wd) -> None:
    """Verify that the XDSM diagram is generated when the option 'xdsm' is passed."""
    set_argv(str(STUDY_FILE), "-x", "-p")
    with mock.patch.object(MDOStudyAnalysis, "generate_xdsm") as generate_xdsm:
        main()

    assert generate_xdsm.call_args.args == (tmp_wd,)
    assert generate_xdsm.call_args.kwargs == {"save_pdf": True, "show_html": True}


@pytest.mark.usefixtures("tmp_wd")
def test_main_xdsm_with_coupling_study(set_argv, snapshot) -> None:
    """Verify that the option 'xdsm' is rejected for a coupling study."""
    set_argv(str(STUDY_FILE_WITHOUT_SCENARIO), "-t", "coupling", "-x")
    with assert_exception(ValueError, snapshot):
        main()
