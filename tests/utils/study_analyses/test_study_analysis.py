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
from __future__ import annotations

import re
import shutil
from pathlib import Path
from unittest import mock

import pytest

from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.utils.study_analyses import coupling_study_analysis
from gemseo.utils.study_analyses.coupling_study_analysis import CouplingStudyAnalysis
from gemseo.utils.study_analyses.mdo_study_analysis import MDOStudyAnalysis
from gemseo.utils.study_analyses.xls_study_parser import XLSStudyParser

INPUT_DIR = Path(__file__).parent / "study_inputs"

try:
    skip_condition = shutil.which("pdflatex") is None
except AttributeError:
    # for python 2.7 which is skipped anyway
    skip_condition = True

has_no_pdflatex = {
    "condition": skip_condition,
    "reason": "no pdflatex available",
}


@pytest.fixture(scope="module")
def study_analysis() -> CouplingStudyAnalysis:
    """A coupling study analysis."""
    return CouplingStudyAnalysis(INPUT_DIR / "disciplines_spec_without_scenario.xlsx")


def test_generate_n2(tmp_wd) -> None:
    analysis = MDOStudyAnalysis(INPUT_DIR / "disciplines_spec.xlsx")
    fpath = Path("xls_n2.pdf")
    analysis.generate_n2(fpath, fig_size=(5, 5))
    assert fpath.exists()


@pytest.mark.skipif(**has_no_pdflatex)
def test_xdsm_mdf(tmp_wd) -> None:
    analysis = MDOStudyAnalysis(INPUT_DIR / "disciplines_spec.xlsx")
    analysis.generate_xdsm(".", save_pdf=True)


def test_discipline_self_coupled_two_disciplines(tmp_wd) -> None:
    """Test that a GEMSEO study can be performed with a self-coupled discipline.

    In this test, two disciplines with one self-coupled discipline are present in the
    MDO process.
    """
    analysis = MDOStudyAnalysis(INPUT_DIR / "discipline_self_coupled.xlsx")
    fpath = Path("xls_n2.pdf")
    analysis.generate_n2(fpath, fig_size=(5, 5))
    analysis.generate_xdsm(".")
    assert fpath.exists()


def test_discipline_self_coupled_one_disc(tmp_wd) -> None:
    """Test that a GEMSEO study can be done with a self-coupled discipline.

    In this test, only one self-coupled discipline is present in the MDO process.
    """
    analysis = MDOStudyAnalysis(INPUT_DIR / "discipline_self_coupled_one_disc.xlsx")
    with pytest.raises(
        ValueError, match=re.escape("N2 diagrams need at least two disciplines.")
    ):
        analysis.generate_n2("xls_n2.pdf", fig_size=(5, 5))

    analysis.generate_xdsm(".")
    assert Path("xdsm.html").exists()


@pytest.mark.skipif(**has_no_pdflatex)
def test_xdsm_mdf_special_characters(tmp_wd) -> None:
    analysis = MDOStudyAnalysis(INPUT_DIR / "disciplines_spec_special_characters.xlsx")
    analysis.generate_xdsm(".", save_pdf=True)


@pytest.mark.skipif(**has_no_pdflatex)
def test_xdsm_idf(tmp_wd) -> None:
    analysis = MDOStudyAnalysis(INPUT_DIR / "disciplines_spec2.xlsx")
    dnames = ["Discipline1", "Discipline2"]
    assert list(analysis.study.disciplines) == dnames

    disc_names = [d.name for d in analysis.disciplines.values()]
    assert disc_names == disc_names
    analysis.generate_xdsm("", save_pdf=True)


def test_xdsm_bilevel(tmp_wd) -> None:
    analysis = MDOStudyAnalysis(INPUT_DIR / "study_bielvel_sobieski.xlsx")
    dnames = [
        "SobieskiAerodynamics",
        "SobieskiStructure",
        "SobieskiPropulsion",
        "SobieskiMission",
    ]
    assert list(analysis.study.disciplines.keys()) == dnames

    disc_names = [d.name for d in analysis.disciplines.values()]
    assert dnames == disc_names
    analysis.generate_n2()
    analysis.generate_xdsm(".")


def test_xdsm_bilevel_d(tmp_wd) -> None:
    analysis = MDOStudyAnalysis(INPUT_DIR / "bilevel_d.xlsx")
    analysis.generate_n2("n2_d.pdf")
    analysis.generate_xdsm(".")


def test_none_inputs() -> None:
    with pytest.raises(IOError):
        MDOStudyAnalysis(INPUT_DIR / "None.xlsx")


@pytest.mark.parametrize("file_index", range(1, 19))
def test_wrong_inputs(tmp_wd, file_index) -> None:
    fname = f"disciplines_spec_fail{file_index}.xlsx"
    with pytest.raises(ValueError):
        MDOStudyAnalysis(INPUT_DIR / fname)


def test_options() -> None:
    """Test that prescribed options are taken into account.

    This test also enables to make sure that there is no need to put '' when prescribing
    a string in option values.
    """
    analysis = MDOStudyAnalysis(INPUT_DIR / "disciplines_spec_options.xlsx")

    mda = analysis.scenarios["Scenario"].formulation.mda

    assert analysis.scenarios["Scenario"].name == "my_test_scenario"
    assert isinstance(mda, MDAGaussSeidel)
    assert mda.settings.warm_start is True
    assert mda.settings.tolerance == pytest.approx(1e-5)
    assert mda.over_relaxation_factor == pytest.approx(1.2)
    assert mda.settings.max_mda_iter == pytest.approx(20)


def test_xls_study_parser(tmp_wd, caplog) -> None:
    """Check the log of the XLSStudyParser."""
    XLSStudyParser(INPUT_DIR / "disciplines_spec.xlsx")
    expected_lines = [
        "2 disciplines detected",
        "   Discipline1",
        "      Inputs: a, b, c",
        "      Outputs: d, e, g",
        "   Discipline2",
        "      Inputs: d, e, x, z",
        "      Outputs: a, b, f",
        "1 scenario detected",
        "   Scenario",
        "      Objectives: f",
        "      Disciplines: Discipline1, Discipline2",
        "      Constraints: g",
        "      Design variables: b, x",
        "      Formulation: MDF",
    ]
    lines = [
        line for (_, _, lines) in caplog.record_tuples for line in lines.split("\n")
    ]
    assert len(expected_lines) == len(lines)
    for expected_line, line in zip(expected_lines, lines):
        assert line == expected_line


def test_coupling_study_analysis_n2_chart(study_analysis, tmp_wd) -> None:
    """Check CouplingStudyAnalysis."""
    """Check that the N2 graph is generated correctly from a study analysis."""
    study_analysis.generate_n2()
    assert Path("n2.pdf").exists()


def test_coupling_study_analysis_coupling_graph(study_analysis) -> None:
    """Check that the coupling graph is generated correctly from a study analysis."""
    file_path = Path("a.pdf")
    with mock.patch.object(coupling_study_analysis, "generate_coupling_graph") as meth:
        study_analysis.generate_coupling_graph(file_path)
        assert meth.call_args.args == (
            list(study_analysis.disciplines.values()),
            file_path,
            True,
        )
        study_analysis.generate_coupling_graph(file_path, False)
        assert meth.call_args.args == (
            list(study_analysis.disciplines.values()),
            file_path,
            False,
        )
