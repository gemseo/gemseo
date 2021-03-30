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

# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

import pytest

from gemseo.core.mdo_scenario import MDOScenario
from gemseo.problems.sobieski.core import SobieskiProblem
from gemseo.problems.sobieski.wrappers import (
    SobieskiAerodynamics,
    SobieskiMission,
    SobieskiPropulsion,
    SobieskiStructure,
)
from gemseo.utils.xdsm_tikz import xdsm_dict2tex
from gemseo.utils.xdsmizer import XDSMizer

from . import test_study_analysis


@pytest.fixture
def sobieski():
    disciplines = [
        SobieskiPropulsion(),
        SobieskiAerodynamics(),
        SobieskiMission(),
        SobieskiStructure(),
    ]
    design_space = SobieskiProblem().read_design_space()
    return disciplines, design_space


@pytest.mark.skipif(**test_study_analysis.has_no_pdflatex)
def test_mdf_jacobi(sobieski, tmp_path):
    disciplines, design_space = sobieski
    scn = MDOScenario(
        disciplines,
        formulation="MDF",
        objective_name="y_4",
        design_space=design_space,
    )
    scn.xdsmize()
    xdsmizer = XDSMizer(scn)
    xdsm = xdsmizer.xdsmize()
    returned = xdsm_dict2tex(xdsm, tmp_path)
    expected = {
        "Dis1": {"current": [2], "end": 2, "next": 3},  # MDAChain
        "Dis2": {"current": [3], "end": 5, "next": 4},  # MDAJacobi
        "Dis3": {"current": [4], "end": 4, "next": 5},  # SobieskiPropulsion
        "Dis4": {"current": [4], "end": 4, "next": 5},  # SobieskiAerodynamics
        "Dis5": {"current": [4], "end": 4, "next": 5},  # SobieskiStructure
        "Dis6": {"current": [6], "end": 6, "next": 7},  # SobieskiMission
        "Opt": {"current": [1], "end": 7, "next": 2},  # Optimizer
    }
    assert expected == returned


@pytest.mark.skipif(**test_study_analysis.has_no_pdflatex)
def test_mdf_gauss_seidel(sobieski, tmp_path):
    disciplines, design_space = sobieski
    scn = MDOScenario(
        disciplines,
        formulation="MDF",
        objective_name="y_4",
        design_space=design_space,
        sub_mda_class="MDAGaussSeidel",
    )
    scn.xdsmize()
    xdsmizer = XDSMizer(scn)
    xdsm = xdsmizer.xdsmize()
    returned = xdsm_dict2tex(xdsm, tmp_path)
    expected = {
        "Dis1": {"current": [2], "end": 2, "next": 3},  # MDAChain
        "Dis2": {"current": [3], "end": 7, "next": 4},  # MDAGaussSeidel
        "Dis3": {"current": [4], "end": 4, "next": 5},  # SobieskiPropulsion
        "Dis4": {"current": [5], "end": 5, "next": 6},  # SobieskiAerodynamics
        "Dis5": {"current": [6], "end": 6, "next": 7},  # SobieskiStructure
        "Dis6": {"current": [8], "end": 8, "next": 9},  # SobieskiMission
        "Opt": {"current": [1], "end": 9, "next": 2},  # Optimizer
    }
    assert expected == returned


@pytest.mark.skipif(**test_study_analysis.has_no_pdflatex)
def test_idf(sobieski, tmp_path):
    disciplines, design_space = sobieski
    scn = MDOScenario(
        disciplines,
        formulation="IDF",
        objective_name="y_4",
        design_space=design_space,
    )
    scn.xdsmize()
    xdsmizer = XDSMizer(scn)
    xdsm = xdsmizer.xdsmize()
    returned = xdsm_dict2tex(xdsm, tmp_path)
    expected = {
        "Dis1": {"current": [2], "end": 2, "next": 3},  # SobieskiPropulsion
        "Dis2": {"current": [2], "end": 2, "next": 3},  # SobieskiAerodynamics
        "Dis3": {"current": [2], "end": 2, "next": 3},  # SobieskiStructure
        "Dis4": {"current": [2], "end": 2, "next": 3},  # SobieskiMission
        "Opt": {"current": [1], "end": 3, "next": 2},  # Optimizer
    }
    assert expected == returned
