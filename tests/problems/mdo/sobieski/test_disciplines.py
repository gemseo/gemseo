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
#      :author: Damien Guenot - 18 mars 2016
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from gemseo.problems.mdo.sobieski._disciplines_sg import SobieskiAerodynamicsSG
from gemseo.problems.mdo.sobieski._disciplines_sg import SobieskiMissionSG
from gemseo.problems.mdo.sobieski._disciplines_sg import SobieskiPropulsionSG
from gemseo.problems.mdo.sobieski._disciplines_sg import SobieskiStructureSG
from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure


def test_init_range() -> None:
    SobieskiMission("float64")


def test_init_weight() -> None:
    SobieskiStructure("float64")


def test_init_aero() -> None:
    SobieskiAerodynamics("float64")


def test_init_power() -> None:
    SobieskiPropulsion("float64")


def test_execute_range() -> None:
    sr = SobieskiMission("complex128")
    sr.execute()
    sr.check_jacobian(derr_approx="complex_step", step=1e-30)


def test_execute_weight() -> None:
    sr = SobieskiStructure("complex128")
    sr.execute()
    sr.check_jacobian(derr_approx="complex_step", step=1e-30)


def test_execute_power() -> None:
    sr = SobieskiPropulsion("complex128")
    sr.execute()
    sr.check_jacobian(derr_approx="complex_step", step=1e-30)


def test_execute_aerodynamics() -> None:
    sr = SobieskiAerodynamics("complex128")
    sr.execute()
    sr.check_jacobian(derr_approx="complex_step", step=1e-30)


DV_NAMES = ["x_shared", "x_1", "x_2", "x_3"]


def test_init_range_sg() -> None:
    SobieskiMissionSG("float64")


def test_init_weight_sg() -> None:
    SobieskiStructureSG("float64")


def test_init_aero_sg() -> None:
    SobieskiAerodynamicsSG("float64")


def test_init_power_sg() -> None:
    SobieskiPropulsionSG("float64")


def test_execute_range_sg() -> None:
    sr = SobieskiMissionSG("complex128")
    indata = SobieskiProblem("complex128").get_default_inputs(names=sr.io.input_grammar)
    sr.execute(indata)
    sr.linearize(indata, compute_all_jacobians=True)


def test_execute_weight_sg() -> None:
    sr = SobieskiStructureSG("float64")
    indata = SobieskiProblem("float64").get_default_inputs(names=sr.io.input_grammar)
    sr.execute(indata)
    sr.linearize(indata, compute_all_jacobians=True)


def test_execute_power_sg() -> None:
    sr = SobieskiPropulsionSG("float64")
    indata = SobieskiProblem("float64").get_default_inputs(names=sr.io.input_grammar)
    sr.execute(indata)
    sr.linearize(indata, compute_all_jacobians=True)


def test_execute_aerodynamics_sg() -> None:
    sr = SobieskiAerodynamicsSG("float64")
    indata = SobieskiProblem("float64").get_default_inputs(names=sr.io.input_grammar)
    sr.execute(indata)
    sr.linearize(indata, compute_all_jacobians=True)
