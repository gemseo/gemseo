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

from gemseo.problems.sobieski._disciplines_sg import SobieskiAerodynamicsSG
from gemseo.problems.sobieski._disciplines_sg import SobieskiMissionSG
from gemseo.problems.sobieski._disciplines_sg import SobieskiPropulsionSG
from gemseo.problems.sobieski._disciplines_sg import SobieskiStructureSG
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure


def test_init_range():
    SobieskiMission("float64")


def test_init_weight():
    SobieskiStructure("float64")


def test_init_aero():
    SobieskiAerodynamics("float64")


def test_init_power():
    SobieskiPropulsion("float64")


def test_execute_range():
    sr = SobieskiMission("complex128")
    sr.execute()
    sr.get_local_data_by_name(sr.get_output_data_names())
    sr.check_jacobian(derr_approx="complex_step", step=1e-30)


def test_execute_weight():
    sr = SobieskiStructure("complex128")
    sr.execute()
    _, _, _, _, _ = sr.get_local_data_by_name(sr.get_output_data_names())
    sr.check_jacobian(derr_approx="complex_step", step=1e-30)


def test_execute_power():
    sr = SobieskiPropulsion("complex128")
    sr.execute()
    _, _, _, _, _ = sr.get_local_data_by_name(sr.get_output_data_names())
    sr.check_jacobian(derr_approx="complex_step", step=1e-30)


def test_execute_aerodynamics():
    sr = SobieskiAerodynamics("complex128")
    sr.execute()
    _, _, _, _, _ = sr.get_local_data_by_name(sr.get_output_data_names())
    sr.check_jacobian(derr_approx="complex_step", step=1e-30)


DV_NAMES = ["x_shared", "x_1", "x_2", "x_3"]


def test_init_range_sg():
    SobieskiMissionSG("float64")


def test_init_weight_sg():
    SobieskiStructureSG("float64")


def test_init_aero_sg():
    SobieskiAerodynamicsSG("float64")


def test_init_power_sg():
    SobieskiPropulsionSG("float64")


def test_execute_range_sg():
    sr = SobieskiMissionSG("complex128")
    indata = SobieskiProblem("complex128").get_default_inputs(
        names=sr.get_input_data_names()
    )
    sr.execute(indata)
    sr.get_local_data_by_name(sr.get_output_data_names())
    sr.linearize(indata, force_all=True)


def test_execute_weight_sg():
    sr = SobieskiStructureSG("float64")
    indata = SobieskiProblem("float64").get_default_inputs(
        names=sr.get_input_data_names()
    )
    sr.execute(indata)
    _, _, _, _, _ = sr.get_local_data_by_name(sr.get_output_data_names())
    sr.linearize(indata, force_all=True)


def test_execute_power_sg():
    sr = SobieskiPropulsionSG("float64")
    indata = SobieskiProblem("float64").get_default_inputs(
        names=sr.get_input_data_names()
    )
    sr.execute(indata)
    _, _, _, _, _ = sr.get_local_data_by_name(sr.get_output_data_names())
    sr.linearize(indata, force_all=True)


def test_execute_aerodynamics_sg():
    sr = SobieskiAerodynamicsSG("float64")
    indata = SobieskiProblem("float64").get_default_inputs(
        names=sr.get_input_data_names()
    )
    sr.execute(indata)
    _, _, _, _, _ = sr.get_local_data_by_name(sr.get_output_data_names())
    sr.linearize(indata, force_all=True)
