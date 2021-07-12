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
#    INITIAL AUTHORS - API and implementation and/or documentation
#      :author: Damien Guenot - 18 mars 2016
#    OTHER AUTHORS   - MACROSCOPIC CHANGES


from __future__ import division, unicode_literals

import unittest

from gemseo.problems.sobieski.core import SobieskiProblem
from gemseo.problems.sobieski.wrappers import (
    SobieskiAerodynamics,
    SobieskiMission,
    SobieskiPropulsion,
    SobieskiStructure,
)
from gemseo.problems.sobieski.wrappers_sg import (
    SobieskiAerodynamicsSG,
    SobieskiMissionSG,
    SobieskiPropulsionSG,
    SobieskiStructureSG,
)


class TestSobieskiWrapper(unittest.TestCase):
    """"""

    def test_init_range(self):
        """"""
        SobieskiMission("float64")

    def test_init_weight(self):
        """"""
        SobieskiStructure("float64")

    def test_init_aero(self):
        """"""
        SobieskiAerodynamics("float64")

    def test_init_power(self):
        """"""
        SobieskiPropulsion("float64")

    def test_execute_range(self):
        """"""
        sr = SobieskiMission("complex128")
        sr.execute()
        sr.get_local_data_by_name(sr.get_output_data_names())
        sr.check_jacobian(derr_approx="complex_step", step=1e-30)

    def test_execute_weight(self):
        """"""
        sr = SobieskiStructure("complex128")
        sr.execute()
        _, _, _, _, _ = sr.get_local_data_by_name(sr.get_output_data_names())
        sr.check_jacobian(derr_approx="complex_step", step=1e-30)

    def test_execute_power(self):
        """"""
        sr = SobieskiPropulsion("complex128")
        sr.execute()
        _, _, _, _, _ = sr.get_local_data_by_name(sr.get_output_data_names())
        sr.check_jacobian(derr_approx="complex_step", step=1e-30)

    def test_execute_aerodynamics(self):
        """"""
        sr = SobieskiAerodynamics("complex128")
        sr.execute()
        _, _, _, _, _ = sr.get_local_data_by_name(sr.get_output_data_names())
        sr.check_jacobian(derr_approx="complex_step", step=1e-30)


class TestSobieskiWrapperSG(unittest.TestCase):
    """"""

    DV_NAMES = ["x_shared", "x_1", "x_2", "x_3"]

    def test_init_range(self):
        """"""
        SobieskiMissionSG("float64")

    def test_init_weight(self):
        """"""
        SobieskiStructureSG("float64")

    def test_init_aero(self):
        """"""
        SobieskiAerodynamicsSG("float64")

    def test_init_power(self):
        """"""
        SobieskiPropulsionSG("float64")

    def test_execute_range(self):
        """"""
        sr = SobieskiMissionSG("complex128")
        indata = SobieskiProblem("complex128").get_default_inputs(
            names=sr.get_input_data_names()
        )
        sr.execute(indata)
        sr.get_local_data_by_name(sr.get_output_data_names())
        sr.linearize(indata, force_all=True)

    def test_execute_weight(self):
        """"""
        sr = SobieskiStructureSG("float64")
        indata = SobieskiProblem("float64").get_default_inputs(
            names=sr.get_input_data_names()
        )
        sr.execute(indata)
        _, _, _, _, _ = sr.get_local_data_by_name(sr.get_output_data_names())
        sr.linearize(indata, force_all=True)

    def test_execute_power(self):
        """"""
        sr = SobieskiPropulsionSG("float64")
        indata = SobieskiProblem("float64").get_default_inputs(
            names=sr.get_input_data_names()
        )
        sr.execute(indata)
        _, _, _, _, _ = sr.get_local_data_by_name(sr.get_output_data_names())
        sr.linearize(indata, force_all=True)

    def test_execute_aerodynamics(self):
        """"""
        sr = SobieskiAerodynamicsSG("float64")
        indata = SobieskiProblem("float64").get_default_inputs(
            names=sr.get_input_data_names()
        )
        sr.execute(indata)
        _, _, _, _, _ = sr.get_local_data_by_name(sr.get_output_data_names())
        sr.linearize(indata, force_all=True)
