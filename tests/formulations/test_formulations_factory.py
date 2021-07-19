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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import division, unicode_literals

import unittest

import numpy as np

from gemseo.algos.design_space import DesignSpace
from gemseo.formulations.formulations_factory import MDOFormulationsFactory
from gemseo.problems.sellar.sellar import Sellar1, Sellar2, SellarSystem


class TestFormulationsFactory(unittest.TestCase):
    """"""

    def setUp(self):
        """"""
        if not hasattr(self, "factory"):
            self.factory = MDOFormulationsFactory()

    def test_list_formulations(self):
        """"""
        f_list = self.factory.formulations
        known_form = ["MDF", "IDF", "DisciplinaryOpt", "BiLevel"]
        for form in known_form:
            assert form in f_list

    @staticmethod
    def get_xzy():
        """Generate initial solution."""
        x_local = np.array([0.0], dtype=np.float64)
        x_shared = np.array([1.0, 0.0], dtype=np.float64)
        y_0 = np.zeros(1, dtype=np.complex128)
        y_1 = np.zeros(1, dtype=np.complex128)
        return x_local, x_shared, y_0, y_1

    @staticmethod
    def get_current_x():
        """Build dictionary with initial solution."""
        x_local, x_shared, y_0, y_1 = TestFormulationsFactory.get_xzy()
        return {"x_local": x_local, "x_shared": x_shared, "y_0": y_0, "y_1": y_1}

    def test_create(self):
        """"""
        self.assertRaises(
            Exception,
            self.factory.create,
            "toto is not a formulation",
            None,
            None,
            None,
        )
        disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        objective_name = "obj"
        design_space = DesignSpace()
        design_space.add_variable("x", 3)
        self.factory.create("MDF", disciplines, objective_name, design_space)

        self.assertRaises(Exception, self.factory.create, Sellar1, None, None, None)

    def test_isavailable(self):
        self.factory.is_available("MDF")
