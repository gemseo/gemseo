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
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest

import numpy as np

from gemseo.core.discipline import MDODiscipline
from gemseo.core.jacobian_assembly import JacobianAssembly
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.mda import MDA
from gemseo.problems.sellar.sellar import Sellar1, Sellar2, SellarSystem

DIRNAME = os.path.dirname(__file__)


class TestMDA(unittest.TestCase):
    """Test of the MDA abstract class."""

    @classmethod
    def setUpClass(cls):
        # initialize disciplines, MDA and input data
        cls.disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        cls.mda_sellar = MDAGaussSeidel(cls.disciplines)
        cls.input_data_sellar = cls.get_sellar_initial_input_data()

    @staticmethod
    def get_sellar_initial():
        """Generate initial solution."""
        x_local = np.array([0.0], dtype=np.float64)
        x_shared = np.array([1.0, 0.0], dtype=np.float64)
        y_0 = np.ones(1, dtype=np.complex128)
        y_1 = np.ones(1, dtype=np.complex128)
        return x_local, x_shared, y_0, y_1

    @classmethod
    def get_sellar_initial_input_data(cls):
        """Build dictionary with initial solution."""
        x_local, x_shared, y_0, y_1 = TestMDA.get_sellar_initial()
        return {"x_local": x_local, "x_shared": x_shared, "y_0": y_0, "y_1": y_1}

    def test_reset(self):
        """Test that the MDA successfully resets its disciplines after their
        executions."""
        for discipline in self.disciplines:
            discipline.execute(self.input_data_sellar)
        for discipline in self.disciplines:
            self.assertEqual(discipline.status, MDODiscipline.STATUS_DONE)

        self.mda_sellar.reset_statuses_for_run()
        for discipline in self.disciplines:
            self.assertEqual(discipline.status, MDODiscipline.STATUS_PENDING)

    def test_input_couplings(self):
        mda = MDA([Sellar1()])
        assert len(mda._current_input_couplings()) == 0

    def test_jacobian(self):
        """Check the Jacobian computation."""
        self.mda_sellar.use_lu_fact = True
        self.mda_sellar.matrix_type = JacobianAssembly.LINEAR_OPERATOR
        self.assertRaises(
            ValueError,
            self.mda_sellar.linearize,
            self.input_data_sellar,
            force_all=True,
        )
        self.mda_sellar.use_lu_fact = False
        self.mda_sellar.linearize(self.input_data_sellar)
        self.assertEqual(self.mda_sellar.jac, {})

        self.mda_sellar._differentiated_inputs = None
        self.mda_sellar._differentiated_outputs = None

        self.mda_sellar.linearize(self.input_data_sellar)

    def test_expected_workflow(self):
        """"""
        expected = (
            "{MDAGaussSeidel(None), [Sellar1(None), Sellar2(None), "
            "SellarSystem(None), ], }"
        )
        self.assertEqual(str(self.mda_sellar.get_expected_workflow()), expected)

    def test_warm_start(self):
        # Check that the warm start works even at first execution
        mda_sellar = MDAGaussSeidel(self.disciplines)
        mda_sellar.warm_start = True
        mda_sellar.execute()
