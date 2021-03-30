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
from builtins import str
from os.path import abspath, dirname, join

from future import standard_library
from numpy import array

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.core.discipline import MDODiscipline
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.problems.sobieski.chains import SobieskiMDAGaussSeidel
from gemseo.third_party.junitxmlreq import link_to

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class TestMDAGaussSeidel(unittest.TestCase):
    """Test the Gauss-Seidel MDA"""

    @staticmethod
    @link_to("Req-MDO-9", "Req-MDO-9.3")
    def test_sobieski():
        """Test the execution of Gauss-Seidel on Sobieski"""
        mda = SobieskiMDAGaussSeidel(tolerance=1e-12, max_mda_iter=30)
        mda.default_inputs["x_shared"] += 0.1
        mda.execute()
        filename = "GaussSeidel.pdf"
        mda.plot_residual_history(False, True, filename=filename)
        mda.default_inputs["x_shared"] += 0.1
        mda.warm_start = True
        mda.execute()

        assert mda.residual_history[-1][0] < 1e-4

        filename = "SobieskiMDAGS_residual_history.pdf"
        mda.plot_residual_history(save=True, filename=filename)
        assert os.path.exists(filename)
        os.remove(filename)

    def test_expected_workflow(self):
        """Test MDA GaussSeidel workflow should be disciplines sequence"""
        disc1 = MDODiscipline()
        disc2 = MDODiscipline()
        disc3 = MDODiscipline()
        disciplines = [disc1, disc2, disc3]

        mda = MDAGaussSeidel(disciplines)
        expected = "{MDAGaussSeidel(None), [MDODiscipline(None), MDODiscipline(None), MDODiscipline(None), ], }"
        self.assertEqual(str(mda.get_expected_workflow()), expected)

    def test_self_coupled(self):
        for plus_y in [False, True]:
            sc_disc = SelfCoupledDisc(plus_y)
            mda = MDAGaussSeidel([sc_disc], tolerance=1e-14, max_mda_iter=40)
            _ = mda.execute()
            # assert abs(out["y"] - 2. / 3.) < 1e-6

            mda.add_differentiated_inputs(["x"])
            mda.add_differentiated_outputs(["o"])
            jac1 = mda.linearize()

            mda.set_jacobian_approximation()
            mda.cache.clear()
            jac2 = mda.linearize()
            assert abs(jac1["o"]["x"][0, 0] - jac2["o"]["x"][0, 0]) < 1e-3


class SelfCoupledDisc(MDODiscipline):
    def __init__(self, plus_y=False):
        MDODiscipline.__init__(self)
        self.input_grammar.initialize_from_data_names(["y", "x"])
        self.output_grammar.initialize_from_data_names(["y", "o"])
        self.default_inputs["y"] = array([0.25])
        self.default_inputs["x"] = array([0.0])
        self.coeff = 1.0
        if not plus_y:
            self.coeff = -1.0

    def _run(self):

        self.local_data["y"] = (
            1.0 + self.coeff * 0.5 * self.local_data["y"] + self.local_data["x"]
        )
        self.local_data["o"] = self.local_data["y"] + self.local_data["x"]

    def _compute_jacobian(self, inputs=None, outputs=None):
        self.jac = {}
        self.jac["y"] = {"y": self.coeff * array([[0.5]]), "x": array([[1.0]])}
        self.jac["o"] = {"y": array([[1.0]]), "x": array([[1.0]])}
