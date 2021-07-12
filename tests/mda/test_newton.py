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
#        :author: Charlie Vanaret, Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import division, unicode_literals

import os
import unittest

from numpy import array, float64, linalg, ones

from gemseo.mda.newton import MDANewtonRaphson, MDAQuasiNewton
from gemseo.problems.sellar.sellar import (
    X_LOCAL,
    X_SHARED,
    Y_1,
    Y_2,
    Sellar1,
    Sellar2,
    SellarSystem,
)
from gemseo.problems.sobieski.wrappers import (
    SobieskiAerodynamics,
    SobieskiMission,
    SobieskiPropulsion,
    SobieskiStructure,
)

from .test_gauss_seidel import SelfCoupledDisc

DIRNAME = os.path.dirname(__file__)


class TestNewton(unittest.TestCase):
    """Test the Newton-Raphson MDA."""

    def test_raphson_sobieski(self):
        """Test the execution of Gauss-Seidel on Sobieski."""
        disciplines = [
            SobieskiAerodynamics(),
            SobieskiStructure(),
            SobieskiPropulsion(),
            SobieskiMission(),
        ]
        mda = MDANewtonRaphson(disciplines)
        mda.reset_history_each_run = True
        mda.execute()
        assert mda.residual_history[-1][0] < 1e-6

        mda.warm_start = True
        mda.execute({"x_1": mda.default_inputs["x_1"] + 1.0e-2})
        assert mda.residual_history[-1][0] < 1e-6

        self.assertRaises(ValueError, MDANewtonRaphson, disciplines, relax_factor=1.1)
        self.assertRaises(ValueError, MDANewtonRaphson, disciplines, relax_factor=-0.1)

    @staticmethod
    def get_sellar_initial():
        """Generate initial solution."""
        x_local = array([0.0], dtype=float64)
        x_shared = array([1.0, 0.0], dtype=float64)
        y_0 = ones(1, dtype=float64)
        y_1 = ones(1, dtype=float64)
        return x_local, x_shared, y_0, y_1

    def test_wrong_name(self):
        disciplines = [
            SobieskiAerodynamics(),
            SobieskiStructure(),
            SobieskiPropulsion(),
            SobieskiMission(),
        ]

        self.assertRaises(ValueError, MDAQuasiNewton, disciplines, method="FAIL")

    @staticmethod
    def get_sellar_initial_input_data():
        """Build dictionary with initial solution."""
        x_local, x_shared, y_0, y_1 = TestNewton.get_sellar_initial()
        return {X_LOCAL: x_local, X_SHARED: x_shared, Y_1: y_0, Y_2: y_1}

    def test_raphson_sellar(self):
        """Test the execution of Newton on Sobieski."""
        disciplines = [Sellar1(), Sellar2()]
        mda = MDANewtonRaphson(disciplines)
        mda.execute()

        assert mda.residual_history[-1][0] < 1e-6

        y_ref = array([0.80004953, 1.79981434])
        y_opt = array([mda.local_data[Y_1][0].real, mda.local_data[Y_2][0].real])
        assert linalg.norm(y_ref - y_opt) / linalg.norm(y_ref) < 1e-4

    # =========================================================================
    #     def test_newton_sellar_parallel(self):
    #         """
    #         Compare Newton and Gauss-Seidel MDA
    #         """
    #         indata = TestNewton.get_sellar_initial_input_data()
    #
    #         mda_1 = MDANewtonRaphson(self.sellar_coupling_structure, n_processes=2)
    #         out1 = mda_1.execute(indata)
    #
    #         mda_2 = MDANewtonRaphson(self.sellar_coupling_structure, n_processes=4)
    #         out2 = mda_2.execute(indata)
    #
    #         for key, value1 in out1.items():
    #             nv1 = linalg.norm(value1)
    #             if nv1 > 1e-14:
    #                 assert linalg.norm(
    #                     out2[key] - value1) / linalg.norm(value1) < 1e-2
    #             else:
    #                 assert linalg.norm(out2[key] - value1) < 1e-2
    # =========================================================================

    def test_broyden_sellar(self):
        """Test the execution of quasi-Newton on Sellar."""
        disciplines = [Sellar1(), Sellar2()]
        mda = MDAQuasiNewton(disciplines, method=MDAQuasiNewton.BROYDEN1)
        mda.reset_history_each_run = True
        mda.execute()
        assert mda.residual_history[-1][0] < 1e-5

        y_ref = array([0.80004953, 1.79981434])
        y_opt = array([mda.local_data[Y_1][0].real, mda.local_data[Y_2][0].real])
        assert linalg.norm(y_ref - y_opt) / linalg.norm(y_ref) < 1e-3

        mda.warm_start = True
        mda.execute({X_SHARED: mda.default_inputs[X_SHARED] + 0.1})

    def test_hybrid_sellar(self):
        """Test the execution of quasi-Newton on Sellar."""
        disciplines = [Sellar1(), Sellar2()]
        mda = MDAQuasiNewton(
            disciplines, method=MDAQuasiNewton.HYBRID, use_gradient=True
        )

        mda.execute()

        y_ref = array([0.80004953, 1.79981434])
        y_opt = array([mda.local_data[Y_1][0].real, mda.local_data[Y_2][0].real])
        assert linalg.norm(y_ref - y_opt) / linalg.norm(y_ref) < 1e-4

    def test_lm_sellar(self):
        """Test the execution of quasi-Newton on Sellar."""
        disciplines = [Sellar1(), Sellar2()]
        mda = MDAQuasiNewton(
            disciplines, method=MDAQuasiNewton.LEVENBERG_MARQUARDT, use_gradient=True
        )
        mda.execute()

        y_ref = array([0.80004953, 1.79981434])
        y_opt = array([mda.local_data[Y_1][0].real, mda.local_data[Y_2][0].real])
        assert linalg.norm(y_ref - y_opt) / linalg.norm(y_ref) < 1e-4

    def test_dfsane_sellar(self):
        """Test the execution of quasi-Newton on Sellar."""
        disciplines = [Sellar1(), Sellar2()]
        mda = MDAQuasiNewton(disciplines, method=MDAQuasiNewton.DF_SANE)
        mda.execute()

        y_ref = array([0.80004953, 1.79981434])
        y_opt = array([mda.local_data[Y_1][0].real, mda.local_data[Y_2][0].real])
        assert linalg.norm(y_ref - y_opt) / linalg.norm(y_ref) < 1e-3

    def test_quasi_newton_fake_method(self):
        """Test the execution of quasi-Newton with fake method."""
        with self.assertRaises(Exception):
            MDAQuasiNewton(self.sellar_coupling_structure, method="space_cowboy")

    def test_broyden_sellar2(self):
        """Test the execution of quasi-Newton on Sellar."""
        disciplines = [Sellar1(), SellarSystem()]
        mda = MDAQuasiNewton(disciplines, method=MDAQuasiNewton.BROYDEN1)
        mda.reset_history_each_run = True
        mda.execute()

    def test_self_coupled(self):
        sc_disc = SelfCoupledDisc()
        mda = MDAQuasiNewton([sc_disc], tolerance=1e-14, max_mda_iter=40)
        out = mda.execute()
        assert abs(out["y"] - 2.0 / 3.0) < 1e-6
