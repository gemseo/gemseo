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

import unittest
from builtins import str
from os.path import abspath, dirname, join

from future import standard_library
from numpy import array, complex128, float64, ones

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.core.discipline import MDODiscipline
from gemseo.mda.jacobi import MDAJacobi
from gemseo.problems.sellar.sellar import Sellar1, Sellar2, SellarSystem
from gemseo.problems.sobieski.chains import SobieskiMDAJacobi
from gemseo.third_party.junitxmlreq import link_to

from .test_gauss_seidel import SelfCoupledDisc

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class TestMDAJacobi(unittest.TestCase):
    """Tests of Jacobi MDA"""

    @staticmethod
    @link_to("Req-MDO-9")
    def test_jacobi_sobieski():
        """Test the execution of Jacobi on Sobieski"""
        mda = SobieskiMDAJacobi()
        mda.execute()
        mda.default_inputs["x_shared"] += 0.02
        mda.warm_start = True
        mda.execute()
        assert mda.residual_history[-1][0] < 1e-4

    def test_secant_acceleration(self):
        tolerance = 1e-12
        mda = SobieskiMDAJacobi(tolerance=tolerance, max_mda_iter=30, acceleration=None)
        mda.execute()
        #         mda.plot_residual_history(False, True, filename="Jacobi.pdf")
        nit1 = mda.residual_history[-1][-1]

        mda = SobieskiMDAJacobi(
            tolerance=tolerance, max_mda_iter=30, acceleration=mda.SECANT_ACCELERATION
        )
        mda.execute()
        filename = "Jacobi_secant.pdf"
        mda.plot_residual_history(False, True, filename=filename)
        nit2 = mda.residual_history[-1][-1]

        mda = SobieskiMDAJacobi(
            tolerance=tolerance, max_mda_iter=30, acceleration=mda.M2D_ACCELERATION
        )
        mda.execute()
        filename = "Jacobi_m2d.pdf"
        mda.plot_residual_history(False, True, filename=filename)
        nit3 = mda.residual_history[-1][-1]
        assert nit2 < nit1
        assert nit3 < nit1
        assert nit3 < nit2

    @staticmethod
    @link_to("Req-MDO-9", "Req-MDO-9.2")
    def test_mda_jacobi_parallel():
        """Comparison of Jacobi on Sobieski problem: 1 and 5 processes"""
        mda_seq = SobieskiMDAJacobi()
        outdata_seq = mda_seq.execute()

        mda_parallel = SobieskiMDAJacobi(n_processes=4)
        mda_parallel.reset_statuses_for_run()
        outdata_parallel = mda_parallel.execute()

        for key, value in outdata_seq.items():
            assert array(outdata_parallel[key] == value).all()

    @staticmethod
    def get_sellar_initial():
        """Generate initial solution"""
        x_local = array([0.0], dtype=float64)
        x_shared = array([1.97763888, 0.0], dtype=float64)
        y_0 = ones((1), dtype=complex128)
        y_1 = ones((1), dtype=complex128)
        return x_local, x_shared, y_0, y_1

    @staticmethod
    def get_sellar_initial_input_data():
        """Build dictionary with initial solution"""
        x_local, x_shared, y_0, y_1 = TestMDAJacobi.get_sellar_initial()
        return {"x_local": x_local, "x_shared": x_shared, "y_0": y_0, "y_1": y_1}

    def test_jacobi_sellar(self):
        """Test the execution of Jacobi on Sobieski"""
        disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        mda = MDAJacobi(disciplines)
        mda.execute()

        assert mda.residual_history[-1][0] < 1e-4

    def test_expected_workflow(self):
        """Test MDAJacobi workflow should be list of one tuple of disciplines
        (meaning parallel execution)


        """
        disc1 = MDODiscipline()
        disc2 = MDODiscipline()
        disc3 = MDODiscipline()
        disciplines = [disc1, disc2, disc3]

        mda = MDAJacobi(disciplines, n_processes=1)
        expected = "{MDAJacobi(None), [MDODiscipline(None), MDODiscipline(None), MDODiscipline(None), ], }"
        self.assertEqual(str(mda.get_expected_workflow()), expected)

        mda = MDAJacobi(disciplines, n_processes=2)
        expected = "{MDAJacobi(None), (MDODiscipline(None), MDODiscipline(None), MDODiscipline(None), ), }"
        self.assertEqual(str(mda.get_expected_workflow()), expected)

    def test_self_coupled(self):
        sc_disc = SelfCoupledDisc()
        mda = MDAJacobi([sc_disc], tolerance=1e-14, max_mda_iter=40)
        out = mda.execute()
        assert abs(out["y"] - 2.0 / 3.0) < 1e-6
