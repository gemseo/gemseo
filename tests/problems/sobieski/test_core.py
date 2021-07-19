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
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import division, unicode_literals

import unittest

from numpy import array, complex128, float64, ones, zeros
from numpy.linalg import norm

from gemseo.problems.sobieski.core import SobieskiProblem


class TestSobieskiCore(unittest.TestCase):
    """"""

    def setUp(self):
        """At creation of unittest, initiate a sobieski problem class."""
        self.problem = SobieskiProblem()
        self.dtype = float64

    def __relative_norm(self, x, x_ref):
        return norm(x - x_ref) / norm(x_ref)

    def __norm(self, x):
        return norm(x)

    def test_get_optimum_range(self):
        """"""
        reference_range = array([3963.98])
        self.assertEqual(self.problem.get_sobieski_optimum_range(), reference_range)

    def test_get_sob_cstr(self):
        gs = self.problem.get_default_inputs(["g_1", "g_2", "g_3"])
        c1 = self.problem.get_sobieski_constraints(
            gs["g_1"], gs["g_2"], gs["g_3"], true_cstr=True
        )
        self.assertEqual(len(c1), 10)
        c2 = self.problem.get_sobieski_constraints(
            gs["g_1"], gs["g_2"], gs["g_3"], true_cstr=False
        )
        self.assertEqual(len(c2), 12)

    def test_normalize(self):
        """"""
        x0 = self.problem.get_default_x0()
        x0_adim = self.problem.normalize_inputs(x0)
        x0_dim = self.problem.unnormalize_inputs(x0_adim)
        self.assertEqual(self.__relative_norm(x0_dim, x0), 0.0)

    def test_design_space(self):
        """"""
        ds = SobieskiProblem("complex128").read_design_space()
        for val in ds.get_current_x_dict().values():
            assert val.dtype == complex128
        ds = SobieskiProblem().read_design_space()
        for val in ds.get_current_x_dict().values():
            assert val.dtype == float64

    def test_constants(self):
        """"""
        cref = zeros(5)
        # Constants of problem
        cref[0] = 2000.0  # minimum fuel weight
        cref[1] = 25000.0  # miscellaneous weight
        cref[2] = 6.0  # Maximum load factor
        cref[3] = 4360.0  # Engine weight reference
        cref[4] = 0.01375  # Minimum drag coefficient
        c = self.problem.default_constants()
        self.assertAlmostEqual(self.__relative_norm(c, cref), 0.0, places=6)

    def test_init(self):
        """"""
        cmod = zeros(5)
        # Constants of problem
        cmod[0] = 2000.0  # minimum fuel weight
        cmod[1] = 25000.0  # miscellaneous weight
        cmod[2] = 6.0  # Maximum load factor
        cmod[3] = 4360.0  # Engine weight reference
        cmod[4] = 0.01375  # Minimum drag coefficient
        problem = SobieskiProblem("complex128")
        self.assertAlmostEqual(cmod.all(), problem.default_constants().all())

        self.assertRaises(Exception, SobieskiProblem, "Toto")

    def test_get_default_inputs_feasible(self):
        """"""
        _ = self.problem.get_default_inputs_feasible()
        _ = self.problem.get_x0_feasible()
        indata = self.problem.get_default_inputs_feasible("x_1")
        refdata = self.problem.get_x0_feasible("x_1")
        self.assertAlmostEqual(indata["x_1"].all(), refdata.all(), 9)

    def test_get_random_inputs(self):
        self.problem.get_random_input(names=None, seed=1)
        assert len(self.problem.get_random_input(names=["x_1", "x_2"], seed=1)) == 2

    def test_get_bounds(self):
        """"""
        lb_ref = array((0.1, 0.75, 0.75, 0.1, 0.01, 30000.0, 1.4, 2.5, 40.0, 500.0))
        ub_ref = array((0.4, 1.25, 1.25, 1.0, 0.09, 60000.0, 1.8, 8.5, 70.0, 1500.0))
        u_b, l_b = self.problem.get_sobieski_bounds()
        self.assertAlmostEqual(self.__relative_norm(l_b, lb_ref), 0.0, places=6)
        self.assertAlmostEqual(self.__relative_norm(u_b, ub_ref), 0.0, places=6)

    def test_get_bounds_tuple(self):
        """"""
        bounds_tuple_ref = (
            (0.1, 0.4),
            (0.75, 1.25),
            (0.75, 1.25),
            (0.1, 1),
            (0.01, 0.09),
            (30000.0, 60000.0),
            (1.4, 1.8),
            (2.5, 8.5),
            (40.0, 70.0),
            (500.0, 1500.0),
        )
        bounds_tuple = self.problem.base.get_sobieski_bounds_tuple()
        for i in range(len(bounds_tuple)):
            self.assertAlmostEqual(
                self.__relative_norm(bounds_tuple[i][0], bounds_tuple_ref[i][0]),
                0.0,
                places=6,
            )
            self.assertAlmostEqual(
                self.__relative_norm(bounds_tuple[i][1], bounds_tuple_ref[i][1]),
                0.0,
                places=6,
            )

    def test_poly_approx(self):
        """test polynomial function approximation."""
        # Reference value from octave computation for polyApprox function
        ff_reference = 1.02046767  # Octave computation
        mach_ini = 1.6
        h_ini = 45000.0
        t_ini = 0.5
        s = array([mach_ini, h_ini, t_ini])
        snew = array([1.5, 50000.0, 0.75], dtype=self.dtype)
        flag = array([2, 4, 2], dtype=self.dtype)
        bound = array([0.25, 0.25, 0.25], dtype=self.dtype)
        ff = self.problem.base.poly_approx(s, snew, flag, bound)
        self.assertAlmostEqual(ff, ff_reference, places=6)

    def test_weight(self):
        """blackbox_structure function test."""
        # Reference value from octaves computation for blackbox_structure
        # function
        y_1_reference = array(
            [3.23358440e04, 7.30620262e03, 1.00000000e00], dtype=self.dtype
        )
        y_14_reference = array([32335.84397838, 7306.20262124], dtype=self.dtype)
        y_12_reference = array([3.23358440e04, 1.00000000e00], dtype=self.dtype)
        g_1_reference = array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=self.dtype)

        i0 = self.problem.get_default_x0()
        i0 = array(i0, dtype=self.dtype)
        i0[0] = i0[0]
        #       [0.25 + 1j * h, 1., 1., 0.5, 0.05, 45000.0, 1.6, 5.5, 55.0, 1000.0]
        x_1 = i0[:2]
        x_shared = i0[4:]
        #         - x_1(0) : wing taper ratio
        #         - x_1(1) : wingbox x-sectional area as poly. funct
        #         - Z(0) : thickness/chord ratio
        #         - Z(1) : altitude
        #         - Z(2) : Mach
        #         - Z(3) : aspect ratio
        #         - Z(4) : wing sweep
        #         - Z(5) : wing surface area
        y_21 = ones(1, dtype=self.dtype)
        y_31 = ones(1, dtype=self.dtype)

        y_1, _, y_12, y_14, g_1 = self.problem.blackbox_structure(
            x_shared, y_21, y_31, x_1, true_cstr=True
        )

        # Check results regression
        self.assertAlmostEqual(self.__relative_norm(y_1, y_1_reference), 0.0, places=2)
        self.assertAlmostEqual(
            self.__relative_norm(y_12, y_12_reference), 0.0, places=2
        )
        self.assertAlmostEqual(
            self.__relative_norm(y_14, y_14_reference), 0.0, places=2
        )
        self.assertAlmostEqual(self.__relative_norm(g_1, g_1_reference), 0.0, places=2)

    def test_dragpolar(self):
        """blackbox_aerodynamics function test."""
        # Reference value from octave computation for blackbox_structure
        # function
        y_2_reference = array([3.23358440e04, 1.25620121e04, 2.57409751e00])
        y_21_reference = array([32335.84397838])
        y_23_reference = array([12562.07000284])
        y_24_reference = array([2.57408564])
        g_2_reference = array([1.0])

        i0 = array(self.problem.get_default_x0(), dtype=self.dtype)
        x_1 = i0[:2]
        x_2 = array([i0[2]], dtype=self.dtype)
        x_shared = i0[4:]
        #         - x_1(0) : wing taper ratio
        #         - x_1(1) : wingbox x-sectional area as poly. funct
        #         - Z(0) : thickness/chord ratio
        #         - Z(1) : altitude
        #         - Z(2) : Mach
        #         - Z(3) : aspect ratio
        #         - Z(4) : wing sweep
        #         - Z(5) : wing surface area
        #         y_12 = ones((2),dtype=self.dtype)
        y_21 = ones(1, dtype=self.dtype)
        y_31 = ones(1, dtype=self.dtype)
        y_32 = ones(1, dtype=self.dtype)

        # Preserve initial values for polynomial calculations

        _, _, y_12, _, _ = self.problem.blackbox_structure(
            x_shared, y_21, y_31, x_1, true_cstr=True
        )
        y_2, y_21, y_23, y_24, g_2 = self.problem.blackbox_aerodynamics(
            x_shared, y_12, y_32, x_2, true_cstr=True
        )
        # Check results regression
        self.assertAlmostEqual(self.__relative_norm(y_2, y_2_reference), 0.0, places=2)
        self.assertAlmostEqual(
            self.__relative_norm(y_21, y_21_reference), 0.0, places=3
        )
        self.assertAlmostEqual(
            self.__relative_norm(y_23, y_23_reference), 0.0, places=2
        )
        self.assertAlmostEqual(
            self.__relative_norm(y_24, y_24_reference), 0.0, places=6
        )
        self.assertAlmostEqual(self.__relative_norm(g_2, g_2_reference), 0.0, places=6)

    def test_power(self):
        """blackbox_propulsion function test."""
        # Reference value from octave computation for blackbox_structure
        # function
        y_3_reference = array([1.10754577e00, 6.55568459e03, 5.17959175e-01])
        y_34_reference = array([1.10754577])
        y_31_reference = array([6555.68459235])
        y_32_reference = array([0.51796156])
        g_3_reference = array([0.51796156, 1.0, 0.16206032])

        i0 = array(self.problem.get_default_x0(), dtype=self.dtype)

        x_1 = i0[:2]
        x_2 = array([i0[2]], dtype=self.dtype)
        x_3 = array([i0[3]], dtype=self.dtype)
        x_shared = i0[4:]
        #         - x_1(0) : wing taper ratio
        #         - x_1(1) : wingbox x-sectional area as poly. funct
        #         - Z(0) : thickness/chord ratio
        #         - Z(1) : altitude
        #         - Z(2) : Mach
        #         - Z(3) : aspect ratio
        #         - Z(4) : wing sweep
        #         - Z(5) : wing surface area
        #         y_12 = ones((2),dtype=self.dtype)
        y_21 = ones(1, dtype=self.dtype)
        y_31 = ones(1, dtype=self.dtype)
        y_32 = ones(1, dtype=self.dtype)

        # Preserve initial values for polynomial calculations

        _, _, y_12, _, _ = self.problem.blackbox_structure(
            x_shared, y_21, y_31, x_1, true_cstr=True
        )
        _, y_21, y_23, _, _ = self.problem.blackbox_aerodynamics(
            x_shared, y_12, y_32, x_2, true_cstr=True
        )
        y_3, y_34, y_31, y_32, g_3 = self.problem.blackbox_propulsion(
            x_shared, y_23, x_3, true_cstr=True
        )

        # Check results regression

        self.assertAlmostEqual(self.__relative_norm(y_3, y_3_reference), 0.0, places=3)
        self.assertAlmostEqual(
            self.__relative_norm(y_31, y_31_reference), 0.0, places=3
        )
        self.assertAlmostEqual(
            self.__relative_norm(y_32, y_32_reference), 0.0, places=6
        )
        self.assertAlmostEqual(
            self.__relative_norm(y_34, y_34_reference), 0.0, places=6
        )
        self.assertAlmostEqual(self.__relative_norm(g_3, g_3_reference), 0.0, places=6)

    def test_range(self):
        """blackbox_mission function test."""
        # Reference value from octave computation for blackbox_structure
        # function
        y_4_reference = array([545.88197472055879])

        # return array((0.25, 1., 1., 0.5, 0.05, 45000.0, 1.6, 5.5, 55.0,
        # 1000.0))
        i0 = array(self.problem.get_default_x0(), dtype=self.dtype)

        x_1 = i0[:2]
        x_2 = array([i0[2]], dtype=self.dtype)
        x_3 = array([i0[3]], dtype=self.dtype)
        x_shared = i0[4:]
        #         - x_1(0) : wing taper ratio
        #         - x_1(1) : wingbox x-sectional area as poly. funct
        #         - Z(0) : thickness/chord ratio
        #         - Z(1) : altitude
        #         - Z(2) : Mach
        #         - Z(3) : aspect ratio
        #         - Z(4) : wing sweep
        #         - Z(5) : wing surface area
        #         y_12 = ones((2),dtype=self.dtype)
        y_21 = ones(1, dtype=self.dtype)
        y_31 = ones(1, dtype=self.dtype)
        y_32 = ones(1, dtype=self.dtype)

        # Preserve initial values for polynomial calculations
        _, _, y_12, y_14, _ = self.problem.blackbox_structure(x_shared, y_21, y_31, x_1)

        _, y_21, y_23, y_24, _ = self.problem.blackbox_aerodynamics(
            x_shared, y_12, y_32, x_2
        )

        _, y_34, y_31, y_32, _ = self.problem.blackbox_propulsion(x_shared, y_23, x_3)

        y_4 = self.problem.blackbox_mission(x_shared, y_14, y_24, y_34)

        self.assertAlmostEqual(y_4[0], y_4_reference[0], places=3)

    def test_range_h35000(self):
        """blackbox_mission function test."""
        # Reference value from octave computation for one MDA loop
        # function
        y_4_reference = array([352.508])

        # return array((0.25, 1., 1., 0.5, 0.05, 45000.0, 1.6, 5.5, 55.0,
        # 1000.0))
        i0 = array(self.problem.get_default_x0(), dtype=self.dtype)

        x_1 = i0[:2]
        x_2 = array([i0[2]], dtype=self.dtype)
        x_3 = array([i0[3]], dtype=self.dtype)
        x_shared = i0[4:]
        x_shared[1] = 35000
        #         - x_1(0) : wing taper ratio
        #         - x_1(1) : wingbox x-sectional area as poly. funct
        #         - Z(0) : thickness/chord ratio
        #         - Z(1) : altitude
        #         - Z(2) : Mach
        #         - Z(3) : aspect ratio
        #         - Z(4) : wing sweep
        #         - Z(5) : wing surface area
        #         y_12 = ones((2),dtype=self.dtype)
        y_21 = ones(1, dtype=self.dtype)
        y_31 = ones(1, dtype=self.dtype)
        y_32 = ones(1, dtype=self.dtype)

        # Preserve initial values for polynomial calculations
        _, _, y_12, y_14, _ = self.problem.blackbox_structure(x_shared, y_21, y_31, x_1)

        _, y_21, y_23, y_24, _ = self.problem.blackbox_aerodynamics(
            x_shared, y_12, y_32, x_2
        )

        _, y_34, y_31, y_32, _ = self.problem.blackbox_propulsion(x_shared, y_23, x_3)

        y_4 = self.problem.blackbox_mission(x_shared, y_14, y_24, y_34)
        self.assertAlmostEqual(y_4[0], y_4_reference[0], places=2)

    def test_optimum_gs(self):
        """MDA analysis of the optimum sample from Sobieski and check range value."""

        # Reference value from octave computation for blackbox_structure function
        #         y_4_reference = self.problem.get_sobieski_optimum_range()

        x_optimum = self.problem.get_sobieski_optimum()

        self.problem.systemanalysis_gauss_seidel(x_optimum)

    def test_constraints(self):
        """MDA analysis of the optimum sample from Sobieski and check range value."""

        # Reference value from octave computation for blackbox_structure function
        #         y_4_reference = self.problem.get_sobieski_optimum_range()

        x_optimum = self.problem.get_sobieski_optimum()

        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            g_1,
            g_2,
            g_3,
        ) = self.problem.systemanalysis_gauss_seidel(x_optimum, true_cstr=True)
        constraints_values = self.problem.get_sobieski_constraints(
            g_1, g_2, g_3, true_cstr=False
        )
        for i in range(constraints_values.shape[0]):
            self.assertLessEqual(constraints_values[i].real, 0.0)

    def test_ineq_constraints(self):
        """"""
        #         y_4_reference = self.problem.get_sobieski_optimum_range()

        x_optimum = self.problem.get_sobieski_optimum()

        g_1, g_2, g_3 = self.problem.get_constraints(x_optimum, true_cstr=False)
        for g in (g_1, g_2, g_3):
            for i in range(g.shape[0]):
                self.assertLessEqual(g[i], 0.0)

    def test_x0_gs(self):
        """MDA analysis of the initial sample from Sobieski and check range value."""

        # Reference value from octave computation for MDA
        # function
        y_4_reference = array([535.79388428])

        x0 = array(self.problem.get_default_x0(), dtype=self.dtype)

        (
            _,
            _,
            _,
            y_4,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.problem.systemanalysis_gauss_seidel(x0)
        self.assertAlmostEqual(int(y_4[0].real), int(y_4_reference[0]), places=0)

    def test_h35000(self):
        """MDA analysis of the initial sample from Sobieski with modified altitude to
        test conditions on altitude in code."""

        # Reference value from octave computation for MDA
        # function
        y_4_reference = array([340.738])

        x0 = array(self.problem.get_default_x0(), dtype=self.dtype)
        x0[5] = 3.50000000e04
        (
            _,
            _,
            _,
            y_4,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.problem.systemanalysis_gauss_seidel(x0)
        self.assertAlmostEqual(int(y_4[0].real), int(y_4_reference[0]), places=0)

    def test_x0_optimum(self):
        """MDA analysis of the initial sample from Sobieski and check range value."""

        # Reference value from octave computation for blackbox_structure
        # function
        y_4_ref = 3963.19894068
        x0 = array(self.problem.get_sobieski_optimum(), dtype=self.dtype)

        (
            _,
            _,
            _,
            y_4,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            g_1,
            g_2,
            g_3,
        ) = self.problem.systemanalysis_gauss_seidel(x0, true_cstr=True)
        constraints_values = self.problem.get_sobieski_constraints(g_1, g_2, g_3)
        self.assertLessEqual(constraints_values.all(), 1e-6)
        self.assertAlmostEqual(y_4[0].real, y_4_ref, places=0)


#
