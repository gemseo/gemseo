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
#        :author: Francois Gallard, refactoring
#    OTHER AUTHORS   - MACROSCOPIC CHANGES


from __future__ import division, unicode_literals

from unittest import TestCase

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.problems.analytical.power_2 import Power2


class TestOptLib(TestCase):
    """"""

    OPT_LIB_NAME = "ScipyOpt"

    def test_handles(self):
        factory = OptimizersFactory()
        if factory.is_available(self.OPT_LIB_NAME):
            lib = factory.create(self.OPT_LIB_NAME)
        else:
            raise ImportError("Scipy is not available")
        assert not lib.algorithm_handles_eqcstr("L-BFGS-B")
        assert not lib.algorithm_handles_ineqcstr("L-BFGS-B")
        assert lib.algorithm_handles_eqcstr("SLSQP")
        assert lib.algorithm_handles_ineqcstr("SLSQP")

        power = Power2()
        assert not lib.is_algorithm_suited(lib.lib_dict["L-BFGS-B"], power)

        self.assertRaises(ValueError, lib._pre_run, power, "SLSQP")

        power.constraints = power.constraints[0:1]

        assert lib.is_algorithm_suited(lib.lib_dict["SLSQP"], power)
        # With only inequality constraints
        assert not lib.is_algorithm_suited(lib.lib_dict["L-BFGS-B"], power)

        self.assertRaises(
            ValueError, lib._check_constraints_handling, "L-BFGS-B", power
        )

        self.assertRaises(KeyError, lib.algorithm_handles_eqcstr, "TOTO")

        algo_name = "L-BFGS-B"
        del lib.lib_dict[algo_name][lib.HANDLE_INEQ_CONS]
        assert not lib.algorithm_handles_ineqcstr("L-BFGS-B")
