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
#        :author: Matthias De Lozzo
from __future__ import division, unicode_literals

from unittest.case import TestCase

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from tests.algos.opt.opt_lib_test_base import OptLibraryTestBase


class TestScipyGlobalOpt(TestCase):
    """"""

    OPT_LIB_NAME = "ScipyGlobalOpt"

    @staticmethod
    def get_problem():
        return Rosenbrock()

    def test_init(self):

        factory = OptimizersFactory()
        if factory.is_available(self.OPT_LIB_NAME):
            factory.create(self.OPT_LIB_NAME)


def get_options(algo_name):
    return {
        "max_iter": 10000,
        "n": 3,
        "iters": 5,
        "sampling_method": "sobol",
    }


suite_tests = OptLibraryTestBase()
for test_method in suite_tests.generate_test("ScipyGlobalOpt", get_options):
    setattr(TestScipyGlobalOpt, test_method.__name__, test_method)
