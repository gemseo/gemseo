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
from __future__ import annotations

from unittest.case import TestCase

import pytest
from gemseo.algos.database import Database
from gemseo.algos.opt.lib_scipy_global import ScipyGlobalOpt
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.problems.analytical.power_2 import Power2
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


@pytest.fixture(scope="module")
def pow2_database() -> Database:
    """The database resulting from the Power2 problem resolution."""
    problem = Power2()
    OptimizersFactory().execute(problem, "SHGO", max_iter=20)
    return problem.database


@pytest.mark.parametrize("name", ["pow2", "ineq1", "ineq2", "eq"])
def test_function_history_length(name, pow2_database):
    assert len(pow2_database.get_func_history(name)) == len(pow2_database)


def get_options(algo_name):
    opts = {
        "max_iter": 3000,
        "n": 100,
        "sampling_method": "sobol",
        "popsize": 5,
        "tol": 0.1,
        "seed": 1,
        "iters": 1,
        "mutation": (0.6, 1),
    }

    if algo_name == "differential_evolution":
        opts["normalize_design_space"] = False
    return opts


suite_tests = OptLibraryTestBase()
for test_method in suite_tests.generate_test("ScipyGlobalOpt", get_options):
    setattr(TestScipyGlobalOpt, test_method.__name__, test_method)


def test_library_name():
    """Check the library name."""
    assert ScipyGlobalOpt.LIBRARY_NAME == "SciPy"
