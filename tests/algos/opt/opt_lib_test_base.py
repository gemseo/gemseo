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
#    INITIAL AUTHORS - initial API and implementation and/or
#                       initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import numpy as np
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.analytical.rastrigin import Rastrigin
from gemseo.problems.analytical.rosenbrock import Rosenbrock


class OptLibraryTestBase:
    """"""

    @staticmethod
    def relative_norm(x, x_ref):
        """

        :param x: param x_ref:
        :param x_ref:

        """
        xr_norm = np.linalg.norm(x_ref)
        if xr_norm < 1e-8:
            return np.linalg.norm(x - x_ref)
        return np.linalg.norm(x - x_ref) / xr_norm

    @staticmethod
    def norm(x):
        """

        :param x:

        """
        return np.linalg.norm(x)

    @staticmethod
    def generate_one_test(opt_lib_name, algo_name, **options):
        """

        :param opt_lib_name: param algo_name:
        :param algo_name:
        :param **options:

        """
        problem = OptLibraryTestBase().get_pb_instance("Power2")
        opt_library = OptimizersFactory().create(opt_lib_name)
        opt_library.execute(problem, algo_name=algo_name, **options)
        return opt_library

    @staticmethod
    def generate_one_test_unconstrained(opt_lib_name, algo_name, **options):
        """

        :param opt_lib_name: param algo_name:
        :param algo_name:
        :param **options:

        """
        problem = OptLibraryTestBase().get_pb_instance("Rosenbrock")
        opt_library = OptimizersFactory().create(opt_lib_name)
        opt_library.execute(problem, algo_name=algo_name, **options)
        return opt_library

    @staticmethod
    def generate_error_test(opt_lib_name, algo_name, **options):
        """

        :param opt_lib_name: param algo_name:
        :param algo_name:
        :param **options:

        """
        problem = Power2(exception_error=True)
        opt_library = OptimizersFactory().create(opt_lib_name)
        opt_library.execute(problem, algo_name=algo_name, **options)
        return opt_library

    @staticmethod
    def run_and_test_problem(problem, opt_library, algo_name, **options):
        """

        :param problem: param opt_library:
        :param algo_name: param **options:
        :param opt_library:
        :param **options:

        """
        opt = opt_library.execute(problem, algo_name=algo_name, **options)
        x_opt, f_opt = problem.get_solution()
        x_err = OptLibraryTestBase.relative_norm(opt.x_opt, x_opt)
        f_err = OptLibraryTestBase.relative_norm(opt.f_opt, f_opt)

        if x_err > 1e-2 or f_err > 1e-2:
            pb_name = problem.__class__.__name__
            error_msg = (
                "Optimization with "
                + algo_name
                + " failed to find solution"
                + " of problem "
                + pb_name
                + " after n calls = "
                + str(len(problem.database))
            )
            return error_msg
        return None

    def create_test(self, problem, opt_library, algo_name, options):
        """

        :param problem: param opt_library:
        :param algo_name: param options:
        :param opt_library:
        :param options:

        """

        def test_algo(self=None):
            """"""
            msg = OptLibraryTestBase.run_and_test_problem(
                problem, opt_library, algo_name, **options
            )
            if msg is not None:
                raise Exception(msg)
            return msg

        return test_algo

    def get_pb_instance(self, pb_name, pb_options=None):
        """
        :param pb_name: the name of the optimization problem
        :param pb_options: the options to be passed to the optimization problem
        """
        if pb_options is None:
            pb_options = {}

        if pb_name == "Rosenbrock":
            return Rosenbrock(2, **pb_options)
        elif pb_name == "Power2":
            return Power2(**pb_options)
        if pb_name == "Rastrigin":
            return Rastrigin(**pb_options)

    def generate_test(self, opt_lib_name, get_options=None, get_problem_options=None):
        """Generates the tests for an opt library Filters algorithms adapted to the
        benchmark problems.

        :param opt_lib_name: name of the library
        :param get_options: Default value = None)
        :returns: list of test methods to be attached to a unitest class
        """
        tests = []
        factory = OptimizersFactory()
        if factory.is_available(opt_lib_name):
            opt_lib = OptimizersFactory().create(opt_lib_name)
            for pb_name in ["Rosenbrock", "Power2", "Rastrigin"]:
                if get_problem_options is not None:
                    pb_options = get_problem_options(pb_name)
                else:
                    pb_options = {}
                problem = self.get_pb_instance(pb_name, pb_options)
                algos = opt_lib.filter_adapted_algorithms(problem)
                for algo_name in algos:
                    # Reinitialize problem between runs
                    problem = self.get_pb_instance(pb_name, pb_options)
                    if get_options is not None:
                        options = get_options(algo_name)
                    else:
                        options = {"max_iter": 10000}
                    test_method = self.create_test(problem, opt_lib, algo_name, options)
                    name = "test_" + opt_lib.__class__.__name__ + "_" + algo_name
                    name += "_on_" + problem.__class__.__name__
                    name = name.replace("-", "_")
                    test_method.__name__ = name

                    tests.append(test_method)
        return tests
