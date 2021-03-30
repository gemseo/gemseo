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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from copy import deepcopy

import numpy as np
from future import standard_library

from gemseo import LOGGER
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.third_party.junitxmlreq import link_to
from gemseo.utils.py23_compat import PY2

standard_library.install_aliases()


class DOELibraryTestBase(object):
    """ """

    @staticmethod
    def relative_norm(x, x_ref):
        """

        :param x: param x_ref:
        :param x_ref:

        """
        return np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)

    @staticmethod
    def norm(x):
        """

        :param x:

        """
        return np.linalg.norm(x)

    @staticmethod
    def generate_one_test(doe_algo_name, dim=3, **options):
        """

        :param doe_algo_name: algorithm name
        :param dim:  (Default value = 3)
        :param options: library options, see associated JSON file

        """
        problem = DOELibraryTestBase.get_problem(dim)
        doe_library = DOEFactory().create(doe_algo_name)
        LOGGER.info("Run test problem " + str(problem.__class__.__name__))
        doe_library.execute(problem, **options)
        return doe_library

    @staticmethod
    def run_and_test_problem(
        dim, doe_library, algo_name, get_expected_nsamples, options
    ):
        """

        :param dim: param doe_library:
        :param algo_name: param get_expected_nsamples:
        :param options:
        :param doe_library:
        :param get_expected_nsamples:

        """
        problem = DOELibraryTestBase.get_problem(dim)
        LOGGER.info("Run test problem " + str(problem.__class__.__name__))
        doe_library.execute(problem, algo_name=algo_name, **options)
        samples = doe_library.samples

        pb_name = problem.__class__.__name__
        error_msg = "DOE with " + algo_name
        error_msg += " failed to generate sample on problem " + pb_name

        if not len(samples.shape) == 2 or samples.shape[0] == 0:
            error_msg += ", wrong samples shapes : " + str(samples.shape)
            return error_msg
        n_samples = options.get("n_samples")
        exp_samples = get_expected_nsamples(algo_name, dim, n_samples)
        get_samples = samples.shape[0]
        if exp_samples is not None and get_samples != exp_samples:
            error_msg += "\n number_samples are not the expected ones : "
            error_msg += (
                "\n expected : " + str(exp_samples) + " got : " + str(get_samples)
            )
            return error_msg
        return None

    def create_test(self, dim, doe_library, algo_name, get_expected_nsamples, options):
        """

        :param dim: param doe_library:
        :param algo_name: param get_expected_nsamples:
        :param options:
        :param doe_library:
        :param get_expected_nsamples:

        """

        @link_to("Req-MDO-2", "Req-MDO-9", "Req-MDO-7")
        def test_algo(self=None):
            """

            :param self: Default value = None)

            """
            msg = DOELibraryTestBase.run_and_test_problem(
                dim, doe_library, algo_name, get_expected_nsamples, options
            )
            if msg is not None:
                raise Exception(msg)

        return test_algo

    @staticmethod
    def get_problem(dim):
        """Reinsantiate problem to do not erase it

        :param dim:

        """
        problem = Rosenbrock(dim)
        problem.check()
        return problem

    def generate_test(self, opt_lib_name, get_expected_nsamples, get_options):
        """Generates the tests for an opt library
        Filters algorithms adapted to the benchmark problems

        :param opt_lib_name: name of the library
        :param get_expected_nsamples: param get_options:
        :param get_options:
        :returns: list of test methods to be attached to a unitest class

        """
        tests = []
        factory = DOEFactory()

        if factory.is_available(opt_lib_name):
            for dim in [1, 5]:
                opt_lib = DOEFactory().create(opt_lib_name)
                algos = opt_lib.filter_adapted_algorithms(
                    DOELibraryTestBase.get_problem(dim)
                )
                for algo_name in algos:
                    options = deepcopy(get_options(algo_name, dim))
                    # Must copy options otherwise they are erased in the loop
                    test_method = self.create_test(
                        dim,
                        opt_lib,
                        algo_name,
                        get_expected_nsamples,
                        deepcopy(options),
                    )
                    name = (
                        "test_"
                        + opt_lib.__class__.__name__
                        + "_lib_"
                        + algo_name
                        + "_on_Rosenbrock_n_"
                        + str(dim)
                    )
                    name = name.replace("-", "_")
                    if PY2:
                        name = name.encode("ascii")
                    test_method.__name__ = name
                    tests.append(test_method)
        return tests
