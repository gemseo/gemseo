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
#      :author: Damien Guenot - 20 avr. 2016
#      :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest

from future import standard_library
from numpy import loadtxt, ones

from gemseo import SOFTWARE_NAME
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.api import configure_logger
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.third_party.junitxmlreq import link_to

from .doe_lib_test_base import DOELibraryTestBase

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class Test_pyDOE(unittest.TestCase):
    """ """

    DOE_LIB_NAME = "PyDOE"

    @link_to("Req-DEP-2", "Req-MDO-7")
    def test_init(self):
        """ """
        factory = DOEFactory()
        if factory.is_available(self.DOE_LIB_NAME):
            factory.create(self.DOE_LIB_NAME)

    def test_phip(self):
        """ """
        algo_name = "fullfact"
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME, algo_name=algo_name, dim=3, n_samples=20
        )
        samples = doe_library.samples
        self.assertAlmostEqual(
            2.9935556852521019, doe_library.compute_phip_criteria(samples), 6
        )

    def test_export_samples(self):
        """ """
        algo_name = "lhs"
        n_samples = 30
        dim = 3
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME, algo_name=algo_name, dim=dim, n_samples=n_samples
        )
        samples = doe_library.samples
        doe_file_name = "test_" + algo_name + ".csv"
        doe_library.export_samples(doe_output_file=doe_file_name)
        file_samples = loadtxt(doe_file_name, delimiter=",")
        self.assertEqual(samples.shape, (n_samples, dim))
        self.assertEqual(file_samples.shape, (n_samples, dim))
        os.remove(doe_file_name)

    def test_invalid_algo(self):
        """ """
        algo_name = "bidon"
        dim = 3
        n_samples = 100
        self.assertRaises(
            KeyError,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples,
        )

    def test_lhs_maximin(self):
        """ """
        dim = 3
        algo_name = "lhs"
        n_samples = 100
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples,
            criterion="maximin",
        )
        samples_maximin = doe_library.samples
        self.assertEqual(samples_maximin.shape, (n_samples, dim))
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME, algo_name=algo_name, dim=dim, n_samples=n_samples
        )
        samples = doe_library.samples
        self.assertEqual(samples.shape, (n_samples, dim))
        self.assertGreater(
            DOELibraryTestBase.relative_norm(samples, samples_maximin), 1e-8
        )

    def test_invalid_criterion(self):
        """ """
        algo_name = "lhs"
        dim = 3
        n_samples = 100
        self.assertRaises(
            Exception,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples,
            criterion="bidon",
        )

    def test_lhs_center(self):
        """ """
        algo_name = "lhs"
        dim = 3
        n_samples = 100
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples,
            criterion="center",
        )
        samples = doe_library.samples
        self.assertEqual(samples.shape, (n_samples, dim))

    def test_lhs_centermaximin(self):
        """ """
        algo_name = "lhs"
        dim = 3
        n_samples = 100
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples,
            criterion="centermaximin",
        )
        samples = doe_library.samples
        self.assertEqual(samples.shape, (n_samples, dim))

    def test_lhs_correlation(self):
        """ """
        algo_name = "lhs"
        dim = 3
        n_samples = 100
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples,
            criterion="corr",
        )
        samples = doe_library.samples
        self.assertEqual(samples.shape, (n_samples, dim))

    def test_iteration_error(self):
        """ """
        algo_name = "lhs"
        dim = 3
        n_samples = 100
        criterion = "corr"
        self.assertRaises(
            Exception,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples,
            criterion=criterion,
            iterations="a",
        )

    def test_center_error(self):
        """ """
        algo_name = "bbdesign"
        dim = 3
        self.assertRaises(
            Exception,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            center_bb=(4, 4),
        )
        algo_name = "ccdesign"
        self.assertRaises(
            Exception,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            center_cc=1,
        )

    def test_alpha_error(self):
        """ """
        algo_name = "ccdesign"
        dim = 3
        self.assertRaises(
            Exception,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            alpha="ValueError",
        )

    def test_face_error(self):
        """ """
        algo_name = "ccdesign"
        dim = 3
        self.assertRaises(
            Exception,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            face="ValueError",
        )

    #

    def test_missing_algo_name(self):
        """ """
        dim = 3
        n_samples = 100
        self.assertRaises(
            Exception,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            dim=dim,
            n_samples=n_samples,
        )

    def test_export_error(self):
        """ """
        algo_name = "lhs"
        dim = 3
        n_samples = 100
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples,
            criterion="corr",
        )
        doe_library.samples = None
        self.assertRaises(Exception, doe_library.export_samples, "test.csv")

    def test_rescale_samples(self):
        """ """
        algo_name = "lhs"
        dim = 3
        n_samples = 100
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples,
            criterion="corr",
        )
        samples = ones((10,))
        samples[0] += 1e-15
        doe_library._rescale_samples(samples)

    def test_bbdesign_center(self):
        """ """
        algo_name = "bbdesign"
        dim = 5
        n_samples = 46
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME, algo_name=algo_name, dim=dim, center=1
        )
        samples = doe_library.samples
        self.assertEqual(samples.shape, (n_samples, dim))

    def test_ccdesign_center(self):
        """ """
        algo_name = "ccdesign"
        dim = 5
        n_samples = 62
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME, algo_name=algo_name, dim=dim, center_cc=[10, 10]
        )
        samples = doe_library.samples
        self.assertEqual(samples.shape, (n_samples, dim))

    def test_integer_lhs(self):
        problem = Rosenbrock()
        problem.design_space.add_variable(
            "y", size=1, var_type="integer", l_b=10.0, u_b=15.0
        )
        DOEFactory().execute(problem, "lhs", n_samples=10)

        for sample in problem.database.get_x_history():
            assert int(sample[-1]) == sample[-1]


def get_expected_nsamples(algo, dim, n_samples=None):
    """

    :param algo: param dim:
    :param n_samples: Default value = None)
    :param dim:

    """
    if algo == "ff2n":
        return 2 ** dim
    if algo == "bbdesign":
        if dim == 5:
            return 46
    if algo == "pbdesign":
        if dim == 1:
            return 4
        if dim == 5:
            return 8
    if algo == "ccdesign":
        if dim == 5:
            return 50
    if algo == "fullfact":
        return None
    return n_samples


def get_options(algo_name, dim):
    """

    :param algo_name: param dim:
    :param dim:

    """
    options = {"n_samples": 13}
    return options


#
suite_tests = DOELibraryTestBase()

for test_method in suite_tests.generate_test(
    Test_pyDOE.DOE_LIB_NAME, get_expected_nsamples, get_options
):
    setattr(Test_pyDOE, test_method.__name__, test_method)
