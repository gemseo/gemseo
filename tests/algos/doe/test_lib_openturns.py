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
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import numpy as np

from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.problems.analytical.rosenbrock import Rosenbrock

from .doe_lib_test_base import DOELibraryTestBase


class TestlibOpenturns(unittest.TestCase):
    """"""

    DOE_LIB_NAME = "OpenTURNS"

    def __create_distr(self, dist_name, dim=1, **kwargs):
        DOELibraryTestBase.get_problem(dim)
        doe_library = DOEFactory().create(self.DOE_LIB_NAME)
        doe_library.create_distribution(distribution_name=dist_name, **kwargs)
        return doe_library

    def __check_dist(self, distrib, dist_name, std_value):
        self.assertEqual(distrib.getName(), dist_name)
        self.assertEqual(np.array(distrib.getMean())[0], 0.5)
        self.assertAlmostEqual(
            np.array(distrib.getStandardDeviation())[0], std_value, places=6
        )

    def test_init(self):
        """"""
        factory = DOEFactory()
        if factory.is_available(self.DOE_LIB_NAME):
            factory.create(self.DOE_LIB_NAME)

    def test_check_float(self):
        """"""
        dist_name = "Normal"
        self.assertRaises(TypeError, self.__create_distr, dist_name, mu="0.5")

    def test_error_level_type(self):
        """"""
        algo_name = "OT_COMPOSITE"
        dim = 3
        self.assertRaises(
            Exception,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=20,
            levels=4,
        )

    def test_error_levels(self):
        """"""
        algo_name = "OT_COMPOSITE"
        dim = 3
        levels = [0.1, 0.2, 1.3]
        centers = [0.0] * dim
        self.assertRaises(
            ValueError,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=20,
            levels=levels,
            centers=centers,
        )
        levels = [-0.1, 0.2, 0.3]
        self.assertRaises(
            ValueError,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=20,
            levels=levels,
            centers=centers,
        )

        factory = DOEFactory()
        lib = factory.create(self.DOE_LIB_NAME)
        self.assertRaises(TypeError, lib._OpenTURNS__set_level_option, {"levels": 1})

    def test_error_centers(self):
        """"""
        algo_name = "OT_COMPOSITE"
        dim = 3
        levels = [0.1, 0.2, 0.3]
        self.assertRaises(
            ValueError,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            centers=[0.5] * (dim + 1),
            levels=levels,
        )

        self.assertRaises(
            Exception,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            centers=0.5,
            levels=levels,
        )

        self.assertRaises(
            ValueError,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            levels=levels,
        )
        factory = DOEFactory()
        openturns = factory.create("OT_LHS")
        openturns.problem = Rosenbrock()
        self.assertRaises(
            TypeError, openturns._OpenTURNS__set_center_option, {"centers": 1}
        )

    def test_composite_centers(self):
        """"""
        algo_name = "OT_COMPOSITE"
        dim = 2
        levels = [0.1, 0.25, 0.5, 1.0]
        centers = [0.2, 0.3]
        n_levels = len(levels)
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            levels=levels,
            centers=centers,
        )
        samples = doe_library.samples
        n_samples = 1 + n_levels * (2 * dim + 2 ** dim)
        self.assertEqual(samples.shape, (n_samples, dim))

    def test_axial_centers(self):
        """"""
        dim = 2
        levels = [0.1, 0.25, 0.5, 1.0]
        centers = [0.2, 0.3]
        n_levels = len(levels)
        algo_name = "OT_AXIAL"
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            levels=levels,
            centers=centers,
        )
        samples = doe_library.samples
        n_samples = 1 + 2 * n_levels * dim
        self.assertEqual(samples.shape, (n_samples, dim))

    #

    def test_factorial_centers(self):
        """"""
        dim = 2
        levels = [0.1, 0.25, 0.5, 1.0]
        centers = [0.2, 0.3]
        n_levels = len(levels)
        algo_name = "OT_FACTORIAL"
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            levels=levels,
            centers=centers,
        )
        samples = doe_library.samples
        n_samples = 1 + 2 * n_levels * dim
        self.assertEqual(samples.shape, (n_samples, dim))

    def test_create_uniform(self):
        """"""
        dist_name = "Uniform"
        distrib = self.__create_distr(dist_name).get_distributions_list()[0]
        self.__check_dist(distrib, dist_name, 0.28867513)

    def test_wrong_distribution(self):
        """"""
        dist_name = "DoNotExist"
        self.assertRaises(ValueError, self.__create_distr, dist_name)
        factory = DOEFactory()
        openturns = factory.create(self.DOE_LIB_NAME)
        name, _ = openturns._OpenTURNS__get_distribution({"centers": 1})
        assert name == openturns.DISTRIBUTION_DEFAULT

    def test_check_stratified_options(self):
        factory = DOEFactory()
        lib = factory.create(self.DOE_LIB_NAME)
        lib.problem = Rosenbrock()
        options = {}
        dimension = lib.problem.dimension
        self.assertRaises(
            KeyError, lib._OpenTURNS__check_stratified_options, dimension, options
        )
        options = {"levels": [0.5, 0.2]}
        options = lib._OpenTURNS__check_stratified_options(dimension, options)
        assert lib.CENTER_KEYWORD in options

    def test_create_triangular(self):
        """"""
        dist_name = "Triangular"
        doe_library = self.__create_distr(dist_name)
        distrib = doe_library.get_distributions_list()[0]
        self.__check_dist(distrib, dist_name, 0.20412415)
        doe_library = self.__create_distr(dist_name, centers=0.6)
        distrib = doe_library.get_distributions_list()[0]
        self.assertAlmostEqual(np.array(distrib.getMean())[0], 0.533333, places=6)

    def test_create_trapezoidal(self):
        """"""
        dist_name = "Trapezoidal"
        distrib = self.__create_distr(dist_name).get_distributions_list()[0]
        self.__check_dist(distrib, dist_name, 0.22821773)

        distribs = self.__create_distr(dist_name, start=0.5, end=0.5)
        distrib = distribs.get_distributions_list()[0]
        self.__check_dist(distrib, dist_name, 0.20412415)
        self.assertAlmostEqual(np.array(distrib.getMean())[0], 0.50, places=6)

    def test_create_beta(self):
        """"""
        dist_name = "Beta"
        distrib = self.__create_distr(dist_name).get_distributions_list()[0]
        self.__check_dist(distrib, dist_name, 0.223607)

        distribs = self.__create_distr(dist_name, mu=0.48, sigma=0.3)
        distrib = distribs.get_distributions_list()[0]
        self.assertAlmostEqual(np.array(distrib.getMean())[0], 0.48, places=6)
        self.assertAlmostEqual(
            np.array(distrib.getStandardDeviation())[0], 0.3, places=6
        )

        self.assertRaises(
            ValueError,
            lambda: self.__create_distr(
                dist_name, mu=1.5, sigma=0.3
            ).get_distributions_list()[0],
        )
        self.assertRaises(
            ValueError, lambda: self.__create_distr(dist_name, mu=0.48, sigma=-0.3)
        )

    def test_create_arcsine(self):
        """"""
        factory = DOEFactory()
        dist_name = "Arcsine"
        distrib = self.__create_distr(dist_name).get_distributions_list()[0]
        self.__check_dist(distrib, dist_name, 0.353553)
        factory.create(self.DOE_LIB_NAME).plot_distribution(distrib, show=False)

    def test_create_truncnormal(self):
        """"""
        dist_name = "TruncatedNormal"
        distrib = self.__create_distr(dist_name).get_distributions_list()[0]
        self.__check_dist(distrib, dist_name, 0.133157)

        distrib = self.__create_distr(dist_name, mu=0.48).get_distributions_list()[0]
        self.assertAlmostEqual(np.array(distrib.getMean())[0], 0.48, places=3)
        distrib = self.__create_distr(dist_name, sigma=0.5)

    def test_create_normal(self):
        """"""
        dist_name = "Normal"
        distrib = self.__create_distr(dist_name).get_distributions_list()[0]
        self.__check_dist(distrib, dist_name, 0.133333)

    def test_dist_list(self):
        """"""
        doe_library = DOEFactory().create(self.DOE_LIB_NAME)
        distname_list = ["Normal", "Arcsine", "Uniform", "Uniform"]
        for dist_name in distname_list:
            doe_library.create_distribution(distribution_name=dist_name, dim=1)
        doe_library.display_distributions_list()
        dist_list = doe_library.get_distributions_list()
        self.assertEqual(len(dist_list), len(distname_list))
        for i, dist_name in enumerate(distname_list):
            self.assertEqual(dist_list[i].getName(), dist_name)
            self.assertEqual(np.array(dist_list[i].getMean())[0], 0.5)

    def test_composed_dist(self):
        """"""
        doe_library = DOEFactory().create(self.DOE_LIB_NAME)
        distname_list = ["Normal", "Arcsine", "Uniform", "Uniform"]
        for dist_name in distname_list:
            doe_library.create_distribution(dist_name)
        doe_library.create_composed_distributions()
        comp_dist = doe_library.get_composed_distributions()
        self.assertEqual(comp_dist.getDimension(), len(distname_list))
        mean = np.array(comp_dist.getMean())
        self.assertEqual(np.min(mean), 0.5)
        self.assertEqual(np.max(mean), 0.5)

    def test_call(self):
        """"""
        algo = DOEFactory().create("OT_LHS")
        lhs = algo(10, 2)
        self.assertEqual(lhs.shape, (10, 2))

    def test_lhs_ot(self):
        """"""
        dim = 4
        algo_name = "OT_LHS"
        n_samples_1 = 3 ** dim
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME, algo_name=algo_name, dim=dim, n_samples=n_samples_1
        )
        samples_lhs = doe_library.samples
        self.assertEqual(samples_lhs.shape, (n_samples_1, dim))

        algo_name = "OT_LHSC"
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME, algo_name=algo_name, dim=dim, n_samples=n_samples_1
        )
        samples_lhsc = doe_library.samples
        self.assertEqual(samples_lhs.shape, (n_samples_1, dim))
        self.assertNotEqual(
            DOELibraryTestBase.relative_norm(samples_lhs, samples_lhsc), 0.0
        )

        algo_name = "OT_OPT_LHS"
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME, algo_name=algo_name, dim=dim, n_samples=n_samples_1
        )
        samples_lhs = doe_library.samples
        self.assertEqual(samples_lhs.shape, (n_samples_1, dim))
        DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples_1,
            n_replicates=10,
        )
        DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples_1,
            criterion="PhiP",
        )
        self.assertRaises(
            ValueError,
            DOELibraryTestBase.generate_one_test,
            doe_algo_name=self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples_1,
            criterion="Foo",
        )
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples_1,
            annealing=True,
            temperature="Linear",
        )
        self.assertRaises(
            ValueError,
            DOELibraryTestBase.generate_one_test,
            doe_algo_name=self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples_1,
            annealing=True,
            temperature="Foo",
        )

    def test_random_ot(self):
        """"""
        dim = 4
        algo_name = "OT_RANDOM"
        n_samples_1 = 3 ** dim
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME, algo_name=algo_name, dim=dim, n_samples=n_samples_1
        )
        samples = doe_library.samples
        self.assertEqual(samples.shape, (n_samples_1, dim))

    def test_fullfact_ot(self):
        """"""
        dim = 3
        algo_name = "OT_FULLFACT"
        n_samples_1 = 3 ** dim
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME, algo_name=algo_name, dim=dim, n_samples=n_samples_1
        )
        samples = doe_library.samples
        self.assertEqual(samples.shape, (n_samples_1, dim))

    def test_sobol_ot(self):
        """"""
        dim = 3
        algo_name = "OT_SOBOL"
        n_samples = 20
        problem = DOELibraryTestBase.get_problem(dim)
        doe_library = DOEFactory().create(self.DOE_LIB_NAME)
        doe_library.execute(problem, algo_name=algo_name, dim=dim, n_samples=n_samples)
        samples = doe_library.samples
        self.assertEqual(samples.shape, (n_samples, dim))

    def test_lhs(self):
        """"""
        dim = 4
        n_samples_1 = 3 ** dim
        algo_name = "OT_LHS"
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME, algo_name=algo_name, dim=dim, n_samples=n_samples_1
        )
        samples_lhs = doe_library.samples
        self.assertEqual(samples_lhs.shape, (n_samples_1, dim))
        dist_list = doe_library.get_distributions_list()
        self.assertEqual(len(dist_list), 1)

    def test_lhs_truncatednormal(self):
        """"""
        dim = 4
        n_samples_1 = 3 ** dim
        algo_name = "OT_LHS"
        dist_name = "TruncatedNormal"
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples_1,
            distribution_name=dist_name,
        )
        samples_lhs = doe_library.samples
        self.assertEqual(samples_lhs.shape, (n_samples_1, dim))
        dist_list = doe_library.get_distributions_list()
        self.assertEqual(len(dist_list), 1)
        self.assertEqual(dist_list[0].getName(), dist_name)

    #

    def test_lhs_error_composed(self):
        """"""
        dim = 3
        n_samples = 150
        algo_name = "OT_LHS"
        dist_namelist = [
            "TruncatedNormal",
            "Normal",
            "Uniform",
            "Trapezoidal",
            "Triangular",
        ]
        problem = DOELibraryTestBase.get_problem(dim)
        doe_library = DOEFactory().create(self.DOE_LIB_NAME)
        for distribution in dist_namelist:
            if distribution == "Trapezoidal":
                doe_library.create_distribution(
                    distribution_name=distribution, start=0.1234
                )
            elif distribution == "TruncatedNormal":
                doe_library.create_distribution(distribution_name=distribution, mu=0.42)
            elif distribution == "Triangular":
                doe_library.create_distribution(
                    distribution_name=distribution, centers=0.45
                )
            else:
                doe_library.create_distribution(distribution_name=distribution)
        dist_list = doe_library.get_distributions_list()
        for i_dist, distribution in enumerate(dist_list):
            self.assertEqual(distribution.getName(), dist_namelist[i_dist])
        self.assertRaises(
            ValueError,
            doe_library.execute,
            problem,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples,
        )
        doe_library.create_composed_distributions()

        self.assertRaises(
            ValueError,
            doe_library.execute,
            problem,
            algo_name=algo_name,
            dim=dim,
            n_samples=n_samples,
        )

    def test_lhs_composed(self):
        """"""
        dim = 5
        n_samples = 150
        algo_name = "OT_LHS"
        dist_namelist = [
            "TruncatedNormal",
            "Normal",
            "Uniform",
            "Trapezoidal",
            "Triangular",
        ]
        problem = DOELibraryTestBase.get_problem(dim)
        doe_library = DOEFactory().create(self.DOE_LIB_NAME)
        for distribution in dist_namelist:
            if distribution == "Trapezoidal":
                doe_library.create_distribution(
                    distribution_name=distribution, start=0.1234
                )
            elif distribution == "TruncatedNormal":
                doe_library.create_distribution(distribution_name=distribution, mu=0.42)
            elif distribution == "Triangular":
                doe_library.create_distribution(
                    distribution_name=distribution, centers=0.45
                )
            else:
                doe_library.create_distribution(distribution_name=distribution)
        dist_list = doe_library.get_distributions_list()
        for i_dist, distribution in enumerate(dist_list):
            self.assertEqual(distribution.getName(), dist_namelist[i_dist])
        doe_library.create_composed_distributions()
        composed_dist = doe_library.get_composed_distributions()
        self.assertEqual("ComposedDistribution", composed_dist.getName())

        parameters_list = composed_dist.getParametersCollection()
        self.assertEqual(composed_dist.getDimension(), dim)
        self.assertEqual(parameters_list[3][1], 0.1234)

        doe_library.execute(problem, algo_name=algo_name, dim=dim, n_samples=n_samples)
        samples_lhs = doe_library.samples
        self.assertEqual(samples_lhs.shape, (n_samples, dim))
        self.assertAlmostEqual(np.mean(samples_lhs[:, 2]), 0.5, 3)
        self.assertAlmostEqual(np.mean(samples_lhs[:, 0]), 0.42, 2)
        self.assertAlmostEqual(np.mean(samples_lhs[:, -1]), 0.4831, 4)

    def test_mc(self):
        """"""
        dim = 4
        n_samples = 20
        algo_name = "OT_MONTE_CARLO"
        dist_name = "TruncatedNormal"
        problem = DOELibraryTestBase.get_problem(dim)
        doe_library = DOEFactory().create(self.DOE_LIB_NAME)
        doe_library.create_distribution(distribution_name=dist_name, mu=0.42)
        doe_library.execute(problem, algo_name=algo_name, dim=dim, n_samples=n_samples)
        samples_lhs = doe_library.samples
        self.assertEqual(samples_lhs.shape, (n_samples, dim))

    def test_mc_composed(self):
        """"""
        dim = 5
        n_samples = 2000
        algo_name = "OT_MONTE_CARLO"
        dist_namelist = [
            "TruncatedNormal",
            "Normal",
            "Uniform",
            "Trapezoidal",
            "Triangular",
        ]
        problem = DOELibraryTestBase.get_problem(dim)
        doe_library = DOEFactory().create(self.DOE_LIB_NAME)
        for distribution in dist_namelist:
            if distribution == "Trapezoidal":
                doe_library.create_distribution(distribution, start=0.1234)
            elif distribution == "TruncatedNormal":
                doe_library.create_distribution(distribution, mu=0.42)
            elif distribution == "Triangular":
                doe_library.create_distribution(distribution, centers=0.45)
            else:
                doe_library.create_distribution(distribution)
        dist_list = doe_library.get_distributions_list()
        for i_dist, distribution in enumerate(dist_list):
            self.assertEqual(distribution.getName(), dist_namelist[i_dist])
        doe_library.create_composed_distributions()
        composed_dist = doe_library.get_composed_distributions()
        self.assertEqual("ComposedDistribution", composed_dist.getName())

        parameters_list = composed_dist.getParametersCollection()
        self.assertEqual(parameters_list[3][1], 0.1234)

        doe_library.execute(problem, algo_name=algo_name, dim=dim, n_samples=n_samples)
        samples_lhs = doe_library.samples
        self.assertEqual(samples_lhs.shape, (n_samples, dim))


def get_expected_nsamples(algo, dim, n_samples=None):
    """

    :param algo: param dim:
    :param n_samples: Default value = None)
    :param dim:

    """
    if algo == "OT_AXIAL":
        if dim == 1:
            return 5
        if dim == 5:
            return 21
    if algo == "OT_COMPOSITE":
        if dim == 1:
            return 9
        if dim == 5:
            return 85
    if algo == "OT_FACTORIAL":
        if dim == 1:
            return 5
        if dim == 5:
            return 65
    if algo == "OT_FULLFACT":
        if dim == 5:
            return 32
    if algo == "OT_SOBOL_INDICES":
        if dim == 1:
            return 12
        if dim == 5:
            return 7

    return n_samples


def get_options(algo_name, dim):
    """

    :param algo_name: param dim:
    :param dim:

    """
    options = {"n_samples": 13}
    options["levels"] = [0.1, 0.9]
    options["centers"] = [0.5] * dim
    return options


suite_tests = DOELibraryTestBase()
for test_method in suite_tests.generate_test(
    TestlibOpenturns.DOE_LIB_NAME, get_expected_nsamples, get_options
):
    setattr(TestlibOpenturns, test_method.__name__, test_method)
