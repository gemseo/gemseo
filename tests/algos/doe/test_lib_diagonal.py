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
#      :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

from gemseo.algos.doe.doe_factory import DOEFactory

from .doe_lib_test_base import DOELibraryTestBase


class TestDiagonalLib(unittest.TestCase):
    """"""

    DOE_LIB_NAME = "DiagonalDOE"

    def test_init(self):
        """"""
        factory = DOEFactory()
        if factory.is_available(self.DOE_LIB_NAME):
            factory.create(self.DOE_LIB_NAME)

    def test_invalid_algo(self):
        """"""
        algo_name = "unknown_algo"
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

    def test_diagonal_doe(self):
        """"""
        algo_name = "DiagonalDOE"
        dim = 3
        n_samples = 10
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME, algo_name=algo_name, dim=dim, n_samples=n_samples
        )
        samples = doe_library.samples
        self.assertEqual(samples.shape, (n_samples, dim))
        self.assertAlmostEqual(samples[4, 0], 0.4, places=1)


def get_expected_nsamples(algo, dim, n_samples=None):
    """

    :param algo: param dim:
    :param n_samples: Default value = None)
    :param dim:

    """
    return n_samples


def get_options(algo_name, dim):
    """

    :param algo_name: param dim:
    :param dim:

    """
    options = {"n_samples": 13}
    return options


suite_tests = DOELibraryTestBase()
for test_method in suite_tests.generate_test(
    TestDiagonalLib.DOE_LIB_NAME, get_expected_nsamples, get_options
):
    setattr(TestDiagonalLib, test_method.__name__, test_method)
