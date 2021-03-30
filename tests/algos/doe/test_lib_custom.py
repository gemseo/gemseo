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
#      :author: Damien Guenot - 28 avr. 2016
#    OTHER AUTHORS   - MACROSCOPIC CHANGES


from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from os.path import dirname, join

from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.api import configure_logger

from .doe_lib_test_base import DOELibraryTestBase

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class Test_CustomLib(unittest.TestCase):
    """ """

    DOE_LIB_NAME = "CustomDOE"

    def test_init(self):
        """ """
        factory = DOEFactory()
        if factory.is_available(self.DOE_LIB_NAME):
            factory.create(self.DOE_LIB_NAME)

    def test_missing_file_except(self):
        """ """
        dim = 3
        self.assertRaises(
            Exception,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            dim=dim,
            n_samples=20,
        )

    def test_delimiter_option(self):
        """ """
        dim = 3
        doe_file = join(dirname(__file__), "dim_" + str(dim) + "_semicolon.csv")
        doe_library = DOELibraryTestBase.generate_one_test(
            self.DOE_LIB_NAME, dim=dim, delimiter=";", doe_file=doe_file
        )
        samples = doe_library.samples
        self.assertEqual(samples.shape, (30, 3))

    def test_check_dv_lenght(self):
        """ """
        dim = 4
        doe_file = join(dirname(__file__), "dim_3_semicolon.csv")
        self.assertRaises(
            ValueError,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            dim=dim,
            delimiter=";",
            doe_file=doe_file,
        )

    def test_read_file_error(self):
        """ """
        dim = 4
        doe_file = join(dirname(__file__), "dim_3_semicolon.csv")
        self.assertRaises(
            Exception,
            DOELibraryTestBase.generate_one_test,
            self.DOE_LIB_NAME,
            dim=dim,
            doe_file=doe_file,
        )


def get_expected_nsamples(algo, dim, n_samples=None):
    """

    :param algo: param dim:
    :param n_samples: Default value = None)
    :param dim:

    """
    if dim == 1:
        return 9
    elif dim == 5:
        return 2


def get_options(algo_name, dim):
    """

    :param algo_name: param dim:
    :param dim:

    """
    options = {"n_samples": 13}
    dname = dirname(__file__)
    options["doe_file"] = join(dname, "dim_" + str(dim) + ".csv")
    options["dim"] = dim
    return options


#
suite_tests = DOELibraryTestBase()
for test_method in suite_tests.generate_test(
    Test_CustomLib.DOE_LIB_NAME, get_expected_nsamples, get_options
):
    setattr(Test_CustomLib, test_method.__name__, test_method)
