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
#      :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from unittest import TestCase

from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.api import configure_logger

from .opt_lib_test_base import OptLibraryTestBase

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class SnoptResult(object):
    """ """

    def __init__(self):
        res = [1] * 14
        self.states = res[0]
        self.x = res[1]
        self.z = res[3]
        self.exit = (res[4] / 10) * 10
        self.info = res[4]
        self.iterations = res[5]
        self.major_itns = res[6]
        self.nS = res[10]
        self.nInf = res[11]
        self.sInf = res[12]
        self.objective = res[13]


class TestSNOPT(TestCase):
    """ """

    def test_init(self):
        """ """
        if OptimizersFactory().is_available("SnOpt"):
            OptimizersFactory().create("SnOpt")


suite_tests = OptLibraryTestBase()
for test_method in suite_tests.generate_test("SnOpt"):
    setattr(TestSNOPT, test_method.__name__, test_method)

if __name__ == "__main__":
    unittest.main()
