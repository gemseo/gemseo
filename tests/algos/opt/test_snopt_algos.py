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
from __future__ import annotations

import unittest
from unittest import TestCase

from gemseo.algos.opt.opt_factory import OptimizersFactory

from .opt_lib_test_base import OptLibraryTestBase


class SnoptResult:
    """"""

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
    """"""

    def test_init(self):
        """"""
        if OptimizersFactory().is_available("SnOpt"):
            OptimizersFactory().create("SnOpt")

    def test_library_name(self):
        """Check the library name."""
        if OptimizersFactory().is_available("SnOpt"):
            from gemseo.algos.opt.lib_snopt import SnOpt

            assert SnOpt.LIBRARY_NAME == "pSeven"


suite_tests = OptLibraryTestBase()
for test_method in suite_tests.generate_test("SnOpt"):
    setattr(TestSNOPT, test_method.__name__, test_method)


if __name__ == "__main__":
    unittest.main()
