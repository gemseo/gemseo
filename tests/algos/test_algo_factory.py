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
#                       initial documentation
#        :author: Remi Lafage
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import division, unicode_literals

import unittest

from gemseo.algos.opt.opt_factory import OptimizersFactory


class TestAlgorithmFactory(unittest.TestCase):
    """"""

    def test_is_available_error(self):
        """"""
        self.assertFalse(OptimizersFactory().is_available("None"))

    def test_init_library_error(self):
        """"""
        OptimizersFactory().create("L-BFGS-B")
        self.assertRaises(Exception, OptimizersFactory().create, "idontexist")

    def test_is_scipy_available(self):
        """"""
        assert OptimizersFactory().is_available("ScipyOpt")
        assert "SLSQP" in OptimizersFactory().algorithms
