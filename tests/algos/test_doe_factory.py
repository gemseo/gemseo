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
#         Francois Gallard : refactoring for v1, May 2016

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

from gemseo.algos.doe.doe_factory import DOEFactory


class TestDOEFactory(unittest.TestCase):
    """"""

    def test_is_available_error(self):
        """"""
        factory = DOEFactory()
        self.assertFalse(factory.is_available("None"))

    def test_error(self):
        """"""
        factory = DOEFactory()
        self.assertRaises(Exception, lambda: factory.create("idontexist"))
