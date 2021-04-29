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
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES


from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

from gemseo.core.discipline import MDODiscipline
from gemseo.core.formulation import MDOFormulation
from gemseo.utils.py23_compat import string_types
from gemseo.utils.source_parsing import SourceParsing


class TestSourceParsing(unittest.TestCase):
    def test_get_default_options_values(self):
        opts = SourceParsing.get_default_options_values(unittest.TestCase)
        assert opts == {"methodName": "runTest"}

        SourceParsing.get_default_options_values(MDOFormulation)

    def test_get_options_doc(self):
        opts_doc = SourceParsing.get_options_doc(MDODiscipline.__init__)
        assert "name" in opts_doc
        for v in opts_doc.values():
            assert isinstance(v, string_types)

    def test_emptydoc(self):
        def f(x):
            return 2 * x

        self.assertRaises(ValueError, SourceParsing.get_options_doc, f)
