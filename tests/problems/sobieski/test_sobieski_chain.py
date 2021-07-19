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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import division, unicode_literals

import unittest

from gemseo.problems.sobieski.chains import SobieskiChain, SobieskiMDAGaussSeidel


class TestSobieskiChain(unittest.TestCase):
    """"""

    def test_exec_chain(self):
        """"""
        chain = SobieskiChain()
        chain.execute()

    def test_exec_mda(self):
        """"""
        mda = SobieskiMDAGaussSeidel()
        mda.execute()
        self.assertAlmostEqual(
            mda.get_outputs_by_name("y_4")[0],
            # 535.78222086,
            535.78213193,
            4,
        )
