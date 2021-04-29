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
#       :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, unicode_literals

import unittest
from os.path import exists

import pytest

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.power_2 import Power2


@pytest.mark.usefixtures("tmp_wd")
class TestSOM(unittest.TestCase):
    """"""

    def test_som(self):
        problem = Power2()
        OptimizersFactory().execute(problem, "SLSQP")
        factory = PostFactory()
        for val in problem.database.values():
            val.pop("pow2")
        post = factory.execute(problem, "SOM", n_x=4, n_y=3, show=False, save=True)
        assert len(post.output_files) == 1
        assert exists(post.output_files[0])

    def test_som_annotate(self):
        problem = Power2()
        OptimizersFactory().execute(problem, "SLSQP")
        factory = PostFactory()
        for val in problem.database.values():
            val.pop("pow2")
        post = factory.execute(
            problem, "SOM", n_x=4, n_y=3, show=False, save=True, annotate=True
        )
        assert len(post.output_files) == 1
        assert exists(post.output_files[0])
