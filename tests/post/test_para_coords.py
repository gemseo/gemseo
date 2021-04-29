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
from os.path import dirname, exists, join

import pytest
from numpy import ones

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.rosenbrock import Rosenbrock

POWER2 = join(dirname(__file__), "power2_opt_pb.h5")


@pytest.mark.usefixtures("tmp_wd")
class TestParaCoords(unittest.TestCase):
    """"""

    def test_scatter(self):
        """"""
        if PostFactory().is_available("ParallelCoordinates"):
            n = 10
            problem = Rosenbrock(n)
            problem.x_0 = ones(n) * 0.99
            OptimizersFactory().execute(problem, "SLSQP")
            post = PostFactory().execute(
                problem,
                "ParallelCoordinates",
                save=True,
                figsize_x=12,
                file_path="para_coords1",
            )
            assert len(post.output_files) == 2
            for outf in post.output_files:
                assert exists(outf)

    def test_scatter_load(self):
        """"""
        if PostFactory().is_available("ParallelCoordinates"):
            post = PostFactory().execute(
                POWER2, "ParallelCoordinates", save=True, figsize_x=6
            )
            assert len(post.output_files) == 2
            for outf in post.output_files:
                assert exists(outf)
