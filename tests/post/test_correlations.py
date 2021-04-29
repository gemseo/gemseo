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

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.rosenbrock import Rosenbrock

POWER2 = join(dirname(__file__), "power2_opt_pb.h5")


@pytest.mark.usefixtures("tmp_wd")
class TestCorrelations(unittest.TestCase):
    """"""

    def test_correlations(self):
        """"""
        factory = PostFactory()
        if factory.is_available("Correlations"):
            problem = Rosenbrock(20)
            OptimizersFactory().execute(problem, "L-BFGS-B")

            post = factory.execute(
                problem,
                "Correlations",
                save=True,
                n_plots_x=4,
                n_plots_y=4,
                coeff_limit=0.95,
                file_path="correlations_1",
            )
            assert len(post.output_files) == 2
            for outf in post.output_files:
                assert exists(outf)

    def test_correlations_import(self):
        """"""
        factory = PostFactory()
        if factory.is_available("Correlations"):
            problem = OptimizationProblem.import_hdf(POWER2)
            post = factory.execute(
                problem,
                "Correlations",
                save=True,
                n_plots_x=4,
                n_plots_y=4,
                coeff_limit=0.999,
                file_path="correlations_2",
            )
            assert len(post.output_files) == 1
            for outf in post.output_files:
                assert exists(outf)
