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
#       :author: Pierre-Jean Barjhoux
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import division, unicode_literals

import unittest
from os.path import dirname, exists, join

import pytest

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.power_2 import Power2

POWER2 = join(dirname(__file__), "power2_opt_pb.h5")


@pytest.mark.usefixtures("tmp_wd")
class TestRadarChart(unittest.TestCase):
    """"""

    def test_radar(self):
        """"""
        factory = PostFactory()
        if factory.is_available("RadarChart"):
            problem = Power2()
            OptimizersFactory().execute(problem, "SLSQP")
            post = factory.execute(
                problem,
                "RadarChart",
                save=True,
                show=False,
                constraints_list=problem.get_constraints_names(),
                iteration=-1,
                file_path="radar1",
            )
            assert len(post.output_files) == 1
            for outf in post.output_files:
                assert exists(outf)

    def test_radar_load(self):
        """"""
        factory = PostFactory()
        if factory.is_available("RadarChart"):
            problem = OptimizationProblem.import_hdf(file_path=POWER2)
            post = factory.execute(
                problem,
                "RadarChart",
                save=True,
                show=False,
                constraints_list=problem.get_constraints_names(),
                iteration=2,
                file_path="radar2",
            )
            assert len(post.output_files) == 1
            for outf in post.output_files:
                assert exists(outf)
            self.assertRaises(
                ValueError,
                factory.execute,
                problem,
                "RadarChart",
                save=True,
                show=False,
                constraints_list=["toto"],
            )
            self.assertRaises(
                ValueError,
                factory.execute,
                problem,
                "RadarChart",
                save=True,
                show=False,
                constraints_list=problem.get_constraints_names(),
                iteration=1000,
            )
