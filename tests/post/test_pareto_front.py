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

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from os.path import exists

import pytest

from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.power_2 import Power2


@pytest.mark.usefixtures("tmp_wd")
class TestParetoFrontPost(unittest.TestCase):
    def test_pareto(self):
        factory = PostFactory()
        if not factory.is_available("ScatterPlotMatrix"):
            return

        problem = Power2()
        DOEFactory().execute(problem, algo_name="fullfact", n_samples=50)
        file_path = "power"
        post = factory.execute(
            problem,
            "ParetoFront",
            save=True,
            file_path=file_path,
            objectives=problem.get_all_functions_names(),
        )
        assert len(post.output_files) == 1
        for outf in post.output_files:
            assert exists(outf)
