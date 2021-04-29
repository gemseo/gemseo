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
from os.path import dirname, exists, join

import pytest
from numpy import repeat

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.post_factory import PostFactory

POWER2 = join(dirname(__file__), "power2_opt_pb.h5")
SSBJ = join(dirname(__file__), "mdf_backup.h5")


@pytest.mark.usefixtures("tmp_wd")
class TestVariableInfluence(unittest.TestCase):
    """"""

    def test_gradient_sensitivity(self):
        """"""
        if PostFactory().is_available("VariableInfluence"):
            problem = OptimizationProblem.import_hdf(POWER2)
            post = PostFactory().execute(
                problem, "VariableInfluence", file_path="var_infl", save=True
            )
            assert len(post.output_files) == 1
            for outf in post.output_files:
                assert exists(outf)
            database = problem.database
            database.filter(["pow2", "@pow2"])
            problem.constraints = []
            for k in list(database.keys()):
                v = database.pop(k)
                v["@pow2"] = repeat(v["@pow2"], 60)
                database[repeat(k.wrapped, 60)] = v

            post = PostFactory().execute(
                problem, "VariableInfluence", file_path="var_infl2", save=True
            )
            assert len(post.output_files) == 1
            for outf in post.output_files:
                assert exists(outf)

    def test_gradient_sensitivity_ssbj(self):
        if PostFactory().is_available("VariableInfluence"):
            problem = OptimizationProblem.import_hdf(SSBJ)
            post = PostFactory().execute(
                problem,
                "VariableInfluence",
                file_path="ssbj",
                log_scale=True,
                absolute_value=False,
                quantile=0.98,
                save=True,
                figsize_y=12,
                save_var_files=True,
            )
            assert len(post.output_files) == 14
            for outf in post.output_files:
                assert exists(outf)
