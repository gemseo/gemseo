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

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.doe_scenario import DOEScenario
from gemseo.post.post_factory import PostFactory
from gemseo.problems.sobieski.wrappers import SobieskiProblem, SobieskiStructure

POWER2 = join(dirname(__file__), "power2_opt_pb.h5")


@pytest.mark.usefixtures("tmp_wd")
class TestGradientSensitivity(unittest.TestCase):
    """"""

    def test_gradient_sensitivity(self):
        """"""
        if not PostFactory().is_available("GradientSensitivity"):
            return

        problem = OptimizationProblem.import_hdf(POWER2)
        post = PostFactory().execute(
            problem, "GradientSensitivity", file_path="grad_sens1", save=True
        )
        assert len(post.output_files) == 1
        assert exists(post.output_files[0])

        x_0 = problem.database.get_x_by_iter(0)
        problem.database[x_0].pop("@eq")
        post = PostFactory().execute(
            problem,
            "GradientSensitivity",
            file_path="grad_sens2",
            save=True,
            iteration=0,
        )
        assert len(post.output_files) == 1
        assert exists(post.output_files[0])

        disc = SobieskiStructure()
        design_space = SobieskiProblem().read_design_space()
        inputs = disc.get_input_data_names()
        design_space.filter(inputs)
        doe_scenario = DOEScenario([disc], "DisciplinaryOpt", "y_12", design_space)
        doe_scenario.execute(
            {
                "algo": "DiagonalDOE",
                "n_samples": 10,
                "algo_options": {"eval_jac": True},
            }
        )
        doe_scenario.post_process(
            "GradientSensitivity", file_path="grad_sens", save=True
        )
        doe_scenario2 = DOEScenario([disc], "DisciplinaryOpt", "y_12", design_space)
        doe_scenario2.execute(
            {
                "algo": "DiagonalDOE",
                "n_samples": 10,
                "algo_options": {"eval_jac": False},
            }
        )
        self.assertRaises(
            ValueError,
            doe_scenario2.post_process,
            "GradientSensitivity",
            file_path="grad_sens",
            save=True,
        )
