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
#        :author: Damien Guenot
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import division, unicode_literals

import unittest
from os.path import dirname, exists, join

import pytest

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.basic_history import BasicHistory

DIRNAME = dirname(__file__)
POWER2 = join(DIRNAME, "power2_opt_pb.h5")
POWER2_NAN = join(DIRNAME, "power2_opt_pb_nan.h5")


@pytest.mark.usefixtures("tmp_wd")
class TestBasicHistory(unittest.TestCase):
    """"""

    def test_basic_history(self):
        problem = OptimizationProblem.import_hdf(POWER2)
        view = BasicHistory(problem)
        view.execute(
            show=False,
            save=True,
            file_path="power2_basic",
            data_list=problem.get_constraints_names(),
        )
        for full_path in view.output_files:
            assert exists(full_path)

    def test_basic_history_desvars(self):
        problem = OptimizationProblem.import_hdf(POWER2)
        view = BasicHistory(problem)
        view.execute(
            show=False,
            save=True,
            file_path="power2_dv",
            data_list=problem.design_space.variables_names,
        )

        for full_path in view.output_files:
            assert exists(full_path)

    def test_basic_hist_nan(self):
        problem = OptimizationProblem.import_hdf(POWER2_NAN)
        view = BasicHistory(problem)
        view.execute(
            show=False,
            save=True,
            file_path="power2_dv_nans",
            data_list=problem.get_constraints_names(),
        )

        for full_path in view.output_files:
            assert exists(full_path)
