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
#      :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES


from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from os.path import join

from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger, create_discipline, create_scenario
from gemseo.problems.sobieski.core import SobieskiProblem
from gemseo.utils.py23_compat import TemporaryDirectory
from gemseo.utils.testing_utils import skip_under_windows

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class Test_DOE_Scenario(unittest.TestCase):
    @skip_under_windows
    def test_parallel_doe_hdf_cache(self):
        disciplines = create_discipline(
            [
                "SobieskiStructure",
                "SobieskiPropulsion",
                "SobieskiAerodynamics",
                "SobieskiMission",
            ]
        )
        with TemporaryDirectory() as out_dir:
            path = join(out_dir, "cache.h5")
            for disc in disciplines:
                disc.set_cache_policy(disc.HDF5_CACHE, cache_hdf_file=path)

            scenario = create_scenario(
                disciplines,
                "DisciplinaryOpt",
                "y_4",
                SobieskiProblem().read_design_space(),
                maximize_objective=True,
                scenario_type="DOE",
            )

            N_SAMPLES = 10
            input_data = {
                "n_samples": N_SAMPLES,
                "algo": "lhs",
                "algo_options": {"n_processes": 2},
            }
            scenario.execute(input_data)
            scenario.print_execution_metrics()
            assert len(scenario.formulation.opt_problem.database) == N_SAMPLES
            for disc in disciplines:
                assert disc.cache.get_length() == N_SAMPLES

            input_data = {
                "n_samples": N_SAMPLES,
                "algo": "lhs",
                "algo_options": {"n_processes": 2, "n_samples": N_SAMPLES},
            }
            scenario.execute(input_data)

    def test_doe_scenario(self):
        disciplines = create_discipline(
            [
                "SobieskiStructure",
                "SobieskiPropulsion",
                "SobieskiAerodynamics",
                "SobieskiMission",
            ]
        )

        scenario = create_scenario(
            disciplines,
            "DisciplinaryOpt",
            "y_4",
            SobieskiProblem().read_design_space(),
            maximize_objective=True,
            scenario_type="DOE",
        )

        N_SAMPLES = 10
        input_data = {
            "n_samples": N_SAMPLES,
            "algo": "lhs",
            "algo_options": {"n_processes": 1},
        }
        scenario.execute(input_data)
        scenario.print_execution_metrics()
        assert len(scenario.formulation.opt_problem.database) == N_SAMPLES
