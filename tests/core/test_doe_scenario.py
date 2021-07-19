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

from __future__ import unicode_literals

import pytest

from gemseo.api import create_discipline, create_scenario
from gemseo.problems.sobieski.core import SobieskiProblem


@pytest.mark.usefixtures("tmp_wd")
@pytest.mark.skip_under_windows
def test_parallel_doe_hdf_cache():
    disciplines = create_discipline(
        [
            "SobieskiStructure",
            "SobieskiPropulsion",
            "SobieskiAerodynamics",
            "SobieskiMission",
        ]
    )
    path = "cache.h5"
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

    n_samples = 10
    input_data = {
        "n_samples": n_samples,
        "algo": "lhs",
        "algo_options": {"n_processes": 2},
    }
    scenario.execute(input_data)
    scenario.print_execution_metrics()
    assert len(scenario.formulation.opt_problem.database) == n_samples
    for disc in disciplines:
        assert disc.cache.get_length() == n_samples

    input_data = {
        "n_samples": n_samples,
        "algo": "lhs",
        "algo_options": {"n_processes": 2, "n_samples": n_samples},
    }
    scenario.execute(input_data)


def test_doe_scenario():
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

    n_samples = 10
    input_data = {
        "n_samples": n_samples,
        "algo": "lhs",
        "algo_options": {"n_processes": 1},
    }
    scenario.execute(input_data)
    scenario.print_execution_metrics()
    assert len(scenario.formulation.opt_problem.database) == n_samples
