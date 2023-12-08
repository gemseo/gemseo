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
"""Test helpers."""

from __future__ import annotations

from functools import partial

import pytest

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.problems.sobieski.core.design_space import SobieskiDesignSpace


def generate_parallel_doe(
    main_mda_name: str = "MDAChain",
    n_samples: int = 4,
    inner_mda_name: str = "MDAJacobi",
) -> float:
    """Execute a parallel DOE with a custom `main_mda_name`.

    Args:
        main_mda_name: The main mda class to be used to execute the
            parallel DOE scenario.
        n_samples: The number of samples for the DOE.
        inner_mda_name: The inner mda class.

    Returns:
        The optimum solution of the parallel DOE scenario.
    """
    design_space = SobieskiDesignSpace()
    scenario = create_scenario(
        create_discipline([
            "SobieskiPropulsion",
            "SobieskiStructure",
            "SobieskiAerodynamics",
            "SobieskiMission",
        ]),
        "MDF",
        objective_name="y_4",
        design_space=design_space,
        scenario_type="DOE",
        maximize_objective=True,
        main_mda_name=main_mda_name,
        inner_mda_name=inner_mda_name,
    )
    scenario.execute({
        "algo": "DiagonalDOE",
        "n_samples": n_samples,
        "algo_options": {"n_processes": 2},
    })
    return scenario.optimization_result.f_opt


@pytest.fixture()
def generate_parallel_doe_data():
    """Wrap a parallel DOE scenario to be used in the MDA tests.

    Returns:
        A wrapped parallel doe scenario for which the `main_mda_name` can be
            given as an argument.
    """
    return partial(generate_parallel_doe)
