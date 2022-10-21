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
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.core.discipline import MDODiscipline
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sellar.sellar import Sellar2
from gemseo.problems.sellar.sellar import SellarSystem
from gemseo.problems.sobieski.core.problem import SobieskiProblem


def generate_parallel_doe(
    main_mda_name: str,
    n_samples: int = 4,
) -> float:
    """Execute a parallel DOE with a custom `main_mda_name`.

    Args:
        main_mda_name: The main mda class to be used to execute the
            parallel DOE scenario.
        n_samples: The number of samples for the DOE.

    Returns:
        The optimum solution of the parallel DOE scenario.
    """
    design_space = SobieskiProblem().design_space
    scenario = create_scenario(
        create_discipline(
            [
                "SobieskiPropulsion",
                "SobieskiStructure",
                "SobieskiAerodynamics",
                "SobieskiMission",
            ]
        ),
        "MDF",
        objective_name="y_4",
        design_space=design_space,
        scenario_type="DOE",
        maximize_objective=True,
        main_mda_name=main_mda_name,
    )
    scenario.execute(
        {
            "algo": "DiagonalDOE",
            "n_samples": n_samples,
            "algo_options": {"n_processes": 2},
        }
    )
    return scenario.get_optimum().to_dict()["f_opt"]


@pytest.fixture
def generate_parallel_doe_data():
    """Wrap a parallel DOE scenario to be used in the MDA tests.

    Returns:
        A wrapped parallel doe scenario for which the `main_mda_name` can be
            given as an argument.
    """
    return partial(generate_parallel_doe)


@pytest.fixture
def sellar_disciplines() -> list[MDODiscipline]:
    """The disciplines of the Sellar problem.

    Returns:
        * A Sellar1 discipline.
        * A Sellar2 discipline.
        * A SellarSystem discipline.
    """
    return [Sellar1(), Sellar2(), SellarSystem()]
