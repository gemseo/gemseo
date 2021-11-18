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

"""Test helpers."""
from __future__ import unicode_literals

from functools import partial
from typing import List

import pytest

from gemseo.api import create_discipline, create_scenario
from gemseo.problems.sellar.sellar import Sellar1, Sellar2, SellarSystem
from gemseo.problems.sobieski.core import SobieskiProblem


def generate_parallel_doe(
    main_mda_class,  # type: str
    n_samples=4,  # type: int
):  # type: (...) -> float
    """Execute a parallel DOE with a custom `main_mda_class`.

    Args:
        main_mda_class: The main mda class to be used to execute the
            parallel DOE scenario.
        n_samples: The number of samples for the DOE.

    Returns:
        The optimum solution of the parallel DOE scenario.
    """
    design_space = SobieskiProblem().read_design_space()
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
        main_mda_class=main_mda_class,
    )
    scenario.execute(
        {
            "algo": "DiagonalDOE",
            "n_samples": n_samples,
            "algo_options": {"n_processes": 2},
        }
    )
    return scenario.get_optimum().get_data_dict_repr()["f_opt"]


@pytest.fixture
def generate_parallel_doe_data():
    """Wrap a parallel DOE scenario to be used in the MDA tests.

    Returns:
        A wrapped parallel doe scenario for which the `main_mda_class` can be
            given as an argument.
    """
    return partial(generate_parallel_doe)


@pytest.fixture
def sellar_disciplines():  # type: (...)-> List[Sellar1,Sellar2,SellarSystem]
    """The disciplines of the Sellar problem.

    Returns:
        * A Sellar1 discipline.
        * A Sellar2 discipline.
        * A SellarSystem discipline.
    """
    return [Sellar1(), Sellar2(), SellarSystem()]
