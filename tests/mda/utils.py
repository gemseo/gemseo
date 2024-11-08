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

from __future__ import annotations  # noqa: I001

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

from ..core.test_chain import two_virtual_disciplines  # noqa: F401
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


def generate_parallel_doe(
    main_mda_name: str = "MDAChain",
    n_samples: int = 4,
    main_mda_settings: StrKeyMapping = READ_ONLY_EMPTY_DICT,
) -> float:
    """Execute a parallel DOE with a custom `main_mda_name`.

    Args:
        main_mda_name: The main mda class to be used to execute the
            parallel DOE scenario.
        n_samples: The number of samples for the DOE.
        main_mda_settings: The main MDA settings.

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
        "y_4",
        design_space,
        formulation_name="MDF",
        scenario_type="DOE",
        maximize_objective=True,
        main_mda_name=main_mda_name,
        main_mda_settings=main_mda_settings,
    )
    scenario.execute(
        algo_name="DiagonalDOE",
        n_samples=n_samples,
        n_processes=2,
    )
    return scenario.optimization_result.f_opt
