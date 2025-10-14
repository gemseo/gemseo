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
"""The optimal LHS algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from numpy import array
from openturns import GeometricProfile
from openturns import LHSExperiment
from openturns import LinearProfile
from openturns import MonteCarloLHS
from openturns import SimulatedAnnealingLHS
from openturns import SpaceFillingC2
from openturns import SpaceFillingMinDist
from openturns import SpaceFillingPhiP
from strenum import StrEnum

from gemseo.algos.doe.openturns._algos.base_ot_doe import BaseOTDOE

if TYPE_CHECKING:
    from openturns import SpaceFillingImplementation
    from openturns import TemperatureProfileImplementation

    from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
    from gemseo.typing import RealArray


class OTOptimalLHS(BaseOTDOE):
    """The optimal LHS algorithm.

    .. note:: This class is a singleton.
    """

    class TemperatureProfile(StrEnum):
        """The name of the temperature profile."""

        GEOMETRIC = "Geometric"
        LINEAR = "Linear"

    class SpaceFillingCriterion(StrEnum):
        """The name of the space-filling criterion."""

        C2 = "C2"
        PHIP = "PhiP"
        MINDIST = "MinDist"

    __TEMPERATURE_PROFILES: Final[str, TemperatureProfileImplementation] = {
        TemperatureProfile.GEOMETRIC: GeometricProfile(),
        TemperatureProfile.LINEAR: LinearProfile(),
    }

    __SPACE_FILLING_CRITERIA: Final[str, SpaceFillingImplementation] = {
        SpaceFillingCriterion.C2: SpaceFillingC2(),
        SpaceFillingCriterion.PHIP: SpaceFillingPhiP(),
        SpaceFillingCriterion.MINDIST: SpaceFillingMinDist(),
    }

    def generate_samples(  # noqa: D102
        self,
        n_samples: int,
        dimension: int,
        annealing: bool = True,
        criterion: SpaceFillingCriterion = SpaceFillingCriterion.C2,
        n_replicates: int = 1_000,
        temperature: TemperatureProfile = TemperatureProfile.GEOMETRIC,
        settings: OT_OPT_LHS_Settings | None = None,
    ) -> RealArray:
        """
        Args:
            annealing: Whether to use simulated annealing to optimize the LHS.
                If ``False``, the crude Monte Carlo method is used.
                Ignored if ``settings`` is not `None`.
            criterion: The space-filling criterion.
                Ignored if ``settings`` is not `None`.
            n_replicates: The number of Monte Carlo replicates to optimize LHS.
                Ignored if ``settings`` is not `None`.
            temperature: The temperature profile for simulated annealing.
                Either "Geometric" or "Linear".
                Ignored if ``settings`` is not `None`.
        """  # noqa: D205, D212
        if settings is not None:
            n_samples = settings.n_samples
            annealing = settings.annealing
            criterion = settings.criterion
            n_replicates = settings.n_replicates
            temperature = settings.temperature

        lhs_experiment = LHSExperiment(
            self._get_uniform_distribution(dimension), n_samples
        )
        lhs_experiment.setAlwaysShuffle(True)
        if annealing:
            lhs_experiment = SimulatedAnnealingLHS(
                lhs_experiment,
                self.__SPACE_FILLING_CRITERIA[criterion],
                self.__TEMPERATURE_PROFILES[temperature],
            )
        else:
            lhs_experiment = MonteCarloLHS(lhs_experiment, n_replicates)

        return array(lhs_experiment.generate())
