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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""OpenTURNS DOE algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import Optional
from typing import Union

import openturns

from gemseo.algos.doe.base_doe_library import BaseDOELibrary
from gemseo.algos.doe.base_doe_library import DOEAlgorithmDescription
from gemseo.algos.doe.openturns._algos.ot_axial_doe import OTAxialDOE
from gemseo.algos.doe.openturns._algos.ot_centered_lhs import OTCenteredLHS
from gemseo.algos.doe.openturns._algos.ot_composite_doe import OTCompositeDOE
from gemseo.algos.doe.openturns._algos.ot_factorial_doe import OTFactorialDOE
from gemseo.algos.doe.openturns._algos.ot_faure_sequence import OTFaureSequence
from gemseo.algos.doe.openturns._algos.ot_full_factorial_doe import OTFullFactorialDOE
from gemseo.algos.doe.openturns._algos.ot_halton_sequence import OTHaltonSequence
from gemseo.algos.doe.openturns._algos.ot_haselgrove_sequence import (
    OTHaselgroveSequence,
)
from gemseo.algos.doe.openturns._algos.ot_monte_carlo import OTMonteCarlo
from gemseo.algos.doe.openturns._algos.ot_optimal_lhs import OTOptimalLHS
from gemseo.algos.doe.openturns._algos.ot_reverse_halton_sequence import (
    OTReverseHaltonSequence,
)
from gemseo.algos.doe.openturns._algos.ot_sobol_doe import OTSobolDOE
from gemseo.algos.doe.openturns._algos.ot_sobol_sequence import OTSobolSequence
from gemseo.algos.doe.openturns._algos.ot_standard_lhs import OTStandardLHS
from gemseo.algos.doe.openturns.settings.ot_axial_doe import OTAxialDOESettings
from gemseo.algos.doe.openturns.settings.ot_centered_lhs import OTCenteredLHSSettings
from gemseo.algos.doe.openturns.settings.ot_composite_doe import OTCompositeDOESettings
from gemseo.algos.doe.openturns.settings.ot_factorial_doe import OTFactorialDOESettings
from gemseo.algos.doe.openturns.settings.ot_faure import OTFaureSequenceSettings
from gemseo.algos.doe.openturns.settings.ot_full_factorial_doe import (
    OTFullFactorialDOESettings,
)
from gemseo.algos.doe.openturns.settings.ot_halton_sequence import (
    OTHaltonSequenceSettings,
)
from gemseo.algos.doe.openturns.settings.ot_haselgrove_sequence import (
    OTHaselgroveSequenceSettings,
)
from gemseo.algos.doe.openturns.settings.ot_monte_carlo import OTMonteCarloSettings
from gemseo.algos.doe.openturns.settings.ot_optimal_lhs import OTOptimalLHSSettings
from gemseo.algos.doe.openturns.settings.ot_reverse_halton_sequence import (
    OTReverseHaltonSequenceSettings,
)
from gemseo.algos.doe.openturns.settings.ot_sobol_doe import OTSobolDOESettings
from gemseo.algos.doe.openturns.settings.ot_sobol_sequence import (
    OTSobolSequenceSettings,
)
from gemseo.algos.doe.openturns.settings.ot_standard_lhs import OTStandardLHSSettings
from gemseo.typing import RealArray

if TYPE_CHECKING:
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.doe.base_doe import BaseDOE
    from gemseo.typing import NumberArray

OptionType = Optional[Union[str, int, float, bool, Sequence[int], RealArray]]


@dataclass
class OpenTURNSAlgorithmDescription(DOEAlgorithmDescription):
    """The description of a DOE algorithm from the OpenTURNS library."""

    library_name: str = "OpenTURNS"
    """The library name."""


class OpenTURNS(BaseDOELibrary):
    """The OpenTURNS DOE algorithms library."""

    # Algorithm names within GEMSEO
    __AXIAL: Final[str] = "OT_AXIAL"
    __COMPOSITE: Final[str] = "OT_COMPOSITE"
    __FACTORIAL: Final[str] = "OT_FACTORIAL"
    __FAURE: Final[str] = "OT_FAURE"
    __FULLFACT: Final[str] = "OT_FULLFACT"
    __HALTON: Final[str] = "OT_HALTON"
    __HASELGROVE: Final[str] = "OT_HASELGROVE"
    __LHS: Final[str] = "OT_LHS"
    __LHSC: Final[str] = "OT_LHSC"
    __MONTE_CARLO: Final[str] = "OT_MONTE_CARLO"
    __OPT_LHS: Final[str] = "OT_OPT_LHS"
    __RANDOM: Final[str] = "OT_RANDOM"
    __REVERSE_HALTON: Final[str] = "OT_REVERSE_HALTON"
    __SOBOL: Final[str] = "OT_SOBOL"
    __SOBOL_INDICES: Final[str] = "OT_SOBOL_INDICES"

    __NAMES_TO_CLASSES: Final[Mapping[str, type[BaseDOE]]] = {
        __AXIAL: OTAxialDOE,
        __COMPOSITE: OTCompositeDOE,
        __FAURE: OTFaureSequence,
        __FACTORIAL: OTFactorialDOE,
        __FULLFACT: OTFullFactorialDOE,
        __HALTON: OTHaltonSequence,
        __HASELGROVE: OTHaselgroveSequence,
        __LHS: OTStandardLHS,
        __LHSC: OTCenteredLHS,
        __MONTE_CARLO: OTMonteCarlo,
        __OPT_LHS: OTOptimalLHS,
        __RANDOM: OTMonteCarlo,
        __REVERSE_HALTON: OTReverseHaltonSequence,
        __SOBOL: OTSobolSequence,
        __SOBOL_INDICES: OTSobolDOE,
    }
    """The algorithm names bound to the OpenTURNS classes."""

    __DOC: Final[str] = "http://openturns.github.io/openturns/latest/user_manual/"

    ALGORITHM_INFOS: ClassVar[dict[str, OpenTURNSAlgorithmDescription]] = {
        __SOBOL: OpenTURNSAlgorithmDescription(
            algorithm_name=__SOBOL,
            description="Sobol sequence",
            internal_algorithm_name=__SOBOL,
            website=f"{__DOC}_generated/openturns.SobolSequence.html",
            settings=OTSobolSequenceSettings,
        ),
        __RANDOM: OpenTURNSAlgorithmDescription(
            algorithm_name=__RANDOM,
            description="Random sampling",
            internal_algorithm_name=__RANDOM,
            website=f"{__DOC}_generated/openturns.Uniform.html",
            settings=OTMonteCarloSettings,
        ),
        __HASELGROVE: OpenTURNSAlgorithmDescription(
            algorithm_name=__HASELGROVE,
            description="Haselgrove sequence",
            internal_algorithm_name=__HASELGROVE,
            website=f"{__DOC}_generated/openturns.HaselgroveSequence.html",
            settings=OTHaselgroveSequenceSettings,
        ),
        __REVERSE_HALTON: OpenTURNSAlgorithmDescription(
            algorithm_name=__REVERSE_HALTON,
            description="Reverse Halton",
            internal_algorithm_name=__REVERSE_HALTON,
            website=f"{__DOC}_generated/openturns.ReverseHaltonSequence.html",
            settings=OTReverseHaltonSequenceSettings,
        ),
        __HALTON: OpenTURNSAlgorithmDescription(
            algorithm_name=__HALTON,
            description="Halton sequence",
            internal_algorithm_name=__HALTON,
            website=f"{__DOC}_generated/openturns.HaltonSequence.html",
            settings=OTHaltonSequenceSettings,
        ),
        __FAURE: OpenTURNSAlgorithmDescription(
            algorithm_name=__FAURE,
            description="Faure sequence",
            internal_algorithm_name=__FAURE,
            website=f"{__DOC}_generated/openturns.FaureSequence.html",
            settings=OTFaureSequenceSettings,
        ),
        __MONTE_CARLO: OpenTURNSAlgorithmDescription(
            algorithm_name=__MONTE_CARLO,
            description="Monte Carlo sequence",
            internal_algorithm_name=__MONTE_CARLO,
            website=f"{__DOC}_generated/openturns.Uniform.html",
            settings=OTMonteCarloSettings,
        ),
        __FACTORIAL: OpenTURNSAlgorithmDescription(
            algorithm_name=__FACTORIAL,
            description="Factorial design",
            internal_algorithm_name=__FACTORIAL,
            website=f"{__DOC}_generated/openturns.Factorial.html",
            settings=OTFactorialDOESettings,
        ),
        __COMPOSITE: OpenTURNSAlgorithmDescription(
            algorithm_name=__COMPOSITE,
            description="Composite design",
            internal_algorithm_name=__COMPOSITE,
            website=f"{__DOC}_generated/openturns.Composite.html",
            settings=OTCompositeDOESettings,
        ),
        __AXIAL: OpenTURNSAlgorithmDescription(
            algorithm_name=__AXIAL,
            description="Axial design",
            internal_algorithm_name=__AXIAL,
            website=f"{__DOC}_generated/openturns.Axial.html",
            settings=OTAxialDOESettings,
        ),
        __OPT_LHS: OpenTURNSAlgorithmDescription(
            algorithm_name=__OPT_LHS,
            description="Optimal Latin Hypercube Sampling",
            internal_algorithm_name=__OPT_LHS,
            website=f"{__DOC}_generated/openturns.SimulatedAnnealingLHS.html",
            settings=OTOptimalLHSSettings,
        ),
        __LHS: OpenTURNSAlgorithmDescription(
            algorithm_name=__LHS,
            description="Latin Hypercube Sampling",
            internal_algorithm_name=__LHS,
            website=f"{__DOC}_generated/openturns.LHS.html",
            settings=OTStandardLHSSettings,
        ),
        __LHSC: OpenTURNSAlgorithmDescription(
            algorithm_name=__LHSC,
            description="Centered Latin Hypercube Sampling",
            internal_algorithm_name=__LHSC,
            website=f"{__DOC}_generated/openturns.LHS.html",
            settings=OTCenteredLHSSettings,
        ),
        __FULLFACT: OpenTURNSAlgorithmDescription(
            algorithm_name=__FULLFACT,
            description="Full factorial design",
            internal_algorithm_name=__FULLFACT,
            website=f"{__DOC}_generated/openturns.Box.html",
            settings=OTFullFactorialDOESettings,
        ),
        __SOBOL_INDICES: OpenTURNSAlgorithmDescription(
            algorithm_name=__SOBOL_INDICES,
            description="DOE for Sobol indices",
            internal_algorithm_name=__SOBOL_INDICES,
            website=f"{__DOC}_generated/openturns.SobolIndicesAlgorithm.html",
            settings=OTSobolDOESettings,
        ),
    }

    def _generate_unit_samples(
        self,
        design_space: DesignSpace,
        n_samples: int = 0,
        seed: int | None = None,
        **settings: OptionType,
    ) -> NumberArray:
        """
        Args:
            n_samples: The number of samples.
                If 0, set from the options.
            seed: The seed used for reproducibility reasons.
                If ``None``, use :attr:`.seed`.
        """  # noqa: D205, D212, D415
        openturns.RandomGenerator.SetSeed(self._seeder.get_seed(seed))
        doe_algo = self.__NAMES_TO_CLASSES[self._algo_name]()

        return doe_algo.generate_samples(n_samples, design_space.dimension, **settings)
