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
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

import openturns

from gemseo.doe.core.base_doe_library import BaseDOELibrary
from gemseo.doe.core.base_doe_library import DOEAlgorithmDescription
from gemseo.doe.openturns._algorithm.ot_axial_doe import OTAxialDOE
from gemseo.doe.openturns._algorithm.ot_centered_lhs import OTCenteredLHS
from gemseo.doe.openturns._algorithm.ot_composite_doe import OTCompositeDOE
from gemseo.doe.openturns._algorithm.ot_factorial_doe import OTFactorialDOE
from gemseo.doe.openturns._algorithm.ot_faure_sequence import OTFaureSequence
from gemseo.doe.openturns._algorithm.ot_full_factorial_doe import OTFullFactorialDOE
from gemseo.doe.openturns._algorithm.ot_halton_sequence import OTHaltonSequence
from gemseo.doe.openturns._algorithm.ot_haselgrove_sequence import OTHaselgroveSequence
from gemseo.doe.openturns._algorithm.ot_monte_carlo import OTMonteCarlo
from gemseo.doe.openturns._algorithm.ot_optimal_lhs import OTOptimalLHS
from gemseo.doe.openturns._algorithm.ot_reverse_halton_sequence import (
    OTReverseHaltonSequence,
)
from gemseo.doe.openturns._algorithm.ot_sobol_doe import OTSobolDOE
from gemseo.doe.openturns._algorithm.ot_sobol_sequence import OTSobolSequence
from gemseo.doe.openturns._algorithm.ot_standard_lhs import OTStandardLHS
from gemseo.doe.openturns.settings.base_openturns_settings import BaseOpenTURNSSettings
from gemseo.doe.openturns.settings.ot_axial import OT_AXIAL_Settings
from gemseo.doe.openturns.settings.ot_composite import OT_COMPOSITE_Settings
from gemseo.doe.openturns.settings.ot_factorial import OT_FACTORIAL_Settings
from gemseo.doe.openturns.settings.ot_faure import OT_FAURE_Settings
from gemseo.doe.openturns.settings.ot_fullfact import OT_FULLFACT_Settings
from gemseo.doe.openturns.settings.ot_halton import OT_HALTON_Settings
from gemseo.doe.openturns.settings.ot_haselgrove import OT_HASELGROVE_Settings
from gemseo.doe.openturns.settings.ot_lhs import OT_LHS_Settings
from gemseo.doe.openturns.settings.ot_lhsc import OT_LHSC_Settings
from gemseo.doe.openturns.settings.ot_monte_carlo import OT_MONTE_CARLO_Settings
from gemseo.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.doe.openturns.settings.ot_random import OT_RANDOM_Settings
from gemseo.doe.openturns.settings.ot_reverse_halton import OT_REVERSE_HALTON_Settings
from gemseo.doe.openturns.settings.ot_sobol import OT_SOBOL_Settings
from gemseo.doe.openturns.settings.ot_sobol_indices import OT_SOBOL_INDICES_Settings
from gemseo.util.typing import RealArray

if TYPE_CHECKING:
    from gemseo.doe.core.base_doe import BaseDOE
    from gemseo.space.design import DesignSpace
    from gemseo.util.typing import NumberArray

OptionType = str | int | float | bool | Sequence[int] | RealArray | None


@dataclass
class OpenTURNSAlgorithmDescription(DOEAlgorithmDescription):
    """The description of a DOE algorithm from the OpenTURNS library."""

    library_name: str = "OpenTURNS"
    """The library name."""


class OpenTURNS(BaseDOELibrary[BaseOpenTURNSSettings]):
    """The OpenTURNS DOE algorithms library."""

    # Algorithm names within GEMSEO
    __axial: Final[str] = "OT_AXIAL"
    __composite: Final[str] = "OT_COMPOSITE"
    __factorial: Final[str] = "OT_FACTORIAL"
    __faure: Final[str] = "OT_FAURE"
    __fullfact: Final[str] = "OT_FULLFACT"
    __halton: Final[str] = "OT_HALTON"
    __haselgrove: Final[str] = "OT_HASELGROVE"
    __lhs: Final[str] = "OT_LHS"
    __lhsc: Final[str] = "OT_LHSC"
    __monte_carlo: Final[str] = "OT_MONTE_CARLO"
    __opt_lhs: Final[str] = "OT_OPT_LHS"
    __random: Final[str] = "OT_RANDOM"
    __reverse_halton: Final[str] = "OT_REVERSE_HALTON"
    __sobol: Final[str] = "OT_SOBOL"
    __sobol_indices: Final[str] = "OT_SOBOL_INDICES"

    __names_to_classes: Final[Mapping[str, type[BaseDOE]]] = MappingProxyType({
        __axial: OTAxialDOE,
        __composite: OTCompositeDOE,
        __faure: OTFaureSequence,
        __factorial: OTFactorialDOE,
        __fullfact: OTFullFactorialDOE,
        __halton: OTHaltonSequence,
        __haselgrove: OTHaselgroveSequence,
        __lhs: OTStandardLHS,
        __lhsc: OTCenteredLHS,
        __monte_carlo: OTMonteCarlo,
        __opt_lhs: OTOptimalLHS,
        __random: OTMonteCarlo,
        __reverse_halton: OTReverseHaltonSequence,
        __sobol: OTSobolSequence,
        __sobol_indices: OTSobolDOE,
    })
    """The algorithm names bound to the OpenTURNS classes."""

    __doc: Final[str] = "http://openturns.github.io/openturns/latest/user_manual/"

    ALGORITHM_INFOS: ClassVar[dict[str, OpenTURNSAlgorithmDescription]] = {
        __sobol: OpenTURNSAlgorithmDescription(
            algorithm_name=__sobol,
            description="Sobol sequence",
            internal_algorithm_name=__sobol,
            website=f"{__doc}_generated/openturns.SobolSequence.html",
            settings_class=OT_SOBOL_Settings,
        ),
        __random: OpenTURNSAlgorithmDescription(
            algorithm_name=__random,
            description="Random sampling",
            internal_algorithm_name=__random,
            website=f"{__doc}_generated/openturns.Uniform.html",
            settings_class=OT_RANDOM_Settings,
        ),
        __haselgrove: OpenTURNSAlgorithmDescription(
            algorithm_name=__haselgrove,
            description="Haselgrove sequence",
            internal_algorithm_name=__haselgrove,
            website=f"{__doc}_generated/openturns.HaselgroveSequence.html",
            settings_class=OT_HASELGROVE_Settings,
        ),
        __reverse_halton: OpenTURNSAlgorithmDescription(
            algorithm_name=__reverse_halton,
            description="Reverse Halton",
            internal_algorithm_name=__reverse_halton,
            website=f"{__doc}_generated/openturns.ReverseHaltonSequence.html",
            settings_class=OT_REVERSE_HALTON_Settings,
        ),
        __halton: OpenTURNSAlgorithmDescription(
            algorithm_name=__halton,
            description="Halton sequence",
            internal_algorithm_name=__halton,
            website=f"{__doc}_generated/openturns.HaltonSequence.html",
            settings_class=OT_HALTON_Settings,
        ),
        __faure: OpenTURNSAlgorithmDescription(
            algorithm_name=__faure,
            description="Faure sequence",
            internal_algorithm_name=__faure,
            website=f"{__doc}_generated/openturns.FaureSequence.html",
            settings_class=OT_FAURE_Settings,
        ),
        __monte_carlo: OpenTURNSAlgorithmDescription(
            algorithm_name=__monte_carlo,
            description="Monte Carlo sequence",
            internal_algorithm_name=__monte_carlo,
            website=f"{__doc}_generated/openturns.Uniform.html",
            settings_class=OT_MONTE_CARLO_Settings,
        ),
        __factorial: OpenTURNSAlgorithmDescription(
            algorithm_name=__factorial,
            description="Factorial design",
            internal_algorithm_name=__factorial,
            website=f"{__doc}_generated/openturns.Factorial.html",
            settings_class=OT_FACTORIAL_Settings,
        ),
        __composite: OpenTURNSAlgorithmDescription(
            algorithm_name=__composite,
            description="Composite design",
            internal_algorithm_name=__composite,
            website=f"{__doc}_generated/openturns.Composite.html",
            settings_class=OT_COMPOSITE_Settings,
        ),
        __axial: OpenTURNSAlgorithmDescription(
            algorithm_name=__axial,
            description="Axial design",
            internal_algorithm_name=__axial,
            website=f"{__doc}_generated/openturns.Axial.html",
            settings_class=OT_AXIAL_Settings,
        ),
        __opt_lhs: OpenTURNSAlgorithmDescription(
            algorithm_name=__opt_lhs,
            description="Optimal Latin Hypercube Sampling",
            internal_algorithm_name=__opt_lhs,
            website=f"{__doc}_generated/openturns.SimulatedAnnealingLHS.html",
            settings_class=OT_OPT_LHS_Settings,
        ),
        __lhs: OpenTURNSAlgorithmDescription(
            algorithm_name=__lhs,
            description="Latin Hypercube Sampling",
            internal_algorithm_name=__lhs,
            website=f"{__doc}_generated/openturns.LHSExperiment.html",
            settings_class=OT_LHS_Settings,
        ),
        __lhsc: OpenTURNSAlgorithmDescription(
            algorithm_name=__lhsc,
            description="Centered Latin Hypercube Sampling",
            internal_algorithm_name=__lhsc,
            website=f"{__doc}_generated/openturns.LHSExperiment.html",
            settings_class=OT_LHSC_Settings,
        ),
        __fullfact: OpenTURNSAlgorithmDescription(
            algorithm_name=__fullfact,
            description="Full factorial design",
            internal_algorithm_name=__fullfact,
            website=f"{__doc}_generated/openturns.Box.html",
            settings_class=OT_FULLFACT_Settings,
        ),
        __sobol_indices: OpenTURNSAlgorithmDescription(
            algorithm_name=__sobol_indices,
            description="DOE for Sobol indices",
            internal_algorithm_name=__sobol_indices,
            website=f"{__doc}_generated/openturns.SobolIndicesAlgorithm.html",
            settings_class=OT_SOBOL_INDICES_Settings,
        ),
    }

    def _generate_unit_samples(
        self,
        design_space: DesignSpace,
    ) -> NumberArray:
        openturns.RandomGenerator.SetSeed(self._seeder.get_seed(self._settings.seed))
        doe_algo = self.__names_to_classes[self._algo_name]()
        return doe_algo.generate_samples(design_space.dimension, self._settings)
