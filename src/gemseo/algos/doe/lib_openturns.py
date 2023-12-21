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

from collections import namedtuple
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Final
from typing import Optional
from typing import Union

import openturns
from numpy import ndarray

from gemseo.algos.doe._openturns.ot_axial_doe import OTAxialDOE
from gemseo.algos.doe._openturns.ot_centered_lhs import OTCenteredLHS
from gemseo.algos.doe._openturns.ot_composite_doe import OTCompositeDOE
from gemseo.algos.doe._openturns.ot_factorial_doe import OTFactorialDOE
from gemseo.algos.doe._openturns.ot_faure_sequence import OTFaureSequence
from gemseo.algos.doe._openturns.ot_full_factorial_doe import OTFullFactorialDOE
from gemseo.algos.doe._openturns.ot_halton_sequence import OTHaltonSequence
from gemseo.algos.doe._openturns.ot_haselgrove_sequence import OTHaselgroveSequence
from gemseo.algos.doe._openturns.ot_monte_carlo import OTMonteCarlo
from gemseo.algos.doe._openturns.ot_optimal_lhs import OTOptimalLHS
from gemseo.algos.doe._openturns.ot_reverse_halton_sequence import (
    OTReverseHaltonSequence,
)
from gemseo.algos.doe._openturns.ot_sobol_doe import OTSobolDOE
from gemseo.algos.doe._openturns.ot_sobol_sequence import OTSobolSequence
from gemseo.algos.doe._openturns.ot_standard_lhs import OTStandardLHS
from gemseo.algos.doe.doe_library import DOEAlgorithmDescription
from gemseo.algos.doe.doe_library import DOELibrary

if TYPE_CHECKING:
    from gemseo.algos.doe._openturns.base_ot_doe import BaseOTDOE

OptionType = Optional[Union[str, int, float, bool, Sequence[int], ndarray]]


_AlgoData = namedtuple("_AlgoData", ["description", "webpage", "doe_algo_class"])


class OpenTURNS(DOELibrary):
    """Library of OpenTURNS DOE algorithms."""

    __OT_WEBPAGE = (
        "http://openturns.github.io/openturns/latest/user_manual/"
        "_generated/openturns.{}.html"
    )

    # Available algorithm for DOE design
    OT_SOBOL = "OT_SOBOL"
    OT_RANDOM = "OT_RANDOM"
    OT_HASEL = "OT_HASELGROVE"
    OT_REVERSE_HALTON = "OT_REVERSE_HALTON"
    OT_HALTON = "OT_HALTON"
    OT_FAURE = "OT_FAURE"
    OT_MC = "OT_MONTE_CARLO"
    OT_FACTORIAL = "OT_FACTORIAL"
    OT_COMPOSITE = "OT_COMPOSITE"
    OT_AXIAL = "OT_AXIAL"
    OT_LHSO = "OT_OPT_LHS"
    OT_LHS = "OT_LHS"
    OT_LHSC = "OT_LHSC"
    OT_FULLFACT = "OT_FULLFACT"  # Box in openturns
    OT_SOBOL_INDICES = "OT_SOBOL_INDICES"
    __ALGO_NAMES_TO_ALGO_DATA: Final[dict[str, tuple[str, str, BaseOTDOE]]] = {
        OT_SOBOL: _AlgoData("Sobol sequence", "SobolSequence", OTSobolSequence),
        OT_RANDOM: _AlgoData("Random sampling", "Uniform", OTMonteCarlo),
        OT_HASEL: _AlgoData(
            "Haselgrove sequence", "HaselgroveSequence", OTHaselgroveSequence
        ),
        OT_REVERSE_HALTON: _AlgoData(
            "Reverse Halton",
            "ReverseHaltonSequence",
            OTReverseHaltonSequence,
        ),
        OT_HALTON: _AlgoData("Halton sequence", "HaltonSequence", OTHaltonSequence),
        OT_FAURE: _AlgoData("Faure sequence", "FaureSequence", OTFaureSequence),
        OT_MC: _AlgoData("Monte Carlo sequence", "Uniform", OTMonteCarlo),
        OT_FACTORIAL: _AlgoData("Factorial design", "Factorial", OTFactorialDOE),
        OT_COMPOSITE: _AlgoData("Composite design", "Composite", OTCompositeDOE),
        OT_AXIAL: _AlgoData("Axial design", "Axial", OTAxialDOE),
        OT_LHSO: _AlgoData(
            "Optimal Latin Hypercube Sampling",
            "SimulatedAnnealingLHS",
            OTOptimalLHS,
        ),
        OT_LHS: _AlgoData("Latin Hypercube Sampling", "LHS", OTStandardLHS),
        OT_LHSC: _AlgoData("Centered Latin Hypercube Sampling", "LHS", OTCenteredLHS),
        OT_FULLFACT: _AlgoData("Full factorial design", "Box", OTFullFactorialDOE),
        OT_SOBOL_INDICES: _AlgoData(
            "DOE for Sobol 'indices",
            "SobolIndicesAlgorithm",
            OTSobolDOE,
        ),
    }

    LIBRARY_NAME = "OpenTURNS"

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        for algo_name, algo_data in self.__ALGO_NAMES_TO_ALGO_DATA.items():
            self.descriptions[algo_name] = DOEAlgorithmDescription(
                algorithm_name=algo_name,
                description=algo_data.description,
                handle_integer_variables=True,
                internal_algorithm_name=algo_name,
                library_name=self.__class__.__name__,
                website=self.__OT_WEBPAGE.format(algo_data.webpage),
            )

    def _get_options(
        self,
        levels: int | Sequence[int] | None = None,
        centers: Sequence[int] | None = None,
        eval_jac: bool = False,
        n_samples: int | None = None,
        n_processes: int = 1,
        wait_time_between_samples: float = 0.0,
        criterion: OTOptimalLHS.SpaceFillingCriterion = OTOptimalLHS.SpaceFillingCriterion.C2,  # noqa: E501
        temperature: OTOptimalLHS.TemperatureProfile = OTOptimalLHS.TemperatureProfile.GEOMETRIC,  # noqa: E501
        annealing: bool = True,
        n_replicates: int = 1000,
        seed: int | None = None,
        max_time: float = 0,
        eval_second_order: bool = True,
        **kwargs: OptionType,
    ) -> dict[str, OptionType]:
        r"""Set the options.

        Args:
            levels: The levels. If there is a parameter ``n_samples``,
                the latter can be specified
                and the former set to its default value ``None``.
            centers: The centers for axial, factorial and composite designs.
                If ``None``, centers = 0.5.
            eval_jac: Whether to evaluate the jacobian.
            n_samples: The number of samples. If there is a parameter ``levels``,
                the latter can be specified
                and the former set to its default value ``None``.
            n_processes: The maximum simultaneous number of processes
                used to parallelize the execution.
            wait_time_between_samples: The waiting time between two samples.
            criterion: The space-filling criterion, either "C2", "PhiP" or "MinDist".
            temperature: The temperature profile for simulated annealing,
                either "Geometric" or "Linear".
            annealing: If ``True``, use simulated annealing to optimize LHS. Otherwise,
                use crude Monte Carlo.
            n_replicates: The number of Monte Carlo replicates to optimize LHS.
            seed: The seed value.
                If ``None``,
                use the seed of the library,
                namely :attr:`.OpenTURNS.seed`.
            max_time: The maximum runtime in seconds,
                disabled if 0.
            eval_second_order: Whether to build a DOE
                to evaluate also the second-order indices;
                otherwise,
                the DOE is designed for first- and total-order indices only.
            **kwargs: The additional arguments.

        Returns:
            The processed options.
        """
        return self._process_options(
            levels=levels,
            centers=centers or [0.5],
            eval_jac=eval_jac,
            n_samples=n_samples,
            n_processes=n_processes,
            wait_time_between_samples=wait_time_between_samples,
            criterion=criterion,
            temperature=temperature,
            annealing=annealing,
            n_replicates=n_replicates,
            seed=seed,
            max_time=max_time,
            eval_second_order=eval_second_order,
            **kwargs,
        )

    def _generate_samples(
        self,
        dimension: int,
        n_samples: int | None = None,
        seed: int | None = None,
        **options: OptionType,
    ) -> ndarray:
        """Generate the samples.

        Args:
            dimension: The dimension of the variables space.
            n_samples: The number of samples.
                If ``None``, set from the options.
            seed: The seed to be used.
                If ``None``, use :attr:`.seed`.
            **options: The options for the DOE algorithm, see associated JSON file.

        Returns:
            The samples for the DOE.
        """
        openturns.RandomGenerator.SetSeed(self._get_seed(seed))
        doe_algo = self.__ALGO_NAMES_TO_ALGO_DATA[self.algo_name].doe_algo_class()
        return doe_algo.generate_samples(n_samples, dimension, **options)
