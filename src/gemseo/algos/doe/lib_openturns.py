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

from collections.abc import Iterable
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import NamedTuple
from typing import Optional
from typing import Union

import openturns

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
from gemseo.algos.doe.base_doe_library import BaseDOELibrary
from gemseo.algos.doe.base_doe_library import DOEAlgorithmDescription
from gemseo.typing import RealArray

if TYPE_CHECKING:
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.doe._openturns.base_ot_doe import BaseOTDOE
    from gemseo.core.parallel_execution.callable_parallel_execution import CallbackType
    from gemseo.typing import NumberArray

OptionType = Optional[Union[str, int, float, bool, Sequence[int], RealArray]]


class _AlgoData(NamedTuple):
    description: str
    webpage: str
    doe_algo_class: type


class OpenTURNS(BaseDOELibrary):
    """Library of OpenTURNS DOE algorithms."""

    __ALGO_NAMES_TO_ALGO_DATA: Final[dict[str, tuple[str, str, BaseOTDOE]]] = {
        "OT_SOBOL": _AlgoData("Sobol sequence", "SobolSequence", OTSobolSequence),
        "OT_RANDOM": _AlgoData("Random sampling", "Uniform", OTMonteCarlo),
        "OT_HASELGROVE": _AlgoData(
            "Haselgrove sequence", "HaselgroveSequence", OTHaselgroveSequence
        ),
        "OT_REVERSE_HALTON": _AlgoData(
            "Reverse Halton",
            "ReverseHaltonSequence",
            OTReverseHaltonSequence,
        ),
        "OT_HALTON": _AlgoData("Halton sequence", "HaltonSequence", OTHaltonSequence),
        "OT_FAURE": _AlgoData("Faure sequence", "FaureSequence", OTFaureSequence),
        "OT_MONTE_CARLO": _AlgoData("Monte Carlo sequence", "Uniform", OTMonteCarlo),
        "OT_FACTORIAL": _AlgoData("Factorial design", "Factorial", OTFactorialDOE),
        "OT_COMPOSITE": _AlgoData("Composite design", "Composite", OTCompositeDOE),
        "OT_AXIAL": _AlgoData("Axial design", "Axial", OTAxialDOE),
        "OT_OPT_LHS": _AlgoData(
            "Optimal Latin Hypercube Sampling",
            "SimulatedAnnealingLHS",
            OTOptimalLHS,
        ),
        "OT_LHS": _AlgoData("Latin Hypercube Sampling", "LHS", OTStandardLHS),
        "OT_LHSC": _AlgoData("Centered Latin Hypercube Sampling", "LHS", OTCenteredLHS),
        "OT_FULLFACT": _AlgoData("Full factorial design", "Box", OTFullFactorialDOE),
        "OT_SOBOL_INDICES": _AlgoData(
            "DOE for Sobol 'indices",
            "SobolIndicesAlgorithm",
            OTSobolDOE,
        ),
    }

    ALGORITHM_INFOS: ClassVar[dict[str, DOEAlgorithmDescription]] = {
        algo_name: DOEAlgorithmDescription(
            algorithm_name=algo_name,
            description=algo_data.description,
            handle_integer_variables=True,
            internal_algorithm_name=algo_name,
            library_name="OpenTURNS",
            website=(
                "http://openturns.github.io/openturns/latest/user_manual/"
                f"_generated/openturns.{algo_data.webpage}.html"
            ),
        )
        for algo_name, algo_data in __ALGO_NAMES_TO_ALGO_DATA.items()
    }

    def _get_options(
        self,
        levels: float | Sequence[float] | None = None,
        centers: Sequence[float] | float = 0.5,
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
        callbacks: Iterable[CallbackType] = (),
        **kwargs: OptionType,
    ) -> dict[str, OptionType]:
        r"""Set the options.

        Args:
            levels: 1) In the case of axial, composite and factorial DOEs,
                the positions of the levels relative to the center;
                the levels will be equispaced and symmetrical relative to the center;
                e.g. ``[0.2, 0.8]`` in dimension 1 will generate
                the samples ``[0.15, 0.6, 0.75, 0.8, 0.95, 1]`` for an axial DOE;
                the values must be in :math:`]0,1]`.
                2) In the case of a full-factorial DOE,
                the number of levels per input direction;
                if scalar, this value is applied to each input direction.
            centers: The center of the unit hypercube
                that the axial, composite or factorial DOE algorithm will sample;
                if scalar, this value is applied to each direction of the hypercube;
                the values must be in :math:`]0,1[`.
            eval_jac: Whether to evaluate the jacobian.
            n_samples: The maximum number of samples required by the user;
                for axial, composite and factorial DOEs,
                a minimum number of samples is required
                and depends on the dimension of the space to sample;
                if ``None``
                in the case of for axial, composite, factorial and full-factorial DOEs
                the effective number of samples is computed
                from this dimension and the number of levels.
            n_processes: The maximum simultaneous number of processes
                used to parallelize the execution.
            wait_time_between_samples: The waiting time between two samples.
            criterion: The space-filling criterion, either "C2", "PhiP" or "MinDist".
            temperature: The temperature profile for simulated annealing,
                either "Geometric" or "Linear".
            annealing: If ``True``, use simulated annealing to optimize LHS. Otherwise,
                use crude Monte Carlo.
            n_replicates: The number of Monte Carlo replicates to optimize LHS.
            seed: The seed used for reproducibility reasons.
                If ``None``, use :attr:`.seed`.
            max_time: The maximum runtime in seconds,
                disabled if 0.
            eval_second_order: Whether to build a DOE
                to evaluate also the second-order indices;
                otherwise,
                the DOE is designed for first- and total-order indices only.
            callbacks: The functions to be evaluated
                after each call to :meth:`.OptimizationProblem.evaluate_functions`;
                to be called as ``callback(index, (output, jacobian))``.
            **kwargs: The additional arguments.

        Returns:
            The processed options.
        """
        return self._process_options(
            levels=levels,
            centers=centers,
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
            callbacks=callbacks,
            **kwargs,
        )

    def _generate_unit_samples(
        self,
        design_space: DesignSpace,
        n_samples: int | None = None,
        seed: int | None = None,
        **options: OptionType,
    ) -> NumberArray:
        """
        Args:
            dimension: The dimension of the variables space.
            n_samples: The number of samples.
                If ``None``, set from the options.
            seed: The seed used for reproducibility reasons.
                If ``None``, use :attr:`.seed`.
            **options: The options for the DOE algorithm, see associated JSON file.
        """  # noqa: D205, D212, D415
        openturns.RandomGenerator.SetSeed(self._seeder.get_seed(seed))
        doe_algo = self.__ALGO_NAMES_TO_ALGO_DATA[self._algo_name].doe_algo_class()
        return doe_algo.generate_samples(n_samples, design_space.dimension, **options)
