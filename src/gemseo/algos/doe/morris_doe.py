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
"""The DOE used by the Morris sensitivity analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Optional
from typing import TextIO
from typing import Union

from numpy import vstack

from gemseo.algos.doe.base_doe_library import BaseDOELibrary
from gemseo.algos.doe.base_doe_library import DOEAlgorithmDescription
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.typing import RealArray
from gemseo.typing import StrKeyMapping
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.parallel_execution.callable_parallel_execution import CallbackType

OptionType = Optional[Union[str, int, float, bool, list[str], Path, TextIO, RealArray]]


class MorrisDOE(BaseDOELibrary):
    """The DOE used by the Morris sensitivity analysis.

    This DOE algorithm applies the :class:`.OATDOE` algorithm at :math:`r` points.
    The number of samples is equal to :math:`r(1+d)`
    where :math:`d` is the space dimension.
    """

    ALGORITHM_INFOS: ClassVar[dict[str, DOEAlgorithmDescription]] = {
        "MorrisDOE": DOEAlgorithmDescription(
            algorithm_name="MorrisDOE",
            description="The DOE used by the Morris sensitivity analysis.",
            internal_algorithm_name="MorrisDOE",
            library_name="MorrisDOE",
        )
    }

    def __init__(self, algo_name: str = "MorrisDOE") -> None:  # noqa:D107
        super().__init__(algo_name)

    def _get_options(
        self,
        n_samples: int | None = None,
        doe_algo_name: str = "lhs",
        doe_algo_options: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        n_replicates: int = 5,
        step: float = 0.05,
        max_time: float = 0,
        eval_jac: bool = False,
        n_processes: int = 1,
        wait_time_between_samples: float = 0.0,
        callbacks: Iterable[CallbackType] = (),
        **kwargs: OptionType,
    ) -> dict[str, OptionType]:
        """Return the processed options.

        Args:
            n_samples: The maximum number of samples required by the user.
                If ``None``,
                deduce it from the design space dimension and ``n_replicates``.
            doe_algo_name: The name of the DOE algorithm to repeat the OAT DOE.
            doe_algo_options: The options of the DOE algorithm.
            n_replicates: The number of OAT repetitions.
            step: The relative step of the OAT DOE.
            eval_jac: Whether to evaluate the jacobian.
            n_processes: The maximum simultaneous number of processes
                used to parallelize the execution.
            wait_time_between_samples: The waiting time between two samples.
            max_time: The maximum runtime in seconds,
                disabled if 0.
            callbacks: The functions to be evaluated
                after each call to :meth:`.OptimizationProblem.evaluate_functions`;
                to be called as ``callback(index, (output, jacobian))``.
            **kwargs: The additional arguments.

        Returns:
            The processed options.
        """
        return self._process_options(
            n_samples=n_samples,
            doe_algo_name=doe_algo_name,
            doe_algo_options=doe_algo_options,
            n_replicates=n_replicates,
            step=step,
            max_time=max_time,
            eval_jac=eval_jac,
            n_processes=n_processes,
            wait_time_between_samples=wait_time_between_samples,
            callbacks=callbacks,
            **kwargs,
        )

    def _generate_unit_samples(
        self,
        design_space: DesignSpace,
        n_samples: int | None,
        doe_algo_name: str,
        doe_algo_options: StrKeyMapping,
        n_replicates: int,
        step: float,
        **options: OptionType,
    ) -> RealArray:
        """
        Args:
            n_samples: The maximum number of samples required by the user.
                If ``None``,
                deduce it from the design space dimension and ``n_replicates``.
            doe_algo_name: The name of the DOE algorithm to repeat the OAT DOE.
            doe_algo_options: The options of the DOE algorithm.
            n_replicates: The number of OAT repetitions.
            step: The relative step of the OAT DOE.

        Raises:
            ValueError: When the number of samples is lower than
                the dimension of the input space plus one.
        """  # noqa: D205, D212
        factory = DOELibraryFactory()
        doe_algo = factory.create(doe_algo_name)
        oat_algo = factory.create("OATDOE")
        dimension = design_space.dimension
        if n_samples:
            n_replicates = n_samples // (dimension + 1)
            if n_replicates == 0:
                msg = (
                    f"The number of samples ({n_samples}) must be "
                    "at least equal to the dimension of the input space plus one "
                    f"({dimension}+1={dimension + 1})."
                )
                raise ValueError(msg)

        initial_points = doe_algo.compute_doe(
            dimension, n_samples=n_replicates, unit_sampling=True, **doe_algo_options
        )
        return vstack([
            oat_algo.compute_doe(
                dimension, unit_sampling=True, step=step, initial_point=initial_point
            )
            for initial_point in initial_points
        ])
