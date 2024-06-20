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
#        :author: Matthias De Lozzo
"""Build a diagonal DOE for scalable model construction."""

from __future__ import annotations

from collections.abc import Container
from collections.abc import Iterable
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Optional
from typing import Union

from numpy import hstack
from numpy import linspace
from numpy import newaxis

from gemseo.algos.doe.base_doe_library import BaseDOELibrary
from gemseo.algos.doe.base_doe_library import DOEAlgorithmDescription

if TYPE_CHECKING:
    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.parallel_execution.callable_parallel_execution import CallbackType
    from gemseo.typing import RealArray

OptionType = Optional[Union[str, int, float, bool, Container[str]]]


class DiagonalDOE(BaseDOELibrary):
    """Class used to create a diagonal DOE."""

    ALGORITHM_INFOS: ClassVar[dict[str, DOEAlgorithmDescription]] = {
        "DiagonalDOE": DOEAlgorithmDescription(
            algorithm_name="DiagonalDOE",
            description="Diagonal design of experiments",
            internal_algorithm_name="DiagonalDOE",
            library_name="GEMSEO",
        )
    }

    def __init__(self, algo_name: str = "DiagonalDOE") -> None:  # noqa:D107
        super().__init__(algo_name)

    def _get_options(
        self,
        eval_jac: bool = False,
        n_processes: int = 1,
        wait_time_between_samples: float = 0.0,
        n_samples: int = 2,
        reverse: Container[str] | None = None,
        max_time: float = 0,
        callbacks: Iterable[CallbackType] = (),
        **kwargs: OptionType,
    ) -> dict[str, OptionType]:  # pylint: disable=W0221
        """Get the options.

        Args:
            eval_jac: Whether to evaluate the Jacobian.
            n_processes: The maximum simultaneous number of processes
                used to parallelize the execution.
            wait_time_between_samples: The waiting time between two samples.
            n_samples: The number of samples.
                The number of samples must be greater than or equal to 2.
            reverse: The dimensions or variables to sample from their upper bounds to
                their lower bounds.
                If ``None``, every dimension will be sampled from its lower bound to its
                upper bound.
            max_time: The maximum runtime in seconds.
                If 0, no maximum runtime is set.
            callbacks: The functions to be evaluated
                after each call to :meth:`.OptimizationProblem.evaluate_functions`;
                to be called as ``callback(index, (output, jacobian))``.
            **kwargs: Additional arguments.

        Returns:
            The processed options.
        """
        return self._process_options(
            eval_jac=eval_jac,
            n_processes=n_processes,
            wait_time_between_samples=wait_time_between_samples,
            n_samples=n_samples,
            reverse=reverse,
            max_time=max_time,
            callbacks=callbacks,
            **kwargs,
        )

    def _generate_unit_samples(
        self, design_space: DesignSpace, **options: OptionType
    ) -> RealArray:
        """
        Raises:
            ValueError: If the number of samples is not set, or is lower than 2.
        """  # noqa: D205, D212, D415
        n_samples = options.get(self._N_SAMPLES)
        if n_samples is None or n_samples < 2:
            msg = (
                "The number of samples must set to a value greater than or equal to 2."
            )
            raise ValueError(msg)

        reverse = options.get("reverse", [])
        if reverse is None:
            reverse = []

        sizes = design_space.variable_sizes
        name_by_index = {}
        start = 0
        for name in design_space.variable_names:
            for index in range(start, start + sizes[name]):
                name_by_index[index] = name
            start += sizes[name]

        samples = []
        for index in range(design_space.dimension):
            if str(index) in reverse or name_by_index[index] in reverse:
                start = 1.0
                end = 0.0
            else:
                start = 0.0
                end = 1.0

            samples.append(linspace(start, end, n_samples)[:, newaxis])

        return hstack(samples)
