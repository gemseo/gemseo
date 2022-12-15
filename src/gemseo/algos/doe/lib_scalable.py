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

from typing import Container
from typing import Optional
from typing import Union

from numpy import hstack
from numpy import linspace
from numpy import ndarray

from gemseo.algos.doe.doe_lib import DOEAlgorithmDescription
from gemseo.algos.doe.doe_lib import DOELibrary

OptionType = Optional[Union[str, int, float, bool, Container[str]]]


class DiagonalDOE(DOELibrary):
    """Class used to create a diagonal DOE."""

    __ALGO_DESC = {"DiagonalDOE": "Diagonal design of experiments"}
    LIBRARY_NAME = "GEMSEO"

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        for algo, description in self.__ALGO_DESC.items():
            self.descriptions[algo] = DOEAlgorithmDescription(
                algorithm_name=algo,
                description=description,
                internal_algorithm_name=algo,
                library_name="GEMSEO",
            )

    def _get_options(
        self,
        eval_jac: bool = False,
        n_processes: int = 1,
        wait_time_between_samples: float = 0.0,
        n_samples: int = 2,
        reverse: Container[str] | None = None,
        max_time: float = 0,
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
                If None, every dimension will be sampled from its lower bound to its
                upper bound.
            max_time: The maximum runtime in seconds.
                If 0, no maximum runtime is set.
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
            **kwargs,
        )

    def _generate_samples(self, **options: OptionType) -> ndarray:
        """Generate the DOE samples.

        Args:
            **options: The options for the algorithm,
                see the associated JSON file.

        Returns:
            The samples.

        Raises:
            ValueError: If the number of samples is not set, or is lower than 2.
        """
        n_samples = options.get(self.N_SAMPLES)
        if n_samples is None or n_samples < 2:
            raise ValueError(
                "The number of samples must set to a value greater than or equal to 2."
            )

        reverse = options.get("reverse", [])
        if reverse is None:
            reverse = []

        sizes = options[self._VARIABLES_SIZES]
        name_by_index = {}
        start = 0
        for name in options[self._VARIABLES_NAMES]:
            for index in range(start, start + sizes[name]):
                name_by_index[index] = name
            start += sizes[name]

        samples = []
        for index in range(options[self.DIMENSION]):
            if str(index) in reverse or name_by_index[index] in reverse:
                start = 1.0
                end = 0.0
            else:
                start = 0.0
                end = 1.0

            samples.append(linspace(start, end, n_samples)[:, None])

        return hstack(samples)
