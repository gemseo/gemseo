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
"""The DOE used by a One-factor-at-a-Time (OAT) sensitivity analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Optional
from typing import TextIO
from typing import Union

from numpy import array

from gemseo.algos.doe.base_doe_library import BaseDOELibrary
from gemseo.algos.doe.base_doe_library import DOEAlgorithmDescription
from gemseo.typing import RealArray

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.parallel_execution.callable_parallel_execution import CallbackType

OptionType = Optional[Union[str, int, float, bool, list[str], Path, TextIO, RealArray]]


class OATDOE(BaseDOELibrary):
    r"""The DOE used by a One-factor-at-a-Time sensitivity analysis.

    The purpose of the OAT is to quantify the elementary effect

    .. math::

       df_i = f(X_1+dX_1,\ldots,X_{i-1}+dX_{i-1},X_i+dX_i,\ldots,X_d)
              -
              f(X_1+dX_1,\ldots,X_{i-1}+dX_{i-1},X_i,\ldots,X_d)

    associated with a small variation :math:`dX_i` of :math:`X_i` with

    .. math::

       df_1 = f(X_1+dX_1,\ldots,X_d)-f(X_1,\ldots,X_d)

    The elementary effects :math:`df_1,\ldots,df_d` are computed sequentially
    from an initial point

    .. math::

       X=(X_1,\ldots,X_d)

    From these elementary effects, we can compare their absolute values
    :math:`|df_1|,\ldots,|df_d|` and sort :math:`X_1,\ldots,X_d` accordingly.

    Note that
    |g| does not implement this sensitivity analysis
    but this DOE is used by the :class:`.MorrisAnalysis`,
    which repeats this sensitivity analysis
    and computes statistics from the repetitions.
    """

    ALGORITHM_INFOS: ClassVar[dict[str, DOEAlgorithmDescription]] = {
        "OATDOE": DOEAlgorithmDescription(
            algorithm_name="OATDOE",
            description="The DOE used by a One-factor-at-a-Time sensitivity analysis.",
            internal_algorithm_name="OATDOE",
            library_name="OATDOE",
        )
    }

    def __init__(self, algo_name: str = "OATDOE") -> None:  # noqa:D107
        super().__init__(algo_name)

    def _get_options(
        self,
        initial_point: RealArray,
        step: float = 0.05,
        max_time: float = 0,
        eval_jac: bool = False,
        n_processes: int = 1,
        wait_time_between_samples: float = 0.0,
        callbacks: Iterable[CallbackType] = (),
        **kwargs: OptionType,
    ) -> dict[str, OptionType]:
        """Set the options.

        Args:
            initial_point: The initial point of the OAT DOE.
            step: The relative step of the OAT DOE
                so that the step in the ``x`` direction is
                ``step*(max_x-min_x)`` if ``x+step*(max_x-min_x)<=max_x``
                and ``-step*(max_x-min_x)`` otherwise.
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
            initial_point=initial_point,
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
        step: float,
        initial_point: RealArray,
        **options: OptionType,
    ) -> RealArray:
        """
        Args:
            initial_point: The initial point of the OAT DOE.
            step: The relative step of the OAT DOE
                so that the step in the ``x`` direction is
                ``step*(max_x-min_x)`` if ``x+step*(max_x-min_x)<=max_x``
                and ``-step*(max_x-min_x)`` otherwise.
        """  # noqa: D205, D212
        points = [initial_point]
        for i in range(len(initial_point)):
            points.append(points[-1].copy())
            current_point = points[-1]
            if current_point[i] + step > 1:
                current_point[i] -= step
            else:
                current_point[i] += step

        return array(points)
