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
"""The base stratified DOE algorithm using the OpenTURNS library."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import array
from numpy import full
from numpy import linspace
from numpy import where

from gemseo.algos.doe.openturns._algos.base_ot_doe import BaseOTDOE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openturns import StratifiedExperiment

    from gemseo.algos.doe.openturns.settings.base_ot_stratified_doe import (
        BaseOTStratifiedDOESettings,
    )
    from gemseo.typing import NumberArray


class BaseOTStratifiedDOE(BaseOTDOE):
    """The base stratified DOE algorithm using the OpenTURNS library."""

    _ALGO_CLASS: ClassVar[type[StratifiedExperiment]]
    """The OpenTURNS class implementing the stratified DOE algorithm."""

    def generate_samples(
        self,
        n_samples: int,
        dimension: int,
        centers: float | Sequence[float] = 0.5,
        levels: Sequence[float] = (),
        settings: BaseOTStratifiedDOESettings | None = None,
    ) -> NumberArray:
        r"""
        Args:
            n_samples: The maximum number of samples.
                If 0, deduce this number from ``dimension`` and ``levels``.
                Otherwise,
                the DOE will use the center of the unit hypercube
                and the levels and the effective number of samples
                will depend on the stratified DOE type.
                Ignored if ``settings`` is not `None`.
            centers: The center of the DOE in the unit hypercube.
                This argument is used when ``n_samples`` is greater than 0.
                If a real number is passed,
                create a ``dimension``-length vector filled with this value.
                Otherwise, the length of ``centers`` must be equal to ``dimension``.
                Ignored if ``settings`` is not `None`.
            levels: The relative positions of the levels
                This argument is used when ``n_samples`` is greater than 0.
                between the center and the bounds
                E.g. [0.2, 0.8] with [0.5] as center
                will generate the values [0.1, 0.4, 0.5, 0.6, 0.9].
                Ignored if ``settings`` is not `None`.
            settings: The settings of the DOE algorithm.

        Raises:
            ValueError: When the number of centers is different from the dimension,
                when a center is outside :math:`]0,1[`
                or when a level is outside :math:`[0,1]`.
        """  # noqa: D205, D212
        if settings is not None:
            n_samples = settings.n_samples
            centers = settings.centers
            levels = settings.levels

        if n_samples > 0:
            n_levels = self._compute_n_levels(n_samples, dimension)
            centers = full(dimension, 0.5)
            levels = linspace(0, 1, n_levels + 1)[1:]
        elif isinstance(centers, float):
            centers = [centers] * dimension
            levels = levels
        elif len(centers) == 1:
            centers = list(centers) * dimension

        n_centers = len(centers)
        if n_centers != dimension:
            msg = f"The number of centers must be {dimension}; got {n_centers}."
            raise ValueError(msg)

        if any(not 0 < center < 1 for center in centers):
            msg = f"The centers must be in ]0,1[; got {centers}."
            raise ValueError(msg)

        if any(not 0 < level <= 1 for level in levels):
            msg = f"The levels must be in ]0,1]; got {levels}."
            raise ValueError(msg)

        # We use openturns.StratifiedExperiment(center, levels).
        # where center is the center of the unit hypercube
        # and levels have different meanings
        # according to the nature of the stratified experiment.
        algo = self._ALGO_CLASS(full(dimension, 0.5), array(levels) / 2)
        samples = (array(algo.generate()) - 0.5) * 2
        centers = array(centers)
        return where(
            samples >= 0,
            centers + samples * (1 - centers),
            centers + samples * centers,
        )

    @staticmethod
    @abstractmethod
    def _compute_n_levels(n_samples: int, dimension: int) -> int:
        """Compute the number of levels.

        Args:
            n_samples: The number of samples.
            dimension: The dimension of the space.

        Returns:
            The number of levels.
        """
