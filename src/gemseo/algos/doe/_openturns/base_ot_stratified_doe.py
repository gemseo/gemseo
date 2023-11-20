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

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from numpy import array
from numpy import max as np_max
from numpy import min as np_min
from numpy import ndarray

from gemseo.algos.doe._openturns.base_ot_doe import BaseOTDOE
from gemseo.utils.string_tools import MultiLineString

if TYPE_CHECKING:
    from openturns import StratifiedExperiment

LOGGER = logging.getLogger(__name__)


class BaseOTStratifiedDOE(BaseOTDOE):
    """The base stratified DOE algorithm using the OpenTURNS library."""

    _ALGO_CLASS: ClassVar[type[StratifiedExperiment]]
    """The OpenTURNS class implementing the stratified DOE algorithm."""

    __LEVELS: Final[str] = "levels"
    __CENTERS: Final[str] = "centers"

    def __check_stratified_options(self, dimension: int, **options: Any) -> None:
        """Check that the mandatory inputs for the composite design are set.

        Args:
            dimension: The parameter space dimension.
            options: The options of the DOE.

        Raises:
            KeyError: If the key `levels` is not in `options`.
        """
        if self.__LEVELS not in options:
            raise KeyError(
                "Missing parameter 'levels', "
                "tuple of normalized levels in [0,1] you need in your design."
            )
        self.__check_and_cast_levels(**options)
        centers = options.get(self.__CENTERS)
        if len(centers) == 1:
            options[self.__CENTERS] = centers * dimension
        self.__check_and_cast_centers(dimension, **options)

    def __check_and_cast_levels(self, **options: Any) -> None:
        """Check that the options ``levels`` is properly defined and cast it to array.

        Args:
            **options: The DOE options.

        Raises:
            ValueError: When a level does not belong to [0, 1].
            TypeError: When the levels are neither a list nor a tuple.
        """
        levels = options[self.__LEVELS]
        if isinstance(levels, (list, tuple)):
            levels = array(levels)
            lower_bound = np_min(levels)
            upper_bound = np_max(levels)
            if lower_bound < 0.0 or upper_bound > 1.0:
                raise ValueError(
                    f"Levels must belong to [0, 1]; got [{lower_bound}, {upper_bound}]."
                )
            options[self.__LEVELS] = levels
        else:
            raise TypeError(
                "The argument 'levels' must be either a list or a tuple; "
                f"got a '{levels.__class__.__name__}'."
            )

    def __check_and_cast_centers(self, dimension: int, **options: Any) -> None:
        """Check that the options ``centers`` is properly defined and cast it to array.

        Args:
            dimension: The parameter space dimension.
            **options: The DOE options.

        Raises:
            ValueError: When the centers dimension has a wrong dimension.
            TypeError: When the centers are neither a list nor a tuple.
        """
        center = options[self.__CENTERS]
        if isinstance(center, (list, tuple)):
            if len(center) != dimension:
                raise ValueError(
                    "Inconsistent length of 'centers' list argument "
                    f"compared to design vector size: {dimension} vs {len(center)}."
                )
            options[self.__CENTERS] = array(center)
        else:
            raise TypeError(
                "Error for 'centers' definition in DOE design; "
                f"a tuple or a list is expected whereas {type(center)} is provided."
            )

    def generate_samples(  # noqa: D102
        self, n_samples: int, dimension: int, **options: Any
    ) -> ndarray:
        self.__check_stratified_options(dimension, **options)
        levels = options[self.__LEVELS]
        centers = options[self.__CENTERS]
        msg = MultiLineString()
        msg.add("Composite design:")
        msg.indent()
        msg.add("centers: {}", centers)
        msg.add("levels: {}", levels)
        LOGGER.debug("%s", msg)
        algo = self._ALGO_CLASS(centers, levels)
        samples = array(algo.generate())
        minimum = samples.min()
        maximum = samples.max()
        return (samples - minimum) / (maximum - minimum)
