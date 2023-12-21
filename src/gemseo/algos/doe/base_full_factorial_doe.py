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
"""The base class of a full-factorial DOE."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

from gemseo.algos.doe._base_doe import BaseDOE

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray

LOGGER = logging.getLogger(__name__)


class BaseFullFactorialDOE(BaseDOE):
    """The base class of a full-factorial DOE."""

    def generate_samples(
        self, n_samples: int, dimension: int, **options: Any
    ) -> ndarray:
        """Generate samples.

        Args:
            n_samples: The number of samples.
            dimension: The dimension of the sampling space.
            **options: The options of the DOE algorithm.

        Returns:
            The samples.
        """
        return self._generate_fullfact(n_samples, dimension, **options)

    def _generate_fullfact(
        self,
        n_samples: int | None,
        dimension: int,
        **options: int | Iterable[int] | None,
    ) -> ndarray:
        """Generate a full-factorial DOE.

        Generate a full-factorial DOE based on either the number of samples,
        or the number of levels per input direction.
        When the number of samples is prescribed,
        the levels are deduced and are uniformly distributed among all the inputs.

        Args:
            dimension: The dimension of the parameter space.
            n_samples: The maximum number of samples from which the number of levels
                per input is deduced.
                The number of samples which is finally applied
                is the product of the numbers of levels.
                If ``None``, the algorithm uses the number of levels per input dimension
                provided by the argument ``levels``.
            levels: The number of levels per input direction.
                If ``levels`` is given as a scalar value, the same number of
                levels is used for all the inputs.
                If ``None``, the number of samples provided in argument ``n_samples``
                is used in order to deduce the levels.
            **options: The options of the DOE algorithm.

        Returns:
            The values of the DOE.

        Raises:
            ValueError:
                * If neither ``n_samples`` nor ``levels`` are provided.
                * If both ``n_samples`` and ``levels`` are provided.
        """
        levels = options.pop("levels", None)

        if not levels and not n_samples:
            raise ValueError(
                "Either 'n_samples' or 'levels' is required as an input "
                "parameter for the full-factorial DOE."
            )

        if levels and n_samples:
            raise ValueError(
                "Only one input parameter among 'n_samples' and 'levels' "
                "must be given for the full-factorial DOE."
            )

        if n_samples is not None:
            levels = self._compute_fullfact_levels(n_samples, dimension)

        if isinstance(levels, int):
            levels = [levels] * dimension

        return self._generate_fullfact_from_levels(levels)

    @abstractmethod
    def _generate_fullfact_from_levels(self, levels) -> ndarray:
        """Generate the full-factorial DOE from levels per input direction.

        Args:
            levels: The number of levels per input direction.

        Returns:
            The values of the DOE.
        """

    @staticmethod
    def _compute_fullfact_levels(n_samples: int, dimension: int) -> list[int]:
        """Compute the number of levels per input dimension for a full factorial design.

        Args:
            n_samples: The number of samples.
            dimension: The dimension of the input space.

        Returns:
            The number of levels per input dimension.
        """
        n_samples_dir = int(n_samples ** (1.0 / dimension))

        # Check for numerical precision issues,
        # e.g. int(10000**(1/3)) = int(9.999999999...) = 9 instead of 10
        # and correct if necessary.
        n_samples_dir_plus_one = n_samples_dir + 1
        if n_samples_dir_plus_one**dimension == n_samples:
            n_samples_dir = n_samples_dir_plus_one

        final_n_samples = n_samples_dir**dimension
        if final_n_samples != n_samples:
            LOGGER.warning(
                (
                    "A full-factorial DOE of %s samples in dimension %s does not exist;"
                    " use %s samples instead, "
                    "i.e. the largest %s-th integer power less than %s."
                ),
                n_samples,
                dimension,
                final_n_samples,
                dimension,
                n_samples,
            )
        return [n_samples_dir] * dimension
