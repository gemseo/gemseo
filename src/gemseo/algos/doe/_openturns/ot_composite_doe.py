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
"""The composite DOE algorithm."""

from __future__ import annotations

import logging
from typing import ClassVar
from typing import Final

from openturns import Composite

from gemseo.algos.doe._openturns.base_ot_stratified_doe import BaseOTStratifiedDOE

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


class OTCompositeDOE(BaseOTStratifiedDOE):
    """The composite DOE algorithm.

    .. note:: This class is a singleton.
    """

    _ALGO_CLASS: ClassVar[type[Composite]] = Composite

    @staticmethod
    def _compute_n_levels(n_samples: int, dimension: int) -> int:
        """
        Raises:
            ValueError: When the number of samples is too small.
        """  # noqa: D205, D212
        n_levels = int((n_samples - 1) / (2 * dimension + 2**dimension))
        if n_levels < 1:
            msg = (
                f"A composite DOE in dimension d={dimension} "
                "requires at least "
                f"1+2*d+2^d={1 + 2 * dimension + 2**dimension} samples; "
                f"got {n_samples}."
            )
            raise ValueError(msg)

        final_n_samples = 1 + n_levels * (2 * dimension + 2**dimension)
        if n_samples > final_n_samples:
            LOGGER.warning(
                (
                    "A composite DOE of %s samples in dimension %s does not exist; "
                    "use %s samples instead."
                ),
                n_samples,
                dimension,
                final_n_samples,
            )

        return n_levels
