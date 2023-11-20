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
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The SciPy-based triangular distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution


class SPTriangularDistribution(SPDistribution):
    """The SciPy-based triangular distribution.

    Examples:
        >>> from gemseo.uncertainty.distributions.scipy.triangular import (
        ...     SPTriangularDistribution
        ... )
        >>> distribution = SPTriangularDistribution('x', -1, 0, 1)
        >>> print(distribution)
        triang(lower=-1, mode=0, upper=1)
    """

    def __init__(
        self,
        variable: str = SPDistribution.DEFAULT_VARIABLE_NAME,
        minimum: float = 0.0,
        mode: float = 0.5,
        maximum: float = 1.0,
        dimension: int = 1,
    ) -> None:
        """
        Args:
            minimum: The minimum of the triangular random variable.
            mode: The mode of the triangular random variable.
            maximum: The maximum of the triangular random variable.
        """  # noqa: D205,D212,D415
        super().__init__(
            variable,
            "triang",
            {
                "loc": minimum,
                "scale": maximum - minimum,
                "c": (mode - minimum) / float(maximum - minimum),
            },
            dimension,
            {
                self._LOWER: minimum,
                self._MODE: mode,
                self._UPPER: maximum,
            },
        )
