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
"""The SciPy-based normal distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution


class SPNormalDistribution(SPDistribution):
    """The SciPy-based normal distribution."""

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
    ) -> None:
        """
        Args:
            mu: The mean of the normal random variable.
            sigma: The standard deviation of the normal random variable.
        """  # noqa: D205,D212,D415
        super().__init__(
            interfaced_distribution="norm",
            parameters={"loc": mu, "scale": sigma},
            standard_parameters={self._MU: mu, self._SIGMA: sigma},
        )
