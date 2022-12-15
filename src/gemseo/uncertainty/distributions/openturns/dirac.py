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
#        :author: Reda El Amri
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The Dirac distribution based on OpenTURNS."""
from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution


class OTDiracDistribution(OTDistribution):
    """The Dirac distribution."""

    def __init__(
        self,
        variable: str,
        variable_value: float = 0.0,
        dimension: int = 1,
        transformation: str | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        threshold: float = 0.5,
    ) -> None:
        """
        Args:
            variable_value: The value of the random variable.
        """  # noqa: D205,D212,D415
        super().__init__(
            variable,
            "Dirac",
            (variable_value,),
            dimension=dimension,
            standard_parameters={self._LOC: variable_value},
            transformation=transformation,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
