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
"""The SciPy-based Beta distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.base_settings.beta_settings import _ALPHA
from gemseo.uncertainty.distributions.base_settings.beta_settings import _BETA
from gemseo.uncertainty.distributions.base_settings.beta_settings import _MAXIMUM
from gemseo.uncertainty.distributions.base_settings.beta_settings import _MINIMUM
from gemseo.uncertainty.distributions.scipy.beta_settings import (
    SPBetaDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution


class SPBetaDistribution(SPDistribution):
    """The SciPy-based Beta distribution."""

    Settings = SPBetaDistribution_Settings

    def __init__(
        self,
        alpha: float = _ALPHA,
        beta: float = _BETA,
        minimum: float = _MINIMUM,
        maximum: float = _MAXIMUM,
        settings: SPBetaDistribution_Settings | None = None,
    ) -> None:
        """
        Args:
            alpha: The first shape parameter of the beta random variable.
            beta: The second shape parameter of the beta random variable.
            minimum: The minimum of the beta random variable.
            maximum: The maximum of the beta random variable.
        """  # noqa: D205,D212,D415
        if settings is None and settings is None:
            settings = SPBetaDistribution_Settings(
                alpha=alpha,
                beta=beta,
                minimum=minimum,
                maximum=maximum,
            )
        super().__init__(
            interfaced_distribution="beta",
            parameters={
                "a": settings.alpha,
                "b": settings.beta,
                "loc": settings.minimum,
                "scale": settings.maximum - settings.minimum,
            },
            standard_parameters={
                self._LOWER: settings.minimum,
                self._UPPER: settings.maximum,
                self._ALPHA: settings.alpha,
                self._BETA: settings.beta,
            },
        )
