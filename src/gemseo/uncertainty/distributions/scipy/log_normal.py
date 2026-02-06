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
"""The SciPy-based log-normal distribution."""

from __future__ import annotations

from numpy import exp

from gemseo.uncertainty.distributions._log_normal_utils import compute_mu_l_and_sigma_l
from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.distribution_settings import (
    SPDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.log_normal_settings import (
    SPLogNormalDistribution_Settings,
)


class SPLogNormalDistribution(SPDistribution):
    """The SciPy-based log-normal distribution."""

    settings_class = SPLogNormalDistribution_Settings

    def __init__(  # noqa: D107
        self, settings: SPLogNormalDistribution_Settings | None = None
    ) -> None:
        if settings is None:
            settings = SPLogNormalDistribution_Settings()

        if settings.set_log:
            log_mu, log_sigma = settings.mu, settings.sigma
        else:
            log_mu, log_sigma = compute_mu_l_and_sigma_l(
                settings.mu, settings.sigma, settings.location
            )

        super().__init__(
            SPDistribution_Settings(
                interfaced_distribution="lognorm",
                parameters={
                    "s": log_sigma,
                    "loc": settings.location,
                    "scale": exp(log_mu),
                },
                standard_parameters={
                    self._MU: settings.mu,
                    self._SIGMA: settings.sigma,
                    self._LOC: settings.location,
                },
            )
        )
