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
"""The OpenTURNS-based log-normal distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions._log_normal_utils import compute_mu_l_and_sigma_l
from gemseo.uncertainty.distributions.base_settings.log_normal_settings import _LOCATION
from gemseo.uncertainty.distributions.base_settings.log_normal_settings import _MU
from gemseo.uncertainty.distributions.base_settings.log_normal_settings import _SET_LOG
from gemseo.uncertainty.distributions.base_settings.log_normal_settings import _SIGMA
from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    _LOWER_BOUND,
)
from gemseo.uncertainty.distributions.openturns.distribution_settings import _THRESHOLD
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    _TRANSFORMATION,
)
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    _UPPER_BOUND,
)
from gemseo.uncertainty.distributions.openturns.log_normal_settings import (
    OTLogNormalDistribution_Settings,
)


class OTLogNormalDistribution(OTDistribution):
    """The OpenTURNS-based log-normal distribution."""

    Settings = OTLogNormalDistribution_Settings

    def __init__(
        self,
        mu: float = _MU,
        sigma: float = _SIGMA,
        location: float = _LOCATION,
        set_log: bool = _SET_LOG,
        transformation: str = _TRANSFORMATION,
        lower_bound: float | None = _LOWER_BOUND,
        upper_bound: float | None = _UPPER_BOUND,
        threshold: float = _THRESHOLD,
        settings: OTLogNormalDistribution_Settings | None = None,
    ) -> None:
        """
        Args:
            mu: Either the mean of the log-normal random variable
                or that of its logarithm when ``set_log`` is ``True``.
            sigma: Either the standard deviation of the log-normal random variable
                or that of its logarithm when ``set_log`` is ``True``.
            location: The location of the log-normal random variable.
            set_log: Whether ``mu`` and ``sigma`` apply
                to the logarithm of the log-normal random variable.
                Otherwise,
                ``mu`` and ``sigma`` apply to the log-normal random variable directly.
        """  # noqa: D205,D212,D415
        if settings is None:
            settings = OTLogNormalDistribution_Settings(
                mu=mu,
                sigma=sigma,
                location=location,
                set_log=set_log,
                transformation=transformation,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                threshold=threshold,
            )

        if settings.set_log:
            log_mu, log_sigma = settings.mu, settings.sigma
        else:
            log_mu, log_sigma = compute_mu_l_and_sigma_l(
                settings.mu, settings.sigma, settings.location
            )

        super().__init__(
            interfaced_distribution="LogNormal",
            parameters=(log_mu, log_sigma, settings.location),
            standard_parameters={
                self._MU: settings.mu,
                self._SIGMA: settings.sigma,
                self._LOC: settings.location,
            },
            transformation=settings.transformation,
            lower_bound=settings.lower_bound,
            upper_bound=settings.upper_bound,
            threshold=settings.threshold,
        )
