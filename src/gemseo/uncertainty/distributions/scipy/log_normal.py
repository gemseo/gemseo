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


class SPLogNormalDistribution(SPDistribution):
    """The SciPy-based log-normal distribution."""

    def __init__(
        self,
        mu: float = 1.0,
        sigma: float = 1.0,
        location: float = 0.0,
        set_log: bool = False,
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
        if set_log:
            log_mu, log_sigma = mu, sigma
        else:
            log_mu, log_sigma = compute_mu_l_and_sigma_l(mu, sigma, location)

        super().__init__(
            interfaced_distribution="lognorm",
            parameters={"s": log_sigma, "loc": location, "scale": exp(log_mu)},
            standard_parameters={self._MU: mu, self._SIGMA: sigma, self._LOC: location},
        )
