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
"""Utils for the log-normal distribution."""

from __future__ import annotations

from numpy import log


def compute_mu_l_and_sigma_l(
    mu: float,
    sigma: float,
    location: float,
) -> tuple[float, float]:
    """Compute the mean and standard deviation of the random variable's logarithm.

    Args:
        mu: The mean of the log-normal random variable.
        sigma: The standard deviation of the log-normal random variable.
        location: The location of the log-normal random variable.

    Returns:
        The mean and standard deviation of
        the logarithm of the log-normal random variable.
    """
    mu_location = mu - location
    mu_l = log(mu_location / ((sigma / mu_location) ** 2 + 1) ** 0.5)
    sigma_l = (2 * (log(mu_location) - mu_l)) ** 0.5
    return mu_l, sigma_l
