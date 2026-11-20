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
"""The parameters of the maximum-entropy distributions."""

from __future__ import annotations

from enum import StrEnum


class MaximumEntropyParameter(StrEnum):
    """A parameter of a maximum-entropy probability distribution."""

    MAXIMUM = "maximum"
    """The maximum of the random variable."""

    MINIMUM = "minimum"
    """The minimum of the random variable."""

    MU = "mu"
    """The mean of the distribution to be truncated, or of the normal distribution."""

    RATE = "rate"
    """The rate of the exponential distribution."""

    SIGMA = "sigma"
    """The standard deviation of the distribution to be truncated,
    or of the normal distribution."""

    TILT = "tilt"
    """The tilt of the tilted uniform distribution."""
