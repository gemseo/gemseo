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
"""The families of the maximum-entropy distributions."""

from __future__ import annotations

from enum import StrEnum


class MaximumEntropyFamily(StrEnum):
    """A family of maximum-entropy probability distributions."""

    EXPONENTIAL = "exponential"
    """The exponential distributions, shifted to start at the minimum.

    They maximize the entropy for a minimum and a mean."""

    NORMAL = "normal"
    """The normal distributions.

    They maximize the entropy for a mean and a standard deviation."""

    REFLECTED_EXPONENTIAL = "reflected exponential"
    """The exponential distributions, reflected to end at the maximum.

    They maximize the entropy for a maximum and a mean."""

    TILTED_UNIFORM = "tilted uniform"
    r"""The uniform distributions, tilted to have a given mean.

    Their probability density function is proportional to $e^{\lambda x}$
    over the support, where $\lambda$ is the tilt.
    They maximize the entropy for a minimum, a maximum and a mean."""

    TRUNCATED_NORMAL = "truncated normal"
    """The normal distributions, truncated to the support.

    They maximize the entropy for a mean, a standard deviation
    and a minimum, a maximum or both."""

    UNIFORM = "uniform"
    """The uniform distributions.

    They maximize the entropy for a minimum and a maximum."""
