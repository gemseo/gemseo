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
"""The approximation modes."""

from __future__ import annotations

from strenum import StrEnum


class ApproximationMode(StrEnum):
    """The approximation derivation modes."""

    COMPLEX_STEP = "complex_step"
    """The complex step method used to approximate the Jacobians by perturbing each
    variable with a small complex number."""

    FINITE_DIFFERENCES = "finite_differences"
    """The finite differences method used to approximate the Jacobians by perturbing
    each variable with a small real number."""

    CENTERED_DIFFERENCES = "centered_differences"
    """The centered differences method used to approximate the Jacobians by perturbing
    each variable with a small real number."""
