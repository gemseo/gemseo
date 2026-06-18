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
    """The modes to approximate all the Jacobian blocks of a discipline."""

    COMPLEX_STEP = "complex_step"
    """Approximate all the Jacobian blocks with the complex-step method,
    perturbing each input with a small imaginary number."""

    FINITE_DIFFERENCES = "finite_differences"
    """Approximate all the Jacobian blocks with first-order forward finite differences,
    perturbing each input with a small real number."""

    CENTERED_DIFFERENCES = "centered_differences"
    """Approximate all the Jacobian blocks with second-order centered finite
    differences, perturbing each input on both sides with a small real number."""


class HybridApproximationMode(StrEnum):
    """The modes for semi-analytical computation of the Jacobian.

    The Jacobian blocks available analytically are computed analytically;
    only the blocks unavailable analytically are approximated.
    """

    HYBRID_COMPLEX_STEP = "hybrid_complex_step"
    """Approximate only the analytically-unavailable Jacobian blocks with the
    complex-step method, perturbing the related inputs with a small imaginary number."""

    HYBRID_FINITE_DIFFERENCES = "hybrid_finite_differences"
    """Approximate only the analytically-unavailable Jacobian blocks with first-order
    forward finite differences, perturbing the related inputs with a small real
    number."""

    HYBRID_CENTERED_DIFFERENCES = "hybrid_centered_differences"
    """Approximate only the analytically-unavailable Jacobian blocks with second-order
    centered finite differences, perturbing the related inputs on both sides with a
    small real number."""
