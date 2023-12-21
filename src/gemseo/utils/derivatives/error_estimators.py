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
"""Error estimators for computing derivatives."""

from __future__ import annotations

import logging
from typing import Final

from numpy import finfo
from numpy import ndarray

LOGGER = logging.getLogger(__name__)
EPSILON: Final[float] = finfo(float).eps


def compute_truncature_error(
    hess: ndarray,
    step: float,
) -> ndarray:
    r"""Compute the truncation error.

    Defined for a first order finite differences scheme.

    Args:
        hess: The second-order derivative :math:`d^2f/dx^2`.
        step: The differentiation step.

    Returns:
        The truncation error.
    """
    return abs(hess) * step / 2


def compute_cancellation_error(
    f_x: ndarray,
    step: float,
    epsilon_mach=EPSILON,
) -> ndarray:
    r"""Compute the cancellation error.

    This is the round-off when doing :math:`f(x+\\delta_x)-f(x)`.

    Args:
        f_x: The value of the function at the current step :math:`x`.
        step: The step used for the calculations of the perturbed functions values.
        epsilon_mach: The machine epsilon.

    Returns:
        The cancellation error.
    """
    return 2 * epsilon_mach * abs(f_x) / step


def compute_hessian_approximation(
    f_p: ndarray,
    f_x: ndarray,
    f_m: ndarray,
    step: float,
) -> ndarray:
    r"""Compute the second-order approximation of the Hessian matrix :math:`d^2f/dx^2`.

    Args:
        f_p: The value of the function :math:`f` at the next step :math:`x+\\delta_x`.
        f_x: The value of the function :math:`f` at the current step :math:`x`.
        f_m: The value of the function :math:`f` at the previous step
            :math:`x-\\delta_x`.
        step: The differentiation step :math:`\\delta_x`.

    Returns:
        The approximation of the Hessian matrix at the current step :math:`x`.
    """
    return (f_p - 2 * f_x + f_m) / step**2


def compute_best_step(
    f_p: ndarray,
    f_x: ndarray,
    f_m: ndarray,
    step: float,
    epsilon_mach: float = EPSILON,
) -> tuple[ndarray | None, ndarray | None, float]:
    r"""Compute the optimal step for finite differentiation.

    Applied to a forward first order finite differences gradient approximation.

    Require a first evaluation of the perturbed functions values.

    The optimal step is reached when the truncation error
    (cut in the Taylor development),
    and the numerical cancellation errors
    (round-off when doing :math:`f(x+step)-f(x))` are equal.

    See Also:
        https://en.wikipedia.org/wiki/Numerical_differentiation
        and *Numerical Algorithms and Digital Representation*,
        Knut Morken, Chapter 11, "Numerical Differenciation"

    Args:
        f_p: The value of the function :math:`f` at the next step :math:`x+\\delta_x`.
        f_x: The value of the function :math:`f` at the current step :math:`x`.
        f_m: The value of the function :math:`f` at the previous step
            :math:`x-\\delta_x`.
        step: The differentiation step :math:`\\delta_x`.

    Returns:
        The estimation of the truncation error.
        None if the Hessian approximation is too small to compute the optimal step.
        The estimation of the cancellation error.
        None if the Hessian approximation is too small to compute the optimal step.
        The optimal step.
    """
    hess = compute_hessian_approximation(f_p, f_x, f_m, step)

    if abs(hess) < 1e-10:
        LOGGER.debug("Hessian approximation is too small, can't compute optimal step.")
        return None, None, step

    opt_step = 2 * (epsilon_mach * abs(f_x) / abs(hess)) ** 0.5
    trunc_error = compute_truncature_error(hess, step)
    cancel_error = compute_cancellation_error(f_x, opt_step)
    return trunc_error, cancel_error, opt_step
