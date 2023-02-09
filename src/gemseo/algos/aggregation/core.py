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
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Constraints aggregation core functions."""
from __future__ import annotations

from math import log
from typing import Sequence

from numpy import argmax as np_argmax
from numpy import array
from numpy import atleast_2d
from numpy import exp as np_exp
from numpy import full
from numpy import heaviside
from numpy import max as np_max
from numpy import multiply
from numpy import ndarray
from numpy import sum as np_sum
from numpy import zeros


# TODO: API: rename to compute_ks_agg
def ks_agg(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> float:
    """Transform a vector of constraint functions into a KS function.

    The Kreisselmeier–Steinhauser function tends to the maximum operator
        when the aggregation parameter tends to infinity.

    Kreisselmeier G, Steinhauser R (1983)
    Application of Vector Performance Optimization
    to a Robust Control Loop Design for a Fighter Aircraft.
    International Journal of Control 37(2):251–284,
    doi:10.1080/00207179.1983.9753066

    Graeme J. Kennedy, Jason E. Hicken,
    Improved constraint-aggregation methods,
    Computer Methods in Applied Mechanics and Engineering,
    Volume 289,
    2015,
    Pages 332-354,
    ISSN 0045-7825,
    https://doi.org/10.1016/j.cma.2015.02.017.
    (http://www.sciencedirect.com/science/article/pii/S0045782515000663)

    Args:
        orig_val: The original constraint values.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        rho: The aggregation parameter.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The KS function value.
    """
    if indices is not None:
        orig_val = orig_val[indices]

    orig_val *= scale
    alpha = len(orig_val)
    m = max(orig_val)

    return (
        m
        + (1.0 / rho) * log((1.0 / alpha) * sum(np_exp(rho * (orig_val + 1.0 - m))))
        - 1.0
    )


# TODO: API: rename to compute_total_ks_agg_jac
def ks_agg_jac_v(
    orig_val: ndarray,
    orig_jac: ndarray,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Compute the Jacobian of KS function with respect to constraint function inputs.

    See :cite:`kennedy2015improved` and  :cite:`kreisselmeier1983application`.

    Args:
        orig_val: The original constraint values.
        orig_jac: The original constraint jacobian.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        rho: The multiplicative parameter in the exponential.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The Jacobian of KS function with respect to constraint function inputs.
    """
    if indices is not None:
        orig_jac = orig_jac[indices, :]
        orig_val = orig_val[indices]

    orig_jac *= scale
    orig_val *= scale

    m = max(orig_val)
    div = np_sum(np_exp(rho * (orig_val + 1.0 - m)))
    weights = np_exp(rho * (orig_val + 1.0 - m)).T / div
    return np_sum(multiply(atleast_2d(weights).T, orig_jac), axis=0)


# TODO: API: rename to compute_partial_ks_agg_jac
def ks_agg_jac(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Compute the Jacobian of KS function with respect to constraint functions.

    See :cite:`kennedy2015improved` and  :cite:`kreisselmeier1983application`.

    Args:
        orig_val: The original constraint values.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        rho: The multiplicative parameter in the exponential.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The Jacobian of KS function with respect to constraint functions.
    """
    full_size = orig_val.size
    if indices is not None:
        orig_val = orig_val[indices]

    orig_val *= scale

    m = max(orig_val)
    div = np_sum(np_exp(rho * (orig_val + 1.0 - m)))
    weights = np_exp(rho * (orig_val + 1.0 - m)).T / div
    der = atleast_2d(multiply(weights, scale))
    return __filter_jac(der, full_size, indices)


# TODO: API: rename to compute_iks_agg
def iks_agg(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> float:
    """Transform a vector of constraint functions into an induces exponential function.

    The induces exponential function (IKS) tends to the maximum operator when
        the aggregation parameter tends to infinity.

    See :cite:`kennedy2015improved`.

    Args:
        orig_val: The original constraint values.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        rho: The multiplicative parameter in the exponential.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The IKS function value.
    """
    if indices is not None:
        orig_val = orig_val[indices]

    orig_val *= scale

    m = max(orig_val)
    iks = sum(orig_val * np_exp(rho * (orig_val + 1.0 - m)))
    iks /= sum(np_exp(rho * (orig_val + 1.0 - m)))

    return iks


# TODO: API: rename to compute_total_iks_agg_jac
def iks_agg_jac_v(
    orig_val: ndarray,
    orig_jac: ndarray,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Compute the Jacobian of IKS function with respect to constraints inputs.

    See :cite:`kennedy2015improved`.

    Args:
        orig_val: The original constraint values.
        orig_jac: The original constraint jacobian.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        rho: The multiplicative parameter in the exponential.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The Jacobian of IKS function with respect to constraints inputs.
    """
    if indices is not None:
        orig_jac = orig_jac[indices, :]
        orig_val = orig_val[indices]

    orig_jac *= scale
    orig_val *= scale

    m = max(orig_val)

    iks_num = sum(orig_val * np_exp(rho * (orig_val + 1.0 - m)))
    iks_den = sum(np_exp(rho * (orig_val + 1.0 - m)))

    iks_num_der = np_sum(
        multiply(atleast_2d(np_exp(rho * (orig_val + 1.0 - m))).T, orig_jac), axis=0
    )
    iks_num_der += np_sum(
        multiply(
            atleast_2d(np_exp(rho * (orig_val + 1.0 - m)) * orig_val).T,
            rho * orig_jac,
        ),
        axis=0,
    )
    iks_den_der = np_sum(
        multiply(atleast_2d(np_exp(rho * (orig_val + 1.0 - m))).T, rho * orig_jac),
        axis=0,
    )
    return (-iks_den_der / iks_den**2) * iks_num + iks_num_der / iks_den


# TODO: API: rename to compute_partial_iks_agg_jac
def iks_agg_jac(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Compute the Jacobian of IKS function with respect to constraints functions.

    Kennedy, Graeme J., and Jason E. Hicken.
    "Improved constraint-aggregation methods."
    Computer Methods in Applied Mechanics and Engineering
    289 (2015): 332-354.

    Args:
        orig_val: The original constraint values.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        rho: The multiplicative parameter in the exponential.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The Jacobian of IKS function with respect to constraints functions.
    """
    full_size = orig_val.size
    if indices is not None:
        orig_val = orig_val[indices]

    orig_val *= scale

    m = max(orig_val)

    iks_num = sum(orig_val * np_exp(rho * (orig_val + 1.0 - m)))
    iks_den = sum(np_exp(rho * (orig_val + 1.0 - m)))

    iks_num_der = atleast_2d(np_exp(rho * (orig_val + 1.0 - m)))
    iks_num_der += rho * atleast_2d(np_exp(rho * (orig_val + 1.0 - m)) * orig_val)
    iks_den_der = rho * atleast_2d(np_exp(rho * (orig_val + 1.0 - m)))
    iks_d = atleast_2d((-iks_den_der / iks_den**2) * iks_num + iks_num_der / iks_den)

    iks_d = multiply(iks_d, scale)

    return __filter_jac(iks_d, full_size, indices)


def __filter_jac(
    orig_jac: ndarray,
    full_size: int,
    indices: Sequence[int],
) -> ndarray:
    """Filters the Jacobian according to the indices.

    Args:
        orig_jac: The original Jacobian.
        full_size: The size of the full flatten Jacobian array.
        indices: The array of indices to filter.

    Returns:
        The filtered Jacobian.
    """
    if indices is not None:
        jac = zeros((1, full_size))
        jac[:, indices] = orig_jac
        return jac

    return orig_jac


# TODO: API: rename to compute_sum_square_agg
def sum_square_agg(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Transform a vector of constraint functions into a sum squared function.

    The sum squared function is the sum of the squares of the input vector components.

    Args:
        orig_val: The input vector.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, scale all the constraint values.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The sum squared function value.
    """
    if indices is not None:
        orig_val = orig_val[indices]
    return np_sum(scale * orig_val**2)


# TODO: API: rename to compute_total_sum_square_agg_jac
def sum_square_agg_jac_v(
    orig_val: ndarray,
    orig_jac: ndarray,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Compute the Jacobian of squared sum function with respect to constraints inputs.

    Args:
        orig_val: The original constraint values.
        orig_jac: The original constraint jacobian.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The Jacobian of squared sum function with respect to constraints inputs.
    """
    orig_jac = atleast_2d(orig_jac)
    if indices is not None:
        orig_jac = orig_jac[indices, :]
        orig_val = orig_val[indices]

    return np_sum((2 * scale * orig_val).flatten() * orig_jac.T, axis=1)


# TODO: API: rename to compute_partial_sum_square_agg_jac
def sum_square_agg_jac(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Compute the Jacobian of squared sum function with respect to constraints.

    Args:
        orig_val: The original constraint values.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The Jacobian of squared sum function with respect to constraints.
    """
    if indices is not None:
        jac = zeros((1, orig_val.size))
        jac[:, indices] = 2.0 * scale * orig_val[indices]
    else:
        jac = full((1, orig_val.size), 2.0) * atleast_2d(scale * orig_val)

    return jac


# TODO: API: rename to compute_max_agg
def max_agg(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> float:
    """Transform a vector of constraints into a max of all values.

    The maximum function is not differentiable for all input values.

    Args:
        orig_val: The original constraint values.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The maximum value.
    """
    if indices is not None:
        orig_val = orig_val[indices]
    orig_val *= scale
    return array([np_max(orig_val)])


# TODO: API: rename to compute_max_agg_jac
def max_agg_jac_v(
    orig_val: ndarray,
    orig_jac: ndarray,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Compute the Jacobian of max function with respect to constraints inputs.

    Args:
        orig_val: The original constraint values.
        orig_jac: The original constraint jacobian.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The Jacobian of max function with respect to constraints inputs.
    """
    if indices is not None:
        orig_jac = orig_jac[indices, :]
        orig_val = orig_val[indices]
    orig_jac *= scale
    orig_val *= scale
    i_max = np_argmax(orig_val)

    return atleast_2d(orig_jac)[i_max, :]


def compute_sum_positive_square_agg(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Transform a vector of constraint functions into a positive sum squared function.

    The positive sum squared function is the sum of the squares of the input vector
        components that are positive.

    Args:
        orig_val: The original constraint values.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, scale all the constraint values.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The positive sum squared function value.
    """
    if indices is not None:
        orig_val = orig_val[indices]
    return np_sum(scale * (orig_val**2) * heaviside(orig_val, 0))


def compute_total_sum_square_positive_agg_jac(
    orig_val: ndarray,
    orig_jac: ndarray,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Compute the Jacobian of positive sum squared function w.r.t. constraints inputs.

    Args:
        orig_val: The original constraint values.
        orig_jac: The original constraint jacobian.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The Jacobian of positive sum squared function w.r.t. constraints inputs.
    """
    orig_jac = atleast_2d(orig_jac)
    if indices is not None:
        orig_jac = orig_jac[indices, :]
        orig_val = orig_val[indices]
    return np_sum(
        (2 * scale * orig_val * heaviside(orig_val, 0)).flatten() * orig_jac.T,
        axis=1,
    )


def compute_partial_sum_positive_square_agg_jac(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Compute the Jacobian of positive sum squared function w.r.t. constraints.

    Args:
        orig_val: The original constraint values.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The Jacobian of positive sum squared function w.r.t. constraints.
    """
    if indices is not None:
        jac = zeros((1, orig_val.size))
        jac[:, indices] = (
            2.0 * scale * orig_val[indices] * heaviside(orig_val[indices], 0.0)
        )
    else:
        jac = atleast_2d(2 * scale * orig_val * heaviside(orig_val, 0.0))

    return jac
