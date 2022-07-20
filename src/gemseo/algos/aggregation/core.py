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
from numpy import max as np_max
from numpy import multiply
from numpy import ndarray
from numpy import sum as np_sum
from numpy import zeros


def ks_agg(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> float:
    """Transform a vector of equalities into a Kreisselmeier–Steinhauser function.

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
        rho: The multiplicative parameter in the exponential.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The aggregated function value.
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


def ks_agg_jac_v(
    orig_val: ndarray,
    orig_jac: ndarray,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Aggregate inequality constraints.

    Jacobian vector product of the constraints aggregation method for inequality
    constraints.

    See :cite:`kennedy2015improved` and  :cite:`kreisselmeier1983application`.

    Args:
        orig_val: The original constraint values.
        orig_jac: The original constraint jacobian.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        rho: The multiplicative parameter in the exponential.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The aggregated function Jacobian vector product.
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


def ks_agg_jac(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Transform a vector of equalities into a scalar equivalent constraint.

    Jacobian of the Constraints aggregation method for inequality constraints.

    See :cite:`kennedy2015improved` and  :cite:`kreisselmeier1983application`.

    Args:
        orig_val: The original constraint values.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        rho: The multiplicative parameter in the exponential.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The aggregated function Jacobian.
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


def iks_agg(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> float:
    """Aggregate IKS Constraints for inequality constraints.

    See :cite:`kennedy2015improved`.

    Args:
        orig_val: The original constraint values.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        rho: The multiplicative parameter in the exponential.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The aggregated value.
    """
    if indices is not None:
        orig_val = orig_val[indices]

    orig_val *= scale

    m = max(orig_val)
    iks = sum(orig_val * np_exp(rho * (orig_val + 1.0 - m)))
    iks /= sum(np_exp(rho * (orig_val + 1.0 - m)))

    return iks


def iks_agg_jac_v(
    orig_val: ndarray,
    orig_jac: ndarray,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Aggregate inequality constraints.

    Jacobian vector product of the IKS Constraints aggregation method for inequality
    constraints.

    See :cite:`kennedy2015improved`.

    Args:
        orig_val: The original constraint values.
        orig_jac: The original constraint jacobian.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        rho: The multiplicative parameter in the exponential.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The aggregated function Jacobian vector product.
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


def iks_agg_jac(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Jacobian of the IKS Constraints aggregation method for inequality constraints.

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
        The aggregated function Jacobian.
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


def sum_square_agg(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Transform a vector of equalities into a sum of squared constraints.

    Args:
        orig_val: The original constraint values.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, scale all the constraint values.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The aggregated function.
    """
    if indices is not None:
        orig_val = orig_val[indices]
    orig_val *= scale
    return np_sum(orig_val**2)


def sum_square_agg_jac_v(
    orig_val: ndarray,
    orig_jac: ndarray,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Transform a vector of equalities into a sum of squared constraints.

    Jacobian vector product of the constraints aggregation method for equality
    constraints.

    Args:
        orig_val: The original constraint values.
        orig_jac: The original constraint jacobian.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The aggregated function Jacobian vector product.
    """
    if indices is not None:
        orig_jac = orig_jac[indices, :]
        orig_val = orig_val[indices]
    orig_jac *= scale
    orig_val *= scale
    return 2 * np_sum(orig_jac * orig_val, axis=0)


def sum_square_agg_jac(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Transform a vector of equalities into a sum of squared constraints.

    Jacobian of the constraints aggregation method for equality constraints.

    Args:
        orig_val: The original constraint values.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The aggregated function Jacobian.
    """
    if indices is not None:
        jac = zeros((1, orig_val.size))
        jac[:, indices] = 2.0
    else:
        jac = full((1, orig_val.size), 2.0)

    return jac


def max_agg(
    orig_val: ndarray,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> float:
    """Transform a vector of equalities into a max of all values.

    Constraints aggregation method for inequality constraints.

    Args:
        orig_val: The original constraint values.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The aggregated function.
    """
    if indices is not None:
        orig_val = orig_val[indices]
    orig_val *= scale
    return array([np_max(orig_val)])


def max_agg_jac_v(
    orig_val: ndarray,
    orig_jac: ndarray,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> ndarray:
    """Transform a vector of equalities into the max of all the values.

    Jacobian vector product of the max constraints aggregation method for inequality
    constraints.

    Args:
        orig_val: The original constraint values.
        orig_jac: The original constraint jacobian.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The aggregated function.
    """
    if indices is not None:
        orig_jac = orig_jac[indices, :]
        orig_val = orig_val[indices]
    orig_jac *= scale
    orig_val *= scale
    i_max = np_argmax(orig_val)
    return atleast_2d(orig_jac)[i_max, :]
