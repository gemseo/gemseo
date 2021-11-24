# -*- coding: utf-8 -*-
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

"""Constraint aggregation methods.

Transform a constraint vector into one scalar equivalent or quasi equivalent constraint.
"""

from __future__ import division, unicode_literals

from functools import wraps
from typing import Any, Callable, Optional, Sequence, Union

from numpy import ndarray

from gemseo.algos.aggregation.core import (
    iks_agg,
    iks_agg_jac_v,
    ks_agg,
    ks_agg_jac_v,
    max_agg,
    max_agg_jac_v,
    sum_square_agg,
    sum_square_agg_jac_v,
)
from gemseo.core.mdofunctions.mdo_function import MDOFunction


def check_constraint_type(
    function_type,  # type: str
):  # type: (...) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]
    """Decorate a function to check whether it is of the expected type.

    Args:
        function_type: The expected function type, ``"ineq"`` or ``"eq"``.

    Returns:
        The decorated function.
    """

    def decorator(
        func,  # type: Callable[[Any], Any]
    ):  # type: (...) -> Callable[[Any], Any]
        """Decorator to check the aggregation function type.

        Args:
            func: The aggregation function.

        Returns:
            The decorated function.
        """

        @wraps(func)
        def function_wrapper(
            *args,  # type: Any
            **kwargs  # type: Any
        ):  # type: (...) -> Any
            """Check that ``func`` has the type `function_type``.

            Args:
                *args: The positional arguments.
                **kwargs: The keyword arguments.

            Raises:
                ValueError: If the type is not correct.

            Returns:
                The return value of ``func``.
            """
            constr = args[0]

            if constr.f_type != function_type:
                msg = (
                    "{} constraint aggregation is only supported"
                    " for func_type {}, got {}"
                ).format(func.__name__, function_type, constr.f_type)
                raise ValueError(msg)

            return func(*args, **kwargs)

        return function_wrapper

    return decorator


@check_constraint_type("eq")
def aggregate_sum_square(
    constr_fct,  # type: MDOFunction
    indices=None,  # type: Optional[Sequence[int]]
    scale=1.0,  # type: Union[float, ndarray]
):  # type: (...) -> MDOFunction
    """Transform a vector of equalities into a sum of squared constraints.

    Args:
        constr_fct: The initial constraint function.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The aggregated function.
    """

    def compute(x):
        return sum_square_agg(constr_fct(x), indices=indices, scale=scale)

    def compute_jac(x):
        return sum_square_agg_jac_v(
            constr_fct(x), constr_fct.jac(x), indices=indices, scale=scale
        )

    return _create_mdofunc(
        constr_fct,
        compute,
        compute_jac,
        "sum²_{}".format(constr_fct.name),
        "sum({}**2)".format(constr_fct.expr),
        "sum_sq_cstr",
    )


@check_constraint_type("ineq")
def aggregate_max(
    constr_fct,  # type: MDOFunction
    indices=None,  # type: Optional[Sequence[int]]
    scale=1.0,  # type: Union[float, ndarray]
):  # type: (...) -> MDOFunction
    """Transform a vector of equalities into a max of all values.

    Args:
        constr_fct: The initial constraint function.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        scale: The scaling factor for multiplying the constraints.

    Returns:
        The aggregated function.
    """

    def compute(x):
        return max_agg(constr_fct(x), indices=indices, scale=scale)

    def compute_jac(x):
        return max_agg_jac_v(
            constr_fct(x), constr_fct.jac(x), indices=indices, scale=scale
        )

    return _create_mdofunc(
        constr_fct,
        compute,
        compute_jac,
        "max_" + constr_fct.name,
        "max({})".format(constr_fct.expr),
        "max_cstr",
    )


@check_constraint_type("ineq")
def aggregate_iks(
    constr_fct,  # type: MDOFunction
    indices=None,  # type: Optional[Sequence[int]]
    rho=1e2,  # type: float
    scale=1.0,  # type: Union[float, ndarray]
):  # type: (...) -> MDOFunction
    """Constraints aggregation method for inequality constraints.

    See :cite:`kennedy2015improved`.

    Args:
        constr_fct: The initial constraint function.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        scale: The scaling factor for multiplying the constraints.
        rho: The multiplicative parameter in the exponential.

    Returns:
        The aggregated function.
    """

    def compute(x):
        return iks_agg(constr_fct(x), indices=indices, rho=rho, scale=scale)

    def compute_jac(x):
        return iks_agg_jac_v(
            constr_fct(x), constr_fct.jac(x), indices=indices, rho=rho, scale=scale
        )

    return _create_mdofunc(
        constr_fct,
        compute,
        compute_jac,
        "IKS({})".format(constr_fct.name),
        "IKS({})".format(constr_fct.expr),
        "IKS",
    )


@check_constraint_type("ineq")
def aggregate_ks(
    constr_fct,  # type: MDOFunction
    indices=None,  # type: Optional[Sequence[int]]
    rho=1e2,  # type: float
    scale=1.0,  # type: Union[float, ndarray]
):  # type: (...) -> MDOFunction
    """Aggregate constraints for inequality constraints.

    See :cite:`kennedy2015improved` and  :cite:`kreisselmeier1983application`.

    Args:
        constr_fct: The initial constraint function.
        indices: The indices to generate a subset of the outputs to aggregate.
            If ``None``, aggregate all the outputs.
        scale: The scaling factor for multiplying the constraints.
        rho: The multiplicative parameter in the exponential.

    Returns:
        The aggregated function.
    """

    def compute(x):
        return ks_agg(constr_fct(x), indices=indices, rho=rho, scale=scale)

    def compute_jac(x):
        return ks_agg_jac_v(
            constr_fct(x), constr_fct.jac(x), indices=indices, rho=rho, scale=scale
        )

    return _create_mdofunc(
        constr_fct,
        compute,
        compute_jac,
        "KS({})".format(constr_fct.name),
        "KS({})".format(constr_fct.expr),
        "KS",
    )


def _create_mdofunc(
    constr_fct,  # type: MDOFunction
    compute_fct,  # type: Callable[[ndarray], ndarray]
    compute_jac_fct,  # type: Callable[[ndarray], ndarray]
    new_name,  # type: str
    new_expr,  # type: str
    new_output_names,  # type: Sequence[str]
):  # type: (...) -> MDOFunction
    """Create an aggregated MDOFunction from a constraint function.

    Args:
        constr_fct: The initial constraint function.
        compute_fct: The aggregated compute function.
        compute_jac_fct: The aggregated compute function jacobian.
        new_name: The name of aggregated function.
        new_expr: The aggregated function expression.
        new_output_names: The aggregated function output names.

    Returns:
        The aggregated MDOFunction.
    """
    return MDOFunction(
        compute_fct,
        new_name,
        constr_fct.f_type,
        compute_jac_fct,
        new_expr,
        constr_fct.args,
        1,
        new_output_names,
        constr_fct.force_real,
    )
