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

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from gemseo.algos.aggregation.core import compute_iks_agg
from gemseo.algos.aggregation.core import compute_lower_bound_ks_agg
from gemseo.algos.aggregation.core import compute_max_agg
from gemseo.algos.aggregation.core import compute_max_agg_jac
from gemseo.algos.aggregation.core import compute_sum_positive_square_agg
from gemseo.algos.aggregation.core import compute_sum_square_agg
from gemseo.algos.aggregation.core import compute_total_iks_agg_jac
from gemseo.algos.aggregation.core import compute_total_ks_agg_jac
from gemseo.algos.aggregation.core import compute_total_sum_square_agg_jac
from gemseo.algos.aggregation.core import compute_total_sum_square_positive_agg_jac
from gemseo.core.mdofunctions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import ndarray


def check_constraint_type(
    function_type: str,
) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    """Decorate a function to check whether it is of the expected type.

    Args:
        function_type: The expected function type, ``"ineq"`` or ``"eq"``.

    Returns:
        The decorated function.
    """

    def decorator(
        func: Callable[[Any], Any],
    ) -> Callable[[Any], Any]:
        """Decorator to check the aggregation function type.

        Args:
            func: The aggregation function.

        Returns:
            The decorated function.
        """

        @wraps(func)
        def function_wrapper(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
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
                    f"{func.__name__} constraint aggregation is only supported"
                    f" for func_type {function_type}, got {constr.f_type}"
                )
                raise ValueError(msg)

            return func(*args, **kwargs)

        return function_wrapper

    return decorator


@check_constraint_type("eq")
def aggregate_sum_square(
    constr_fct: MDOFunction,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> MDOFunction:
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
        return compute_sum_square_agg(constr_fct(x), indices=indices, scale=scale)

    def compute_jac(x):
        return compute_total_sum_square_agg_jac(
            constr_fct(x), constr_fct.jac(x), indices=indices, scale=scale
        )

    return _create_mdofunc(
        constr_fct,
        compute,
        compute_jac,
        f"sum²_{constr_fct.name}",
        f"sum({constr_fct.expr}**2)",
        "sum_sq_cstr",
    )


@check_constraint_type("ineq")
def aggregate_positive_sum_square(
    constr_fct: MDOFunction,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> MDOFunction:
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
        return compute_sum_positive_square_agg(
            constr_fct(x), indices=indices, scale=scale
        )

    def compute_jac(x):
        return compute_total_sum_square_positive_agg_jac(
            constr_fct(x), constr_fct.jac(x), indices=indices, scale=scale
        )

    return _create_mdofunc(
        constr_fct,
        compute,
        compute_jac,
        f"pos_sum_{constr_fct.name}",
        f"sum(heaviside({constr_fct.expr})*{constr_fct.expr}**2)",
        "pos_sum_sq_cstr",
    )


@check_constraint_type("ineq")
def aggregate_max(
    constr_fct: MDOFunction,
    indices: Sequence[int] | None = None,
    scale: float | ndarray = 1.0,
) -> MDOFunction:
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
        return compute_max_agg(constr_fct(x), indices=indices, scale=scale)

    def compute_jac(x):
        return compute_max_agg_jac(
            constr_fct(x), constr_fct.jac(x), indices=indices, scale=scale
        )

    return _create_mdofunc(
        constr_fct,
        compute,
        compute_jac,
        "max_" + constr_fct.name,
        f"max({constr_fct.expr})",
        "max_cstr",
    )


@check_constraint_type("ineq")
def aggregate_iks(
    constr_fct: MDOFunction,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> MDOFunction:
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
        return compute_iks_agg(constr_fct(x), indices=indices, rho=rho, scale=scale)

    def compute_jac(x):
        return compute_total_iks_agg_jac(
            constr_fct(x), constr_fct.jac(x), indices=indices, rho=rho, scale=scale
        )

    return _create_mdofunc(
        constr_fct,
        compute,
        compute_jac,
        f"IKS({constr_fct.name})",
        f"IKS({constr_fct.expr})",
        "IKS",
    )


@check_constraint_type("ineq")
def aggregate_lower_bound_ks(
    constr_fct: MDOFunction,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> MDOFunction:
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
        return compute_lower_bound_ks_agg(
            constr_fct(x), indices=indices, rho=rho, scale=scale
        )

    def compute_jac(x):
        return compute_total_ks_agg_jac(
            constr_fct(x), constr_fct.jac(x), indices=indices, rho=rho, scale=scale
        )

    return _create_mdofunc(
        constr_fct,
        compute,
        compute_jac,
        f"lower_bound_KS({constr_fct.name})",
        f"lower_bound_KS({constr_fct.expr})",
        "lower_bound_KS",
    )


@check_constraint_type("ineq")
def aggregate_upper_bound_ks(
    constr_fct: MDOFunction,
    indices: Sequence[int] | None = None,
    rho: float = 1e2,
    scale: float | ndarray = 1.0,
) -> MDOFunction:
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
        return compute_lower_bound_ks_agg(
            constr_fct(x), indices=indices, rho=rho, scale=scale
        )

    def compute_jac(x):
        return compute_total_ks_agg_jac(
            constr_fct(x), constr_fct.jac(x), indices=indices, rho=rho, scale=scale
        )

    return _create_mdofunc(
        constr_fct,
        compute,
        compute_jac,
        f"upper_bound_KS({constr_fct.name})",
        f"upper_bound_KS({constr_fct.expr})",
        "upper_bound_KS",
    )


def _create_mdofunc(
    constr_fct: MDOFunction,
    compute_fct: Callable[[ndarray], ndarray],
    compute_jac_fct: Callable[[ndarray], ndarray],
    new_name: str,
    new_expr: str,
    new_output_names: Sequence[str],
) -> MDOFunction:
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
        constr_fct.input_names,
        1,
        new_output_names,
        constr_fct.force_real,
        original_name=constr_fct.original_name,
    )
