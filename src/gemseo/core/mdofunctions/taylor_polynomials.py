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
"""Functions computing first- and second-order Taylor polynomials from a function."""
from __future__ import annotations

from typing import Sequence

from numpy import matmul
from numpy import ndarray

from gemseo.core.mdofunctions.mdo_function import ArrayType
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from gemseo.core.mdofunctions.mdo_quadratic_function import MDOQuadraticFunction


def compute_linear_approximation(
    function: MDOFunction,
    x_vect: ArrayType,
    name: str | None = None,
    f_type: str | None = None,
    args: Sequence[str] | None = None,
) -> MDOLinearFunction:
    r"""Compute a first-order Taylor polynomial of a function.

    :math:`\newcommand{\xref}{\hat{x}}\newcommand{\dim}{d}`
    The first-order Taylor polynomial of a (possibly vector-valued) function
    :math:`f` at a point :math:`\xref` is defined as

    .. math::
        \newcommand{\partialder}{\frac{\partial f}{\partial x_i}(\xref)}
        f(x)
        \approx
        f(\xref) + \sum_{i = 1}^{\dim} \partialder \, (x_i - \xref_i).

    Args:
        function: The function to be linearized.
        x_vect: The input vector at which to build the Taylor polynomial.
        name: The name of the linear approximation function.
            If ``None``, create a name from the name of the function.
        f_type: The type of the linear approximation function.
            If ``None``, the function will have no type.
        args: The names of the inputs of the linear approximation function,
            or a name base.
            If ``None``, use the names of the inputs of the function.

    Returns:
        The first-order Taylor polynomial of the function at the input vector.

    Raises:
        AttributeError: If the function does not have a Jacobian function.
    """
    if not function.has_jac():
        raise AttributeError("Function Jacobian unavailable for linear approximation.")

    coefficients = function.jac(x_vect)
    func_val = function.evaluate(x_vect)
    if isinstance(func_val, ndarray):
        # Make sure the function value is at most 1-dimensional
        func_val = func_val.flatten()

    return MDOLinearFunction(
        coefficients,
        f"{function.name}_linearized" if name is None else name,
        f_type,
        args if args else function.args,
        func_val - matmul(coefficients, x_vect),
    )


def compute_quadratic_approximation(
    function: MDOFunction,
    x_vect: ArrayType,
    hessian_approx: ArrayType,
    args: Sequence[str] | None = None,
) -> MDOQuadraticFunction:
    r"""Build a quadratic approximation of a function at a given point.

    The function must be scalar-valued.

    :math:`\newcommand{\xref}{\hat{x}}\newcommand{\dim}{d}\newcommand{
    \hessapprox}{\hat{H}}`
    For a given approximation :math:`\hessapprox` of the Hessian matrix of a
    function :math:`f` at a point :math:`\xref`, the quadratic approximation of
    :math:`f` is defined as

    .. math::
        \newcommand{\partialder}{\frac{\partial f}{\partial x_i}(\xref)}
        f(x)
        \approx
        f(\xref)
        + \sum_{i = 1}^{\dim} \partialder \, (x_i - \xref_i)
        + \frac{1}{2} \sum_{i = 1}^{\dim} \sum_{j = 1}^{\dim}
        \hessapprox_{ij} (x_i - \xref_i) (x_j - \xref_j).

    Args:
        function: The function to be approximated.
        x_vect: The input vector at which to build the quadratic approximation.
        hessian_approx: The approximation of the Hessian matrix
            at this input vector.
        args: The names of the inputs of the quadratic approximation function,
            or a name base.
            If ``None``, use the ones of the current function.

    Returns:
        The second-order Taylor polynomial of the function at the given point.
    """
    if (
        not isinstance(hessian_approx, ndarray)
        or hessian_approx.ndim != 2
        or hessian_approx.shape[0] != hessian_approx.shape[1]
    ):
        raise ValueError("Hessian approximation must be a square ndarray.")

    if hessian_approx.shape[1] != x_vect.size:
        raise ValueError("Hessian approximation and vector must have same dimension.")

    if not function.has_jac():
        raise AttributeError("Jacobian unavailable.")

    gradient = function.jac(x_vect)
    hess_dot_vect = matmul(hessian_approx, x_vect)

    return MDOQuadraticFunction(
        quad_coeffs=0.5 * hessian_approx,
        linear_coeffs=gradient - hess_dot_vect,
        value_at_zero=(
            matmul(0.5 * hess_dot_vect - gradient, x_vect) + function.evaluate(x_vect)
        ),
        name=f"{function.name}_quadratized",
        args=args if args else function.args,
    )
