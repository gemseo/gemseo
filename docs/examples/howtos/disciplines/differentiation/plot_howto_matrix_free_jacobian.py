# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
r"""# Use a matrix-free Jacobian

## Problem

Matrix-free Jacobians are typically used when the matrix representation of the
Jacobian is not affordable, in terms of memory or computation, and only the
product of the Jacobian (and of its transpose) with a vector is available.
This is typical of adjoint-based solvers, convolution operators or low-rank
update formulas.

## Solution

A discipline usually stores a dense or sparse matrix in its `jac` dictionary.
For a matrix-free Jacobian, it can instead store a
[JacobianOperator][gemseo.core.derivative.jacobian_operator.JacobianOperator],
a SciPy [LinearOperator][scipy.sparse.linalg.LinearOperator] that, rather than
holding the matrix, defines how it acts on a vector through two products:
`_matvec` computing $J_f(x)\,w$ and `_rmatvec` computing $J_f(x)^{\top} w$.

## Step-by-step guide
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import eye
from numpy import ones
from numpy import outer

from gemseo.core.derivative.jacobian_operator import JacobianOperator
from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray
    from gemseo.util.typing import StrKeyMapping

# %%
# ### 1. Create a matrix-free Jacobian operator
#
# We consider a discipline computing $y = f(x) = x + u(v^{\top}x)$, whose Jacobian is
# the low-rank update of the identity $J_f(x) = I + uv^{\top}$.
# Forming $J_f(x)$ as a matrix costs $O(n^2)$ memory,
# while applying it to a vector only requires $O(n)$ operations.
# We therefore implement it as a
# [JacobianOperator][gemseo.core.derivative.jacobian_operator.JacobianOperator]
# by subclassing it and implementing the `_matvec` and `_rmatvec` methods.
#
# !!! note
#     Following the SciPy [LinearOperator][scipy.sparse.linalg.LinearOperator] contract,
#     `_matvec` and `_rmatvec` can receive the vector either with shape
#     $(n,)$ or as a column matrix with shape $(n, 1)$, hence the call to `ravel`.


class LowRankJacobianOperator(JacobianOperator):
    r"""The Jacobian operator $J_f(x) = I + uv^{\top}$, applied without forming it."""

    def __init__(self, u: RealArray, v: RealArray) -> None:
        """
        Args:
            u: The left vector of the rank-one update.
            v: The right vector of the rank-one update.
        """  # noqa: D205 D212 D415
        super().__init__(dtype=u.dtype, shape=(u.size, v.size))
        self.__u = u
        self.__v = v

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the operator to.

        Returns:
            The product of the operator with the vector.
        """  # noqa: D205 D212
        x = x.ravel()
        return x + self.__u * (self.__v @ x)

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the adjoint of the operator to.

        Returns:
            The product of the adjoint of the operator with the vector.
        """  # noqa: D205 D212
        x = x.ravel()
        return x + self.__v * (self.__u @ x)


# %%
# ### 2. Create the discipline
#
# The discipline stores this operator in its `jac` dictionary,
# exactly where a dense or sparse Jacobian matrix would go:


class LowRankDiscipline(Discipline):
    r"""A discipline computing $y = x + u(v^{\top}x)$ with a matrix-free Jacobian."""

    def __init__(self, u: RealArray, v: RealArray) -> None:
        """
        Args:
            u: The left vector of the rank-one update.
            v: The right vector of the rank-one update.
        """  # noqa: D205 D212 D415
        super().__init__()
        self.input_grammar.update_from_names(["x"])
        self.output_grammar.update_from_names(["y"])
        self.default_input_data = {"x": ones(v.size)}
        self.__u = u
        self.__v = v

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x = input_data["x"]
        return {"y": x + self.__u * (self.__v @ x)}

    def _compute_jacobian(self, input_names=(), output_names=()) -> None:
        self.jac = {"y": {"x": LowRankJacobianOperator(self.__u, self.__v)}}


# %%
# ### 3. Linearize the discipline
#
# The [linearize()][gemseo.core.discipline.discipline.Discipline.linearize]
# method returns the Jacobian operator:
n = 5
u = array([0.1, 0.2, 0.3, 0.4, 0.5])
v = array([1.0, 1.25, 1.5, 1.75, 2.0])
discipline = LowRankDiscipline(u, v)
jacobian = discipline.linearize(compute_all_jacobians=True)["y"]["x"]  # ∂y/∂x
jacobian

# %%
# !!! note
#     The returned operator is not exactly the `LowRankJacobianOperator` we
#     stored: [linearize()][gemseo.core.discipline.discipline.Discipline.linearize]
#     discards the imaginary part of every Jacobian, hence it wraps our operator
#     in a real-casting
#     [JacobianOperator][gemseo.core.derivative.jacobian_operator.JacobianOperator].
#     The two behave identically for real inputs.
#
# The operator applies the Jacobian to a vector without ever forming it:
jacobian.dot(ones(n))

# %%
# and similarly for its transpose:
jacobian.T.dot(ones(n))

# %%
# ## Summary
#
# Subclass
# [JacobianOperator][gemseo.core.derivative.jacobian_operator.JacobianOperator]
# and implement `_matvec` (the product $J_f(x)\,w$) and `_rmatvec` (the product
# $J_f(x)^{\top} w$). Store an instance in the `jac` dictionary of the discipline in
# `_compute_jacobian()`, and everything stays matrix-free.
#
# ## One step further
#
# ### Materialize the operator (for debugging)
#
# The
# [get_matrix_representation()][gemseo.core.derivative.jacobian_operator.JacobianOperator.get_matrix_representation]
# method assembles the matrix by applying the operator to the identity.
# This is convenient to inspect or check the Jacobian,
# but defeats the purpose of the matrix-free setting:
# only use it on small problems.
jacobian.get_matrix_representation()

# %%
# We can read off that this matches the expected Jacobian $J_f(x) = I + uv^{\top}$:
eye(n) + outer(u, v)

# %%
# ### Linear algebra operations
#
# A [JacobianOperator][gemseo.core.derivative.jacobian_operator.JacobianOperator]
# can be combined with other Jacobian operators, NumPy arrays and SciPy sparse
# arrays using `+`, `-` and `@` in either operand order. Each operation returns
# a new operator whose evaluation remains lazy: no matrix is ever assembled.
# For instance, adding a NumPy array to the operator:
shifted = eye(n) + jacobian
shifted.dot(ones(n))
