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
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import arange
from numpy import ndarray
from numpy import ones
from numpy import sin
from numpy.linalg import norm

from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.algos.sequence_transformer.sequence_transformer_factory import (
    SequenceTransformerFactory,
)

if TYPE_CHECKING:
    from gemseo.algos.sequence_transformer.sequence_transformer import (
        SequenceTransformer,
    )

A_TOL: float = 1e-6
DIMENSION: int = 100
MEAN_ANOMALY = arange(DIMENSION) + 1
EXCENTRICITY = 0.95
INITIAL_VECTOR = ones(DIMENSION)

factory = SequenceTransformerFactory()


def g(x: ndarray) -> ndarray:
    """The function for fixed-point iteration method.

    Args:
        x: The input vector.

    Returns:
        The image G(x).
    """
    return MEAN_ANOMALY + EXCENTRICITY * sin(x)


def fixed_point_method(
    x_0: ndarray, tol: float = A_TOL, transformer: SequenceTransformer | None = None
):
    """The fixed-point iteration method associated with the function g above.

    Args:
        x_0: The initial vector.
        tol: The tolerance to consider convergence of the method.
        transformer: The sequence transformer to apply.

    Returns:
        A vector x* satisfying ||g(x*) - x*|| < tol.
    """
    x_n = x_0.copy()

    k = 0

    is_converged = False
    while not is_converged and k < 100:
        gxn = g(x_n)

        # Apply sequence transformer if required.
        if transformer is not None:
            x_n1 = transformer.compute_transformed_iterate(x_n, gxn)
        else:
            x_n1 = gxn

        # Test convergence
        is_converged = norm(x_n1 - x_n) < tol

        x_n = x_n1.copy()
        k += 1

    return k


@pytest.fixture(scope="module")
def compute_reference_n_iter():
    """Compute the number of fixed-point iteration without sequence transformer."""
    return fixed_point_method(INITIAL_VECTOR, tol=A_TOL)


@pytest.mark.parametrize(
    "transformer",
    [
        "Alternate2Delta",
        "AlternateDeltaSquared",
        "Secant",
        "Aitken",
        "MinimumPolynomial",
        "OverRelaxation",
    ],
)
def test_sequence_transformer(compute_reference_n_iter, transformer):
    """Tests that the sequence transform reduces the number of iterations."""
    _transformer = factory.create(transformer)

    n_iter_ref = compute_reference_n_iter
    n_iter = fixed_point_method(INITIAL_VECTOR, tol=A_TOL, transformer=_transformer)

    assert n_iter <= n_iter_ref


def test_clear():
    """Tests the clear mechanism of the double-ended queue."""
    transformer = SequenceTransformerFactory().create(
        AccelerationMethod.ALTERNATE_2_DELTA
    )

    assert len(transformer._iterates) == 0
    assert len(transformer._residuals) == 0

    transformer.compute_transformed_iterate(ones(1), ones(1))

    assert len(transformer._iterates) == 1
    assert len(transformer._residuals) == 1

    transformer.clear()

    assert len(transformer._iterates) == 0
    assert len(transformer._residuals) == 0


def test_composite(compute_reference_n_iter):
    """Tests that the Composite structure reduces the number of iterations."""
    aitken = factory.create(AccelerationMethod.AITKEN)
    relaxation = factory.create("OverRelaxation", factor=0.8)

    _transformer = factory.create(
        "CompositeSequenceTransformer", sequence_transformers=[aitken, relaxation]
    )

    n_iter_ref = compute_reference_n_iter
    n_iter = fixed_point_method(INITIAL_VECTOR, tol=A_TOL, transformer=_transformer)

    assert n_iter <= n_iter_ref

    _transformer.clear()
