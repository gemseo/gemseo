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

import pytest
from gemseo.algos.sequence_transformer.sequence_transformer_factory import (
    SequenceTransformerFactory,
)
from numpy import allclose
from numpy import arange
from numpy import ndarray
from numpy import ones
from numpy import sin
from numpy import vstack
from scipy.linalg import lstsq

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


def test_no_acceleration():
    """Tests the case where no acceleration is applied."""
    x_0 = INITIAL_VECTOR.copy()
    transformer = factory.create("NoAcceleration")

    x_1 = g(x_0)
    new_iterate = transformer.compute_transformed_iterate(x_0, x_1)
    assert new_iterate is x_1


def test_alternate_2_delta():
    """Tests the alternate 2-δ acceleration method."""
    x_0 = INITIAL_VECTOR.copy()
    transformer = factory.create("Alternate2Delta")

    x_1 = g(x_0)
    new_iterate = transformer.compute_transformed_iterate(x_0, x_1)
    assert allclose(new_iterate, x_1)

    x_2 = g(x_1)
    new_iterate = transformer.compute_transformed_iterate(x_1, x_2)
    assert allclose(new_iterate, x_2)

    x_3 = g(x_2)
    new_iterate = transformer.compute_transformed_iterate(x_2, x_3)

    dxn_2, dxn_1, dxn = x_1 - x_0, x_2 - x_1, x_3 - x_2
    gxn_2, gxn_1, gxn = x_1, x_2, x_3

    y, _, _, _ = lstsq(vstack([dxn - dxn_1, dxn_1 - dxn_2]).T, dxn)
    new_iterate_ref = gxn - y[0] * (gxn - gxn_1) - y[1] * (gxn_1 - gxn_2)

    assert allclose(new_iterate, new_iterate_ref)


def test_alternate_delta_squared():
    """Tests the alternate δ² acceleration method."""
    x_0 = INITIAL_VECTOR.copy()
    transformer = factory.create("AlternateDeltaSquared")

    x_1 = g(x_0)
    new_iterate = transformer.compute_transformed_iterate(x_0, x_1)
    assert allclose(new_iterate, x_1)

    x_2 = g(x_1)
    new_iterate = transformer.compute_transformed_iterate(x_1, x_2)
    assert allclose(new_iterate, x_2)

    x_3 = g(x_2)
    new_iterate = transformer.compute_transformed_iterate(x_2, x_3)

    dxn_2, dxn_1, dxn = x_1 - x_0, x_2 - x_1, x_3 - x_2
    gxn_2, gxn_1, gxn = x_1, x_2, x_3

    dz = dxn - 2 * dxn_1 + dxn_2
    gz = gxn - 2 * gxn_1 + gxn_2
    new_iterate_ref = gxn - (dz.T @ dxn) / (dz.T @ dz) * gz

    assert allclose(new_iterate, new_iterate_ref)


def test_secant():
    """Tests the secant method."""
    x_0 = INITIAL_VECTOR.copy()
    transformer = factory.create("Secant")

    x_1 = g(x_0)
    new_iterate = transformer.compute_transformed_iterate(x_0, x_1)
    assert allclose(new_iterate, x_1)

    x_2 = g(x_1)
    new_iterate = transformer.compute_transformed_iterate(x_1, x_2)

    dxn_1, dxn = x_1 - x_0, x_2 - x_1
    gxn_1, gxn = x_1, x_2

    d2xn = dxn - dxn_1
    dgxn = gxn - gxn_1
    new_iterate_ref = gxn - (d2xn.T @ dxn) / (d2xn.T @ d2xn) * dgxn

    assert allclose(new_iterate, new_iterate_ref)


def test_aitken():
    """Tests the Aitken method."""
    x_0 = INITIAL_VECTOR.copy()
    transformer = factory.create("Secant")

    x_1 = g(x_0)
    new_iterate = transformer.compute_transformed_iterate(x_0, x_1)
    assert allclose(new_iterate, x_1)

    x_2 = g(x_1)
    new_iterate = transformer.compute_transformed_iterate(x_1, x_2)

    dxn_1, dxn = x_1 - x_0, x_2 - x_1
    gxn = x_2

    d2xn = dxn - dxn_1
    new_iterate_ref = gxn - (d2xn.T @ dxn) / (d2xn.T @ d2xn) * dxn

    assert allclose(new_iterate, new_iterate_ref)


@pytest.mark.parametrize(
    "window_size",
    [0, 2, "foo"],
)
def test_minimum_polynomial_parameters(window_size):
    """Tests the window size argument of MinimumPolynomial."""
    if window_size in [0, "foo"]:
        with pytest.raises(ValueError):
            factory.create("MinimumPolynomial", window_size=window_size)
    else:
        factory.create("MinimumPolynomial", window_size=window_size)
