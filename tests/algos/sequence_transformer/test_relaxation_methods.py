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
from numpy import allclose
from numpy import arange
from numpy import ndarray
from numpy import ones
from numpy import sin

from gemseo.algos.sequence_transformer.sequence_transformer_factory import (
    SequenceTransformerFactory,
)

A_TOL: float = 1e-6
DIMENSION: int = 100
MEAN_ANOMALY = arange(DIMENSION) + 1
EXCENTRICITY = 0.95
INITIAL_VECTOR = ones(DIMENSION)


def g(x: ndarray) -> ndarray:
    """The function for fixed-point iteration method.

    Args:
        x: The input vector.

    Returns:
        The image G(x).
    """
    return MEAN_ANOMALY + EXCENTRICITY * sin(x)


@pytest.mark.parametrize(
    "factor",
    [0.5, 1.0, 1.5],
)
def test_overrelaxation(factor):
    """Tests the over relaxation method."""
    x_0 = INITIAL_VECTOR.copy()
    transformer = SequenceTransformerFactory().create("OverRelaxation", factor=factor)

    x_1 = g(x_0)
    new_iterate = transformer.compute_transformed_iterate(x_1, x_1 - x_0)
    assert allclose(new_iterate, x_1)

    x_2 = g(x_1)
    new_iterate = transformer.compute_transformed_iterate(x_2, x_2 - x_1)

    gxn_1, gxn = x_1, x_2
    new_iterate_ref = factor * gxn + (1.0 - factor) * gxn_1

    assert allclose(new_iterate, new_iterate_ref)


@pytest.mark.parametrize(
    "factor",
    [-1, 1, 3, "foo"],
)
def test_relaxation_factor(factor):
    """Tests the relaxation factor argument of OverRelaxation."""
    if factor in [-1, 3]:
        with pytest.raises(ValueError):
            SequenceTransformerFactory().create("OverRelaxation", factor=factor)
    elif factor == "foo":
        with pytest.raises(TypeError):
            SequenceTransformerFactory().create("OverRelaxation", factor=factor)
    else:
        SequenceTransformerFactory().create("OverRelaxation", factor=factor)
