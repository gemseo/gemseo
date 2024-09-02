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
from numpy import array
from numpy.testing import assert_allclose

from gemseo.problems.uncertainty.ishigami.functions import compute_gradient
from gemseo.problems.uncertainty.ishigami.functions import compute_output


@pytest.mark.parametrize(
    ("input_value", "output_value"),
    [
        (array([1.0, 1.0, 1.0]), 5.9),
        (array([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0]]), array([5.9, 7.1])),
    ],
)
def test_compute_output(input_value, output_value) -> None:
    """Check the output of the Ishigami function."""
    result = compute_output(input_value)
    assert_allclose(result, output_value, atol=1)
    assert isinstance(result, float) is (input_value.ndim == 1)


@pytest.mark.parametrize(
    ("input_value", "gradient_value"),
    [
        (array([1.0, 1.0, 1.0]), array([0.6, 6.4, 0.3])),
        (
            array([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0]]),
            array([[0.6, 6.4, 0.3], [1.4, 6.4, 2.7]]),
        ),
    ],
)
def test_compute_gradient(input_value, gradient_value) -> None:
    """Check the gradient of the Ishigami function."""
    assert_allclose(compute_gradient(input_value), gradient_value, atol=1)
