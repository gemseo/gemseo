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

from gemseo.utils.metrics.element_wise_metric import ElementWiseMetric
from gemseo.utils.metrics.mean_metric import MeanMetric
from gemseo.utils.metrics.squared_error_metric import SquaredErrorMetric


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([1, 2], [3, 4]),
        ([array([1, 2]), array([1, 2])], [array([3, 4]), array([3, 4])]),
    ],
)
def test_mse_metric(a, b):
    """Check the MSE metric on scalars and NumPy arrays."""
    mse = MeanMetric(ElementWiseMetric(SquaredErrorMetric()))
    assert mse.compute(a, b) == pytest.approx(4)
