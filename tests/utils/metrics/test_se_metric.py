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
from numpy import full
from numpy import ndarray
from numpy.testing import assert_almost_equal

from gemseo.utils.metrics.squared_error_metric import SquaredErrorMetric


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (1, 3),
        (array([1, 2]), array([3, 4])),
        (array([[1, 1.5], [1.5, 2]]), array([[3, 3.5], [3.5, 4]])),
    ],
)
def test_se_metric(a, b):
    """Check the SE metric on floats and NumPy arrays."""
    c = SquaredErrorMetric().compute(a, b)
    if isinstance(a, ndarray):
        assert_almost_equal(c, full(a.shape, 4))
    else:
        assert c == pytest.approx(4)
