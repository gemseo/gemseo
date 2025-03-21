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
"""Test the comparisons module."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import nan
from scipy.sparse import csr_array

from gemseo.utils.comparisons import compare_dict_of_arrays


@pytest.mark.parametrize("tolerance", [0.0, 0.1])
@pytest.mark.parametrize(
    "other_dict",
    [
        {"x": array([1.0])},
        {"x": array([[1.0, 1.0]])},
        {"x": array([[1.0], [1.0]])},
    ],
)
def test_different_sizes(tolerance, other_dict):
    """Test comparison regarding the array sizes."""
    assert not compare_dict_of_arrays({"x": array([1.0, 1.0])}, other_dict, tolerance)


@pytest.mark.parametrize("tolerance", [0.0, 0.1])
@pytest.mark.parametrize(
    "other_dict",
    [
        {"y": array([1.0])},
        {"x": array([1.0]), "y": array([1.0])},
        {"x": {"x": array([1.0])}},
    ],
)
def test_keys(tolerance, other_dict):
    """Test comparison regarding the dictionnaries' keys."""
    assert not compare_dict_of_arrays({"x": array([1.0])}, other_dict, tolerance)


@pytest.mark.parametrize("array_1_dense", [False, True])
@pytest.mark.parametrize("array_2_dense", [False, True])
@pytest.mark.parametrize(
    ("data_1", "data_2", "equal_nan", "are_equal"),
    [
        ([[1.0, 1.0]], [[1.0, 1.0]], False, True),
        ([[1.0, 1.0]], [[1.0, 1.0]], True, True),
        ([[1.0, nan]], [[1.0, nan]], False, False),
        ([[1.0, nan]], [[1.0, nan]], True, True),
        ([[2.0, 1.0]], [[1.0, 1.0]], False, False),
        ([[2.0, 1.0]], [[1.0, 1.0]], True, False),
        ([[2.0, nan]], [[1.0, nan]], False, False),
        ([[2.0, nan]], [[1.0, nan]], True, False),
    ],
)
def test_array_are_equal(
    array_1_dense, array_2_dense, data_1, data_2, equal_nan, are_equal
):
    """Test equality comparison between arrays."""
    array_1 = array(data_1) if array_1_dense else csr_array(data_1)
    dict_1 = {"x": array_1}

    array_2 = array(data_2) if array_2_dense else csr_array(data_2)
    dict_2 = {"x": array_2}

    assert compare_dict_of_arrays(dict_1, dict_2, nan_are_equal=equal_nan) is are_equal


@pytest.mark.parametrize("array_1_dense", [False, True])
@pytest.mark.parametrize("array_2_dense", [False, True])
@pytest.mark.parametrize(
    ("data_1", "data_2", "equal_nan", "are_equal"),
    [
        ([[1.5, 0.0]], [[1.0, 0.0]], False, True),
        ([[1.5, 0.0]], [[1.0, 0.0]], True, True),
        ([[1.5, nan]], [[1.0, nan]], False, False),
        ([[1.5, nan]], [[1.0, nan]], True, True),
        ([[10.0, 0.0]], [[1.0, 0.0]], False, False),
        ([[10.0, 0.0]], [[1.0, 0.0]], True, False),
        ([[10.0, nan]], [[1.0, nan]], False, False),
        ([[10.0, nan]], [[1.0, nan]], True, False),
    ],
)
def test_array_are_close(
    array_1_dense, array_2_dense, data_1, data_2, equal_nan, are_equal
):
    """Test closeness comparison between arrays."""
    array_1 = array(data_1) if array_1_dense else csr_array(data_1)
    dict_1 = {"x": array_1}

    array_2 = array(data_2) if array_2_dense else csr_array(data_2)
    dict_2 = {"x": array_2}

    assert (
        compare_dict_of_arrays(dict_1, dict_2, nan_are_equal=equal_nan, tolerance=1.0)
        is are_equal
    )
