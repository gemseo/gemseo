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
# Matthias De Lozzo
# Antoine DECHAUME
"""Data comparison tools."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Union

from numpy import allclose
from numpy import array_equal
from numpy import asarray
from numpy import ndarray

from gemseo.typing import SparseOrDenseRealArray
from gemseo.utils.compatibility.scipy import sparse_classes
from gemseo.utils.data_conversion import flatten_nested_dict

DataToCompare = Union[
    Mapping[str, SparseOrDenseRealArray],
    Mapping[str, Mapping[str, SparseOrDenseRealArray]],
]


# TODO: add runtime optimization to detect,
# until some point in gemseo process (first algo iteration?),
# if sparse arrays are actually used and
# then allow this function to use a specialized implementation.
def compare_dict_of_arrays(
    dict_of_arrays: DataToCompare,
    other_dict_of_arrays: DataToCompare,
    tolerance: float = 0.0,
    nan_are_equal: bool = False,
) -> bool:
    """Check if two dictionaries of NumPy and/or SciPy sparse arrays are equal.

    The dictionaries can be nested, in which case they are flattened. If the tolerance
    is set, then arrays are considered equal if ``norm(dict_of_arrays[name] -
    other_dict_of_arrays[name]) /(1 + norm(other_dict_of_arrays[name])) <= tolerance``.

    Args:
        dict_of_arrays: A dictionary of NumPy arrays and/or SciPy sparse matrices.
        other_dict_of_arrays: Another dictionary of NumPy arrays and/or SciPy sparse
            matrices.
        tolerance: The relative tolerance. If 0.0, the array must be exactly equal.
        nan_are_equal: Whether to compare NaN's as equal.

    Returns:
        Whether the dictionaries are equal.
    """
    # Flatten the dictionaries if nested
    if any(isinstance(value, Mapping) for value in dict_of_arrays.values()):
        dict_of_arrays = flatten_nested_dict(dict_of_arrays)
    if any(isinstance(value, Mapping) for value in other_dict_of_arrays.values()):
        other_dict_of_arrays = flatten_nested_dict(other_dict_of_arrays)

    # Check the keys
    if dict_of_arrays.keys() != other_dict_of_arrays.keys():
        return False

    if tolerance:
        compare_arrays = partial(
            allclose,
            rtol=tolerance,
            atol=tolerance,
            equal_nan=nan_are_equal,
        )
    else:
        compare_arrays = partial(array_equal, equal_nan=nan_are_equal)

    # Check the values
    for key, array_ in dict_of_arrays.items():
        other_array = other_dict_of_arrays[key]

        if not isinstance(array_, (ndarray, sparse_classes)):
            array_ = asarray(array_)

        if not isinstance(other_array, (ndarray, sparse_classes)):
            other_array = asarray(other_array)

        if array_.shape != other_array.shape:
            return False

        array_is_dense = isinstance(array_, ndarray)
        other_array_is_dense = isinstance(other_array, ndarray)

        if array_is_dense or other_array_is_dense:
            if not array_is_dense:
                array_ = array_.toarray()
            if not other_array_is_dense:
                other_array = other_array.toarray()
        # Sparsity is kept only when both arrays are sparse
        else:
            array_ = array_.data.reshape(-1)
            other_array = other_array.data.reshape(-1)

        if not compare_arrays(array_, other_array):
            return False

    return True
