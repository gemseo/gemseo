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

from typing import Mapping
from typing import Union

from numpy import asarray
from numpy import ndarray
from numpy.linalg import norm
from scipy.sparse import spmatrix
from scipy.sparse.linalg import norm as spnorm

from gemseo.utils.data_conversion import flatten_nested_dict

ArrayLike = Union[ndarray, spmatrix]


def compare_dict_of_arrays(
    dict_of_arrays: Mapping[str, ArrayLike] | Mapping[str, Mapping[str, ArrayLike]],
    other_dict_of_arrays: Mapping[str, ArrayLike]
    | Mapping[str, Mapping[str, ArrayLike]],
    tolerance: float = 0.0,
) -> bool:
    """Check if two dictionaries of NumPy arrays and/or SciPy sparse matrices are equal.

    These dictionaries can be nested.

    Args:
        dict_of_arrays: A dictionary of NumPy arrays and/or SciPy sparse matrices.
        other_dict_of_arrays: Another dictionary of NumPy arrays and/or SciPy sparse
            matrices.
        tolerance: A relative tolerance. The dictionaries are considered equal if for
            any key ``reference_name`` of ``reference_dict_of_arrays``,
            ``norm(dict_of_arrays[name] - reference_dict_of_arrays[name])
            /(1 + norm(reference_dict_of_arrays)) <= tolerance``.

    Returns:
        Whether the dictionaries are equal.
    """
    # Flatten the dictionnaries if nested
    if any(isinstance(value, Mapping) for value in dict_of_arrays.values()):
        dict_of_arrays = flatten_nested_dict(dict_of_arrays)
        other_dict_of_arrays = flatten_nested_dict(other_dict_of_arrays)

    # Check the keys
    if dict_of_arrays.keys() != other_dict_of_arrays.keys():
        return False

    # Check the values
    if tolerance:
        for key, value in dict_of_arrays.items():
            difference = other_dict_of_arrays[key] - value

            if isinstance(difference, spmatrix):
                norm_diff = spnorm(difference)
            else:
                norm_diff = norm(difference)

            if isinstance(value, spmatrix):
                norm_ref = spnorm(value)
            else:
                norm_ref = norm(value)

            if norm_diff > tolerance * (1.0 + norm_ref):
                return False
    else:
        for key, value in dict_of_arrays.items():
            is_different = other_dict_of_arrays[key] != value

            if isinstance(is_different, spmatrix):
                is_different = is_different.data

            if asarray(is_different).any():
                return False

    return True
