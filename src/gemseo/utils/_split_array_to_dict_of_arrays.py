# -*- coding: utf-8 -*-
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
"""Split an array to a dictionary of arrays (Python 3)."""
from typing import Dict
from typing import Iterable
from typing import Mapping

from numpy import ndarray


def split_array_to_dict_of_arrays(
    array,  # type: ndarray
    names_to_sizes,  # type: Mapping[str,int]
    *names,  # type: Iterable[str]
    check_consistency=False,  # type: bool
):  # type: (...) -> Dict[str,ndarray]
    """Split a NumPy array into a dictionary of NumPy arrays.

    Example:
        >>> result_1 = split_array_to_dict_of_arrays(
        ...     array([1., 2., 3.]), {"x": 1, "y": 2}, ["x", "y"]
        ... )
        >>> print(result_1)
        {'x': array([1.]), 'y': array([2., 3.])}
        >>> result_2 = split_array_to_dict_of_arrays(
        ...     array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        ...     {"y1": 1, "y2": 2, "x2": 2, "x1": 1},
        ...     ["y1", "y2"],
        ...     ["x1", "x2"]
        ... )
        >>> print(result_2)
        {
            "y1": {"x1": array([[1.0]]), "x2": array([[2.0, 3.0]])},
            "y2": {"x1": array([[4.0], [7.0]]), "x2": array([[5.0, 6.0], [8.0, 9.0]])},
        }

    Args:
        array: The NumPy array.
        names_to_sizes: The sizes of the values related to names.
        *names: The names related to the NumPy array dimensions,
            starting from the last one;
            in the second example (see ``result_2``),
            the last dimension of ``array`` represents the variables ``["y1", "y2"]``
            while the penultimate one represents the variables ``["x1", "x2"]``.
        check_consistency: Whether to check the consistency of the sizes of ``*names``
            with the ``array`` shape.

    Returns:
        A dictionary of NumPy arrays related to ``*names``.

    Raises:
        ValueError: When ``check_consistency`` is ``True`` and
            the sizes of the ``*names`` is inconsistent with the ``array`` shape.
    """
    dimension = -len(names)
    if check_consistency:
        variables_size = sum([names_to_sizes[name] for name in names[0]])
        array_dimension_size = array.shape[dimension]
        if variables_size != array_dimension_size:
            raise ValueError(
                "The total size of the elements ({}) "
                "and the size of the last dimension of the array ({}) "
                "are different.".format(variables_size, array_dimension_size)
            )

    result = {}
    first_index = 0
    for name in names[0]:
        size = names_to_sizes[name]
        indices = [slice(None)] * array.ndim
        indices[dimension] = slice(first_index, first_index + size)
        if dimension == -1:
            result[name] = array[tuple(indices)]
        else:
            result[name] = split_array_to_dict_of_arrays(
                array[tuple(indices)],
                names_to_sizes,
                *names[1:],
                check_consistency=check_consistency,
            )

        first_index += size

    return result
