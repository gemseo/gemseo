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

# Contributors:
# Matthias De Lozzo
# Antoine DECHAUME

from __future__ import division, unicode_literals

from typing import Mapping

from numpy import array_equal, ndarray

from gemseo.utils.data_conversion import flatten_nested_dict


def compare_dict_of_arrays(
    dict_of_arrays,  # type: Mapping[str,ndarray]
    other_dict_of_arrays,  # type: Mapping[str,ndarray]
):  # type: (...) -> bool
    """Check if two dictionaries of NumPy arrays are equal.

    These dictionaries can be nested.

    Args:
        dict_of_arrays: A dictionary of NumPy arrays.
        other_dict_of_arrays: Another dictionary of NumPy arrays.

    Returns:
        Whether the dictionaries are equal.
    """
    if any(isinstance(value, Mapping) for value in dict_of_arrays.values()):
        dict_of_arrays = flatten_nested_dict(dict_of_arrays)
        other_dict_of_arrays = flatten_nested_dict(other_dict_of_arrays)

    for key, value in other_dict_of_arrays.items():
        if key not in dict_of_arrays or not array_equal(dict_of_arrays[key], value):
            return False

    return True
