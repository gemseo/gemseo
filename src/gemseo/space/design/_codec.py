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
"""Codec for switching between the variables and the full vector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import concatenate

from gemseo.util._numpy import get_common_dtype
from gemseo.util.data_conversion import split_array_to_dict_of_arrays

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from numpy import ndarray

    from gemseo.space.design._variables import Variables


def split_full_value(value: ndarray, variables: Variables) -> dict[str, ndarray]:
    """Split a full value by variable name.

    Args:
        value: The full value.
        variables: The variables that defines the full value.

    Returns:
        The map from a variable name to a variable value.
    """
    name_to_size = {name: variable.size for name, variable in variables.items()}
    return split_array_to_dict_of_arrays(value, name_to_size, variables)


def concatenate_values(
    name_to_value: Mapping[str, ndarray],
    names: Iterable[str],
) -> ndarray:
    """Concatenate the values of the variables.

    Args:
        name_to_value: The map from a variable name to a variable value.
        names: The names of the variables to be concatenated, in order.

    Returns:
        The full value.
    """
    values = tuple(name_to_value[name] for name in names)
    if not values:
        return array([])

    common_dtype = get_common_dtype(values)
    return concatenate(values, axis=-1).astype(common_dtype)
