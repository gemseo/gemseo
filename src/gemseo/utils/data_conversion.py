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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A set of functions to convert data structures."""

from __future__ import annotations

import collections
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any

from numpy import array
from numpy import concatenate
from numpy import ndarray

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Iterable
    from collections.abc import Mapping

    from numpy.typing import ArrayLike

    from gemseo.typing import StrKeyMapping


STRING_SEPARATOR = "#&#"


def concatenate_dict_of_arrays_to_array(
    dict_of_arrays: Mapping[str, ArrayLike],
    names: Iterable[str],
) -> ndarray:
    """Concatenate some values of a dictionary of NumPy arrays.

    The concatenation is done according to the last dimension of the NumPy arrays.
    This dimension apart, the NumPy arrays must have the same shape.

    Examples:
        >>> result = concatenate_dict_of_arrays_to_array(
        ...     {"x": array([1.0]), "y": array([2.0]), "z": array([3.0, 4.0])},
        ...     ["x", "z"],
        ... )
        >>> print(result)
        array([1., 3., 4.])

    Args:
        dict_of_arrays: The dictionary of NumPy arrays.
        names: The keys of the dictionary for which to concatenate the values.

    Returns:
        The concatenated array if ``names`` is not empty, otherwise an empty array.
    """
    if not names:
        return array([])

    return concatenate([dict_of_arrays[key] for key in names], axis=-1)


def split_array_to_dict_of_arrays(
    array: ndarray,
    names_to_sizes: Mapping[str, int],
    *names: Iterable[str],
    check_consistency: bool = False,
) -> dict[str, ndarray | dict[str, ndarray]]:
    """Split a NumPy array into a dictionary of NumPy arrays.

    Examples:
        >>> result_1 = split_array_to_dict_of_arrays(
        ...     array([1.0, 2.0, 3.0]), {"x": 1, "y": 2}, ["x", "y"]
        ... )
        >>> print(result_1)
        {'x': array([1.]), 'y': array([2., 3.])}
        >>> result_2 = split_array_to_dict_of_arrays(
        ...     array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        ...     {"y1": 1, "y2": 2, "x2": 2, "x1": 1},
        ...     ["y1", "y2"],
        ...     ["x1", "x2"],
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
        variables_size = sum(names_to_sizes[name] for name in names[0])
        array_dimension_size = array.shape[dimension]
        if variables_size != array_dimension_size:
            msg = (
                f"The total size of the elements ({variables_size}) "
                f"and the size of the last dimension of the array "
                f"({array_dimension_size}) are different."
            )
            raise ValueError(msg)

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


def deepcopy_dict_of_arrays(
    dict_of_arrays: StrKeyMapping,
    names: Iterable[str] = (),
) -> StrKeyMapping:
    """Perform a deep copy of a dictionary of NumPy arrays.

    This treats the NumPy arrays specially
    using ``array.copy()`` instead of ``deepcopy``.

    Examples:
        >>> result = deepcopy_dict_of_arrays(
        ...     {"x": array([1.0]), "y": array([2.0])}, ["x"]
        ... )
        >>> print(result)
        >>> {"x": array([1.0])}

    Args:
        dict_of_arrays: The dictionary of NumPy arrays to be copied.
        names: The keys of the dictionary for which to deepcopy the items.
            If empty, consider all the dictionary keys.

    Returns:
        A deep copy of the dictionary of NumPy arrays.
    """
    deep_copy = {}
    selected_keys = dict_of_arrays.keys()
    if names:
        selected_keys = [name for name in names if name in selected_keys]
        # TODO: either let the following block raise a KeyError or log a warning

    for key in selected_keys:
        value = dict_of_arrays[key]
        if isinstance(value, ndarray):
            deep_copy[key] = value.copy()
        else:
            deep_copy[key] = deepcopy(value)

    return deep_copy


def nest_flat_bilevel_dict(
    flat_dict: StrKeyMapping,
    separator: str = STRING_SEPARATOR,
) -> StrKeyMapping:
    """Nest a flat bi-level dictionary where sub-dictionaries will have the same keys.

    Examples:
        >>> result = nest_flat_bilevel_dict({"a_b": 1, "c_b": 2}, "_")
        >>> print(result)
        {"a": {"b": 1}, "c": {"b": 2}}

    Args:
        flat_dict: The dictionary to be nested.
        separator: The keys separator, to be used as ``{parent_key}{sep}{child_key}``.

    Returns:
        A nested dictionary.
    """
    keys = [key.split(separator) for key in flat_dict]
    top_keys = {key[0] for key in keys}
    sub_keys = {key[1] for key in keys}
    nested_dict = {}
    for top_key in top_keys:
        top_value = nested_dict[top_key] = {}
        for sub_key in sub_keys:
            key = separator.join([top_key, sub_key])
            top_value[sub_key] = flat_dict[key]

    return nested_dict


def nest_flat_dict(
    flat_dict: StrKeyMapping,
    prefix: str = "",
    separator: str = STRING_SEPARATOR,
) -> StrKeyMapping:
    """Nest a flat dictionary.

    Examples:
        >>> result = nest_flat_dict({"a_b": 1, "c_b": 2}, separator="_")
        >>> print(result)
        {"a": {"b": 1}, "c": {"b": 2}}

    Args:
        flat_dict: The dictionary to be nested.
        prefix: The prefix to be removed from the keys.
        separator: The keys separator,
            to be used as ``{parent_key}{separator}{child_key}``.

    Returns:
        A nested dictionary.
    """
    nested_dict = {}
    for key, value in flat_dict.items():
        key = key.removeprefix(prefix)
        __nest_flat_mapping(nested_dict, key, value, separator)

    return nested_dict


def __nest_flat_mapping(
    mapping: StrKeyMapping,
    key: str,
    value: Any,
    separator: str,
) -> None:
    """Nest a flat mapping.

    Args:
        mapping: The mapping to be nested.
        key: The current key.
        value: The current value.
        separator: The keys separator,
            to be used as ``{parent_key}{separator}{child_key}``.
    """
    keys = key.split(separator)
    top_key = keys[0]
    sub_keys = separator.join(keys[1:])
    if sub_keys:
        __nest_flat_mapping(mapping.setdefault(top_key, {}), sub_keys, value, separator)
    else:
        mapping[top_key] = value


def flatten_nested_bilevel_dict(
    nested_dict: StrKeyMapping,
    separator: str = STRING_SEPARATOR,
) -> StrKeyMapping:
    """Flatten a nested bi-level dictionary whose sub-dictionaries have the same keys.

    Examples:
        >>> result = flatten_nested_bilevel_dict({"y": {"x": array([[1.0], [2.0]])}})
        >>> print(result)
        {"y#&#x": array([[1.0], [2.0]])}

    Args:
        nested_dict: The dictionary to be flattened.
        separator: The keys separator,
            to be used as ``{parent_key}{separator}{child_key}``.

    Returns:
        A flat dictionary.
    """
    flat_dict = {}
    for top_key, top_value in nested_dict.items():
        for sub_key, sub_value in top_value.items():
            key = separator.join([top_key, sub_key])
            flat_dict[key] = sub_value

    return flat_dict


def flatten_nested_dict(
    nested_dict: StrKeyMapping,
    prefix: str = "",
    separator: str = STRING_SEPARATOR,
) -> StrKeyMapping:
    """Flatten a nested dictionary.

    Examples:
        >>> result = flatten_nested_dict({"y": {"x": array([[1.0], [2.0]])}})
        >>> print(result)
        {"y#&#x": array([[1.0], [2.0]])}

    Args:
        nested_dict: The dictionary to be flattened.
        prefix: The prefix to be prepended to the keys.
        separator: The keys separator,
            to be used as ``{parent_key}{separator}{child_key}``.

    Returns:
        A flat dictionary.
    """
    return dict(__flatten_nested_mapping(nested_dict, prefix, separator))


def __flatten_nested_mapping(
    nested_mapping: StrKeyMapping,
    parent_key: str,
    separator: str,
) -> Generator[tuple[str, Any], None, None]:
    """Flatten a nested mapping.

    Args:
        nested_mapping: The mapping to be flattened.
        parent_key: The key for which ``mapping`` is the value.
        separator: The keys separator,
            to be used as ``{parent_key}{separator}{child_key}``.

    Yields:
        The new keys and values of the mapping.
    """
    for key, value in nested_mapping.items():
        new_key = separator.join([parent_key, key]) if parent_key else key

        if isinstance(value, collections.abc.Mapping):
            yield from flatten_nested_dict(value, new_key, separator=separator).items()
        else:
            yield new_key, value
