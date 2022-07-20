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
from typing import Any
from typing import Generator
from typing import Iterable
from typing import Mapping

from numpy import array
from numpy import concatenate
from numpy import ndarray

STRING_SEPARATOR = "#&#"


def concatenate_dict_of_arrays_to_array(
    dict_of_arrays: Mapping[str, ndarray],
    names: Iterable[str],
) -> ndarray:
    """Concatenate some values of a dictionary of NumPy arrays.

    The concatenation is done according to the last dimension of the NumPy arrays.
    This dimension apart, the NumPy arrays must have the same shape.

    Example:
        >>> result = concatenate_dict_of_arrays_to_array(
        ...     {'x': array([1.]), 'y': array([2.]), 'z': array([3., 4.])}, ['x', 'z']
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

    return concatenate([dict_of_arrays[key] for key in names], -1)


dict_to_array = concatenate_dict_of_arrays_to_array


def split_array_to_dict_of_arrays(
    array: ndarray,
    names_to_sizes: Mapping[str, int],
    *names: Iterable[str],
    check_consistency: bool = False,
) -> dict[str, ndarray | dict[str, ndarray]]:
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
        variables_size = sum(names_to_sizes[name] for name in names[0])
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


array_to_dict = split_array_to_dict_of_arrays


def update_dict_of_arrays_from_array(
    dict_of_arrays: Mapping[str, ndarray],
    names: Iterable[str],
    array: ndarray,
    copy: bool = True,
    cast_complex: bool = False,
) -> Mapping[str, ndarray]:
    """Update some values of a dictionary of NumPy arrays from a NumPy array.

    The order of the data in ``array`` follows the order of ``names``.
    The original data type is kept
    except if `array` is complex and ``cast_complex`` is ``False``.

    Example:
        >>> result = update_dict_of_arrays_from_array(
        ...     {"x": array([0.0, 1.0]), "y": array([2.0]), "z": array([3, 4])},
        ...     ["y", "z"],
        ...     array([0.5, 1.0, 2.0])
        ... )
        >>> print(result)
        {"x": array([0.0, 1.0]), "y": array([0.5]), "z": array([1, 2])}

    Args:
        dict_of_arrays: The dictionary of NumPy arrays to be updated.
        names: The keys of the dictionary for which to update the values.
        array: The NumPy array with which to update the dictionary of NumPy arrays.
        copy: Whether to update a copy ``reference_input_data``.
        copy: Whether to update ``dict_of_arrays`` or a copy of ``dict_of_arrays``.
        cast_complex: Whether to cast ``array`` when its data type is complex.

    Returns:
        A deep copy of ``dict_of_arrays``
        whose values of ``names``, if any, have been updated with ``array``.

    Raises:
        TypeError: If ``array`` is not a NumPy array.
        ValueError:

            * If a name of ``names`` is not a key of ``dict_of_arrays``.
            * If the size of ``array`` is inconsistent
              with the shapes of the values of ``dict_of_arrays``.
    """
    if not isinstance(array, ndarray):
        raise TypeError(f"The array must be a NumPy one, got instead: {type(array)}.")

    if copy:
        data = deepcopy(dict_of_arrays)
    else:
        data = dict_of_arrays

    if not names:
        return data

    i_min = 0
    i_max = 0
    full_size = array.size
    try:
        for data_name in names:
            data_value = dict_of_arrays[data_name]
            i_max = i_min + data_value.size
            new_data_value = array[range(i_min, i_max)]
            is_complex = new_data_value.dtype.kind == "c"
            if not is_complex or (is_complex and cast_complex):
                new_data_value = new_data_value.astype(data_value.dtype)

            data[data_name] = new_data_value
            i_min = i_max
    except IndexError as err:
        if full_size < i_max:
            raise ValueError(
                "Inconsistent input array size of values array {} "
                "with reference data shape {} "
                "for data named: {}.".format(array, data_value.shape, data_name)
            )
        else:
            raise err

    if i_max != full_size:
        raise ValueError(
            "Inconsistent data shapes: "
            "could not use the whole data array of shape {} "
            "(only reached max index = {}), "
            "while updating data dictionary names {} "
            "of shapes: {}.".format(
                array.shape,
                i_max,
                names,
                [(data_name, dict_of_arrays[data_name].shape) for data_name in names],
            )
        )

    return data


def deepcopy_dict_of_arrays(
    dict_of_arrays: Mapping[str, ndarray],
    names: Iterable[str] | None = None,
) -> dict[str, ndarray]:
    """Perform a deep copy of a dictionary of NumPy arrays.

    This treats the NumPy arrays specially
    using ``array.copy()`` instead of ``deepcopy``.

    Example:
        >>> result = deepcopy_dict_of_arrays(
        ...     {"x": array([1.]), "y": array([2.])}, ["x"]
        ... )
        >>> print(result)
        >>> {"x": array([1.])}

    Args:
        dict_of_arrays: The dictionary of NumPy arrays to be copied.
        names: The keys of the dictionary for which to deepcopy the items.
            If None, consider all the dictionary keys.

    Returns:
        A deep copy of the dictionary of NumPy arrays.
    """
    deep_copy = {}
    selected_keys = dict_of_arrays.keys()
    if names is not None:
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
    flat_dict: Mapping[str, Any],
    separator: str = STRING_SEPARATOR,
) -> dict[str, Any]:
    """Nest a flat bi-level dictionary where sub-dictionaries will have the same keys.

    Example:
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
    flat_dict: Mapping[str, Any],
    prefix: str = "",
    separator: str = STRING_SEPARATOR,
) -> dict[str, Any]:
    """Nest a flat dictionary.

    Example:
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
        if key.startswith(prefix):
            key = key[len(prefix) :]
        __nest_flat_mapping(nested_dict, key, value, separator)

    return nested_dict


def __nest_flat_mapping(
    mapping: Mapping[str, Any],
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
    nested_dict: Mapping[str, Any],
    separator: str = STRING_SEPARATOR,
) -> dict[str, Any]:
    """Flatten a nested bi-level dictionary whose sub-dictionaries have the same keys.

    Example:
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
    nested_dict: Mapping[str, Any],
    prefix: str = "",
    separator: str = STRING_SEPARATOR,
) -> dict[str, Any]:
    """Flatten a nested dictionary.

    Example:
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
    nested_mapping: Mapping[str, Any],
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
        if parent_key:
            new_key = separator.join([parent_key, key])
        else:
            new_key = key

        if isinstance(value, collections.abc.Mapping):
            yield from flatten_nested_dict(value, new_key, separator=separator).items()
        else:
            yield new_key, value
