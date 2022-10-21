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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re

import pytest
from gemseo.utils.data_conversion import array_to_dict
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import deepcopy_dict_of_arrays
from gemseo.utils.data_conversion import dict_to_array
from gemseo.utils.data_conversion import flatten_nested_bilevel_dict
from gemseo.utils.data_conversion import flatten_nested_dict
from gemseo.utils.data_conversion import nest_flat_bilevel_dict
from gemseo.utils.data_conversion import nest_flat_dict
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.data_conversion import update_dict_of_arrays_from_array
from gemseo.utils.testing import compare_dict_of_arrays
from numpy import array
from numpy import array_equal
from numpy import ndarray


@pytest.fixture
def dict_to_be_updated() -> dict[str, ndarray]:
    """A dictionary to be updated."""
    return {"x": array([0.0, 1.0]), "y": array([2.0]), "z": array([3, 4])}


@pytest.mark.parametrize(
    "values_array,cast_complex,expected",
    [
        (
            array([0.5, 1.0, 2.0]),
            False,
            {"y": array([0.5]), "z": array([1, 2])},
        ),
        (
            array([0, 1, 2]),
            True,
            {"y": array([0.5]), "z": array([1, 2])},
        ),
        (
            array([0.5 + 0j, 1.0 + 0j, 2.0 + 0j]),
            False,
            {"y": array([0.5 + 0j]), "z": array([1.0 + 0j, 2.0 + 0j])},
        ),
        (
            array([0.5 + 0j, 1.0 + 0j, 2.0 + 0j]),
            True,
            {"y": array([0.5]), "z": array([1, 2])},
        ),
    ],
)
def test_update_dict_of_arrays_from_array(
    dict_to_be_updated, values_array, expected, cast_complex
):
    """Check the update of a data mapping from a data array and variables names."""
    new_data_dict = update_dict_of_arrays_from_array(
        dict_to_be_updated, ["y", "z"], values_array, cast_complex=cast_complex
    )
    array_equal(new_data_dict, expected)


def test_update_dict_of_arrays_from_array_without_variables(dict_to_be_updated):
    """Check that a dictionary cannot be update without variables names."""
    new_data_dict = update_dict_of_arrays_from_array(
        dict_to_be_updated, [], array([0.5, 1.0, 2.0])
    )
    assert compare_dict_of_arrays(new_data_dict, dict_to_be_updated)


def test_update_dict_of_arrays_from_array_wrong_data_type(dict_to_be_updated):
    """Check that a dictionary cannot be updated from wrongly typed data."""
    expected = r"The array must be a NumPy one, got instead: <.+ 'float'>\."

    with pytest.raises(TypeError, match=expected):
        update_dict_of_arrays_from_array(
            dict_to_be_updated,
            ["y"],
            1.0,
        )


def test_update_dict_of_arrays_from_array_wrong_name():
    with pytest.raises(KeyError, match="y"):
        update_dict_of_arrays_from_array(
            {"z": array([1.0])},
            ["y"],
            array([0.5]),
        )
    with pytest.raises(AttributeError, match="'int' object has no attribute 'size'"):
        update_dict_of_arrays_from_array(
            {"y": 1},
            ["y"],
            array([0.5]),
        )


def test_update_dict_of_arrays_from_array_too_long(dict_to_be_updated):
    """Check that updating a dictionary with an array that is too long raises an
    error."""
    with pytest.raises(
        ValueError,
        match=(
            r"Inconsistent data shapes: "
            r"could not use the whole data array of shape \(2L?,\) "
            r"\(only reached max index = 1\), "
            r"while updating data dictionary names y of shapes: \[\(u?'y', \(1L?,\)\)\]\."
        ),
    ):
        update_dict_of_arrays_from_array(dict_to_be_updated, "y", array([0.5, 1.5]))


def test_update_dict_of_arrays_from_array_too_short(dict_to_be_updated):
    """Check that updating a dictionary with an array that is too short raises an
    error."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Inconsistent input array size of values array [0.5] "
            "with reference data shape (2,) for data named: z."
        ),
    ):
        update_dict_of_arrays_from_array(dict_to_be_updated, "z", array([0.5]))


def test_nest_flat_bilevel_dict_dict():
    """Check that a flattened bi-level mapping is correctly nested."""
    expected = {"a": {"b": 1}, "c": {"b": 2}}
    mapping = {"a_b": 1, "c_b": 2}
    assert nest_flat_bilevel_dict(mapping, separator="_") == expected


@pytest.fixture(scope="module")
def xy_sizes():
    """The sizes of x and y."""
    return {"x": 1, "y": 2}


@pytest.fixture(scope="module")
def xy_dict() -> dict[str, ndarray]:
    """The values of x and y."""
    return {"x": array([1.0]), "y": array([2.0, 3.0])}


@pytest.fixture(params=[False, True])
def possibly_nested_xy_dict(
    request,
) -> dict[str, ndarray | dict[str, ndarray]]:
    """A NumPy array with values for x and y."""
    if request.param:
        return {"x": {"x_1": array([1.0])}, "y": array([2.0, 3.0])}
    else:
        return {"x": array([1.0]), "y": array([2.0, 3.0])}


@pytest.fixture(scope="module")
def xy_array() -> ndarray:
    """The values of x and y."""
    return array([1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "names,expected",
    [
        ("x", array([1.0])),
        ("y", array([2.0, 3.0])),
        (["x", "y"], array([1.0, 2.0, 3.0])),
        (["y", "x"], array([2.0, 3.0, 1.0])),
        ([], array([])),
    ],
)
def test_concatenate_dict_of_arrays_to_array(xy_dict, names, expected):
    """Check concatenate_dict_of_arrays_to_array."""
    assert array_equal(concatenate_dict_of_arrays_to_array(xy_dict, names), expected)


@pytest.mark.parametrize(
    "names,expected",
    [
        (["x", "y"], {"x": array([1.0]), "y": array([2.0, 3.0])}),
        (["y", "x"], {"x": array([3.0]), "y": array([1.0, 2.0])}),
    ],
)
def test_split_array_to_dict_of_arrays(xy_array, xy_sizes, names, expected):
    """Check split_array_to_dict_of_arrays."""
    dict_1 = flatten_nested_dict(
        split_array_to_dict_of_arrays(xy_array, xy_sizes, names)
    )
    dict_2 = flatten_nested_dict(expected)
    assert compare_dict_of_arrays(dict_1, dict_2)


@pytest.mark.parametrize("y_size", [1, 3])
def test_split_array_to_dict_of_arrays_with_inconsistency_check(xy_array, y_size):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The total size of the elements ({}) "
            "and the size of the last dimension of the array ({}) "
            "are different.".format(1 + y_size, xy_array.shape[-1])
        ),
    ):
        split_array_to_dict_of_arrays(
            xy_array, {"x": 1, "y": y_size}, ["x", "y"], check_consistency=True
        )


@pytest.mark.parametrize("y_size", [1, 3])
def test_split_array_to_dict_of_arrays_without_inconsistency_check(xy_array, y_size):
    split_array_to_dict_of_arrays(xy_array, {"x": 1, "y": y_size}, ["x", "y"])


@pytest.fixture(scope="module")
def list_grouped_xy_dict(xy_dict) -> list[dict[str, dict[str, ndarray]]]:
    """A list of grouped data dictionaries."""
    return [
        {"g1": {"x": xy_dict["x"]}, "g2": {"y": xy_dict["y"]}},
        {"g1": {"x": xy_dict["x"] * 2}, "g2": {"y": xy_dict["y"] * 2}},
    ]


@pytest.fixture(scope="module")
def list_xy_dict(xy_dict) -> list[dict[str, ndarray]]:
    """A list of data dictionaries."""
    return [
        {"x": xy_dict["x"], "y": xy_dict["y"]},
        {"x": xy_dict["x"] * 2, "y": xy_dict["y"] * 2},
    ]


def test_split_array_to_dict_of_arrays_nested():
    """Check split_array_to_dict_of_arrays with a bi-level nested dictionary.

    The array has 2 dimensions.
    """
    jac_array = array([[1.0, 2.0, 2.0], [2.0, 4.0, 4.0], [2.0, 4.0, 4.0]])
    sizes = {"y1": 1, "y2": 2, "x2": 2, "x1": 1}
    jac_3d_dict = split_array_to_dict_of_arrays(
        jac_array, sizes, ["y1", "y2"], ["x1", "x2"]
    )
    expected = {
        "y1": {"x1": array([[1.0]]), "x2": array([[2.0, 2.0]])},
        "y2": {"x1": array([[2.0], [2.0]]), "x2": array([[4.0, 4.0], [4.0, 4.0]])},
    }
    assert compare_dict_of_arrays(jac_3d_dict, expected)


def test_split_array_to_dict_of_arrays_nested_3d():
    """Check split_array_to_dict_of_arrays with a bi-level nested dictionary.

    The array has 3 dimensions.
    """
    jac_array = array([[[1.0, 2.0, 2.0], [2.0, 4.0, 4.0], [2.0, 4.0, 4.0]]])
    sizes = {"y1": 1, "y2": 2, "x2": 2, "x1": 1}
    jac_3d_dict = split_array_to_dict_of_arrays(
        jac_array, sizes, ["y1", "y2"], ["x1", "x2"]
    )
    expected = {
        "y1": {"x1": array([[[1.0]]]), "x2": array([[[2.0, 2.0]]])},
        "y2": {"x1": array([[[2.0], [2.0]]]), "x2": array([[[4.0, 4.0], [4.0, 4.0]]])},
    }
    assert compare_dict_of_arrays(jac_3d_dict, expected)


def test_flatten_nested_dict():
    """Check flatten_nested_dict."""
    nested_jac_dict = {"y": {"x": array([[1.0], [2.0]])}}
    flatten_jac_dict = flatten_nested_dict(nested_jac_dict)
    assert compare_dict_of_arrays(flatten_jac_dict, {"y#&#x": array([[1.0], [2.0]])})


def test_flatten_nested_bilevel_dict():
    """Check flatten_nested_bilevel_dict."""
    nested_jac_dict = {"y": {"x": array([[1.0], [2.0]])}}
    flatten_jac_dict = flatten_nested_bilevel_dict(nested_jac_dict)
    assert compare_dict_of_arrays(flatten_jac_dict, {"y#&#x": array([[1.0], [2.0]])})


def test_nest_flat_dict():
    """Check nest_flat_dict."""
    flatten_jac_dict = {"y#&#x": array([[1.0], [2.0]])}
    nested_jac_dict = nest_flat_dict(flatten_jac_dict)
    assert compare_dict_of_arrays(nested_jac_dict, {"y": {"x": array([[1.0], [2.0]])}})


@pytest.mark.parametrize("names", ["x", "y", ["x", "y"], ["y", "x"], None])
def test_deepcopy_dict_of_arrays(possibly_nested_xy_dict, names):
    """Check deepcopy_dict_of_arrays."""
    original_dict = possibly_nested_xy_dict
    dict_copy = deepcopy_dict_of_arrays(original_dict)
    assert len(dict_copy) == len(original_dict or names)
    for key in dict_copy:
        if isinstance(original_dict[key], dict):
            assert array_equal(original_dict[key]["x_1"], dict_copy[key]["x_1"])
            assert id(original_dict[key]["x_1"]) != dict_copy[key]["x_1"]
        else:
            assert array_equal(original_dict[key], dict_copy[key])
            assert id(original_dict[key]) != id(dict_copy[key])


@pytest.mark.parametrize(
    "d_1,d_2,d_3,d_4",
    [
        (
            {"x": array([1.0]), "y": array([1, 2])},
            {"x": array([1.0]), "y": array([1, 2])},
            {"x": array([1.0]), "y": array([1, 3])},
            {"xx": array([1.0]), "y": array([1, 3])},
        ),
        (
            {"x": array([1.0]), "y": {"z": array([1, 2])}},
            {"x": array([1.0]), "y": {"z": array([1, 2])}},
            {"x": array([1.0]), "y": {"z": array([1, 3])}},
            {"xx": array([1.0]), "y": {"z": array([1, 3])}},
        ),
    ],
)
def test_compare_dict_of_arrays(d_1, d_2, d_3, d_4):
    """Check the comparison of dictionaries of NumPy arrays."""
    assert compare_dict_of_arrays(d_1, d_2)
    assert not compare_dict_of_arrays(d_1, d_3)
    assert not compare_dict_of_arrays(d_1, d_4)
    d_1_copy = deepcopy_dict_of_arrays(d_1)
    d_1_copy["x"] += 0.1
    assert not compare_dict_of_arrays(d_1, d_1_copy)
    assert compare_dict_of_arrays(d_1, d_1_copy, tolerance=0.1)


def test_aliases():
    """Check aliases."""
    assert dict_to_array == concatenate_dict_of_arrays_to_array
    assert array_to_dict == split_array_to_dict_of_arrays
