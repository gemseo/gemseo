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
"""Test the class Dataset."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy import allclose
from numpy import arange
from numpy import array
from numpy import concatenate
from numpy import int64 as np_int
from numpy import ndarray
from numpy import savetxt
from numpy import unique
from numpy import vstack
from numpy.testing import assert_equal
from packaging import version
from pandas import DataFrame
from pandas import MultiIndex
from pandas import __version__ as pandas_version
from pandas import concat
from pandas.testing import assert_frame_equal

from gemseo import create_dataset
from gemseo.datasets.dataset import Dataset
from gemseo.problems.dataset.iris import create_iris_dataset

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture()
def dataset(data: NDArray[np_int]) -> Dataset:
    """A dataset built from ``data``.

    The first feature is "var_1" and the other two are "var_2".
    """
    return Dataset.from_array(
        data, ["var_1", "var_2"], {"var_1": 1, "var_2": 2}, {"var_2": "foo"}
    )


@pytest.fixture()
def small_dataset(small_data: NDArray[np_int]) -> Dataset:
    """A small view of ``dataset``."""
    return Dataset.from_array(
        small_data, ["var_1", "var_2"], {"var_1": 1, "var_2": 2}, {"var_2": "foo"}
    )


@pytest.fixture()
def io_data() -> ndarray:
    """Input-output data."""
    return concatenate([arange(50).reshape(10, 5), arange(20).reshape(10, 2)], axis=1)


@pytest.fixture()
def io_dataset(io_data) -> Dataset:
    """An input-output dataset.

    The inputs are "i1" (2 components) and "i2" (3 components)
    and the output name is "o1" (2 components).

    The input group is "g1" and the output group is "g2".

    The input values are: 0   1  2  3  4 5   6  7  8  9 ... 45 46 47 48 49

    The output values are: 0 1 2 3 ... 8 9
    """
    return Dataset.from_array(
        io_data,
        ["i1", "i2", "o1"],
        {"i1": 2, "i2": 3, "o1": 2},
        {"i1": "g1", "i2": "g1", "o1": "g2"},
    )


@pytest.fixture()
def small_file_dataset(tmp_wd, small_data) -> str:
    """The generation of a small file.txt containing ``small_data``."""
    filename = "dataset.txt"
    savetxt(filename, small_data, delimiter=",")
    return filename


@pytest.fixture()
def file_dataset(
    tmp_wd,
    io_dataset,
) -> tuple[str, NDArray[int], list[str], dict[str, int], dict[str, str]]:
    """The information to build a dataset from a file."""
    data = io_dataset.to_numpy()
    variables = io_dataset.variable_names
    variable_names_to_n_components = io_dataset.variable_names_to_n_components
    groups = {
        variable: io_dataset.get_group_names(variable)[0] for variable in variables
    }
    filename = "dataset.txt"
    savetxt(filename, data, delimiter=",")
    return filename, data, variables, variable_names_to_n_components, groups


@pytest.mark.parametrize(
    (
        "update",
        "group_names",
        "variable_names",
        "components",
        "indices",
        "expected_array",
    ),
    [
        (
            1,
            (),
            "var_1",
            (),
            (),
            [
                [1, 1, 2],
                [1, 4, 5],
                [1, 7, 8],
            ],
        ),
        (
            [1, 2, 3],
            (),
            "var_1",
            (),
            (),
            [
                [1, 1, 2],
                [2, 4, 5],
                [3, 7, 8],
            ],
        ),
        (
            666,
            (),
            "var_2",
            (),
            (),
            [
                [0, 666, 666],
                [3, 666, 666],
                [6, 666, 666],
            ],
        ),
        (
            666,
            "foo",
            (),
            (),
            (),
            [
                [0, 666, 666],
                [3, 666, 666],
                [6, 666, 666],
            ],
        ),
        (
            666,
            (),
            ["var_1", "var_2"],
            (),
            (),
            [
                [666, 666, 666],
                [666, 666, 666],
                [666, 666, 666],
            ],
        ),
        (
            666,
            (),
            ["var_2"],
            (),
            1,
            [
                [0, 1, 2],
                [3, 666, 666],
                [6, 7, 8],
            ],
        ),
        (
            666,
            (),
            ["var_2"],
            (),
            [0, 2],
            [
                [0, 666, 666],
                [3, 4, 5],
                [6, 666, 666],
            ],
        ),
        (
            666,
            (),
            ["var_2"],
            0,
            [0, 2],
            [
                [0, 666, 2],
                [3, 4, 5],
                [6, 666, 8],
            ],
        ),
        (
            666,
            (),
            (),
            0,
            [0, 2],
            [
                [666, 666, 2],
                [3, 4, 5],
                [666, 666, 8],
            ],
        ),
        (
            [[900, 800, 700], [600, 500, 400]],
            (),
            (),
            [0, 1],
            [0, 2],
            [
                [900, 800, 700],
                [3, 4, 5],
                [600, 500, 400],
            ],
        ),
        (
            [[900, 800, 700], [600, 500, 400]],
            (),
            ["var_2", "var_1"],
            [0, 1],
            [0, 2],
            [
                [700, 900, 800],
                [3, 4, 5],
                [400, 600, 500],
            ],
        ),
        (
            666,
            ["parameters", "foo"],
            "var_1",
            [0, 1],
            [0, 2],
            [
                [666, 1, 2],
                [3, 4, 5],
                [666, 7, 8],
            ],
        ),
    ],
)
def test_update_data(
    small_dataset,
    update,
    group_names,
    variable_names,
    components,
    indices,
    expected_array,
):
    dataset_to_update = small_dataset.copy()
    dataset_to_update.update_data(
        update,
        group_names=group_names,
        variable_names=variable_names,
        components=components,
        indices=indices,
    )
    assert (dataset_to_update.to_numpy() == expected_array).all()


def test_update_data_errors(dataset):
    """Test the ``update_data`` method when inputs are incorrect."""
    dataset_to_update = dataset.copy()
    with pytest.raises(ValueError):
        dataset_to_update.update_data(
            [[900, 800], [600, 500]],
            group_names=(),
            variable_names=(),
            components=[0, 1],
            indices=[1, 3],
        )


def test_from_array(data):
    """Test the method from_array."""
    dataset = Dataset.from_array(data)
    columns = MultiIndex.from_tuples(
        [("parameters", "x_0", 0), ("parameters", "x_1", 0), ("parameters", "x_2", 0)],
        names=["GROUP", "VARIABLE", "COMPONENT"],
    )
    dataframe = DataFrame(data, columns=columns)
    dataframe._metadata = ["name"]
    dataframe.name = Dataset.__name__
    assert_frame_equal(dataset, dataframe)


from_array_parameters = pytest.mark.parametrize(
    "variable_names,variable_names_to_n_components,variable_names_to_group_names,expected_column_multi_index",
    [
        (
            ["y", "x"],
            {"x": 2},
            {"y": "g2"},
            [("g2", "y", 0), ("parameters", "x", 0), ("parameters", "x", 1)],
        ),
        (
            ["a", "b", "c"],
            None,
            {"a": "g2", "c": "g5"},
            [("g2", "a", 0), ("parameters", "b", 0), ("g5", "c", 0)],
        ),
        (
            ["a", "b", "c"],
            None,
            None,
            [("parameters", "a", 0), ("parameters", "b", 0), ("parameters", "c", 0)],
        ),
        (
            "x",
            {"x": 3},
            None,
            [("parameters", "x", 0), ("parameters", "x", 1), ("parameters", "x", 2)],
        ),
        (
            (),
            None,
            None,
            [
                ("parameters", "x_0", 0),
                ("parameters", "x_1", 0),
                ("parameters", "x_2", 0),
            ],
        ),
    ],
)


@from_array_parameters
def test_from_array_with_options(
    data,
    variable_names,
    variable_names_to_n_components,
    variable_names_to_group_names,
    expected_column_multi_index,
):
    """Test the method from_array with its options."""
    dataset = Dataset.from_array(
        data,
        variable_names=variable_names,
        variable_names_to_n_components=variable_names_to_n_components,
        variable_names_to_group_names=variable_names_to_group_names,
    )
    columns = MultiIndex.from_tuples(
        expected_column_multi_index,
        names=["GROUP", "VARIABLE", "COMPONENT"],
    )
    dataframe = DataFrame(data, columns=columns)
    dataframe._metadata = ["name"]
    dataframe.name = Dataset.__name__
    assert_frame_equal(dataset, dataframe)


@pytest.mark.parametrize(
    "excluded_variable_names", [["i1"], ["i2"], ["o1"], ["i1", "o1"], []]
)
@pytest.mark.parametrize("excluded_group_names", [["g1"], ["g2"], ["g1", "g2"], []])
@pytest.mark.parametrize("use_min_max", [True, False])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("scale", [True, False])
def test_get_normalized_dataset(
    io_dataset: Dataset,
    excluded_group_names,
    excluded_variable_names,
    use_min_max,
    center,
    scale,
):
    """Check the normalization of a dataset."""
    for group in excluded_group_names:
        excluded_variable_names.extend(io_dataset.get_variable_names(group))

    excluded_variable_names = unique(excluded_variable_names)
    dataset = io_dataset.get_normalized(
        excluded_variable_names, excluded_group_names, use_min_max, center, scale
    )

    # The excluded variables are not modified.
    assert_frame_equal(
        dataset.get_view(variable_names=excluded_variable_names),
        io_dataset.get_view(variable_names=excluded_variable_names),
    )

    for variable_name in dataset.variable_names:
        data = dataset.get_view(variable_names=variable_name).to_numpy()
        if variable_name not in excluded_variable_names:
            # Consider expected_data from simple requirements.
            expected_data = io_dataset.get_view(variable_names=variable_name).to_numpy()
            if use_min_max:
                expected_data = (expected_data - expected_data.min(axis=0)) / (
                    expected_data.max(axis=0) - expected_data.min(axis=0)
                )
            if center:
                expected_data -= expected_data.mean(axis=0)
            if scale:
                expected_data /= expected_data.std(axis=0)

            assert allclose(expected_data, data)


def test_transform_data():
    """Check the method transform_data."""
    dataset = Dataset()
    dataset.add_variable("x", 1.0, group_name="a")
    dataset.add_variable("x", 1.0, group_name="b")
    dataset.transform_data(lambda x: -x, variable_names="x")

    expected_dataset = Dataset()
    expected_dataset.add_variable("x", -1.0, group_name="a")
    expected_dataset.add_variable("x", -1.0, group_name="b")

    assert_frame_equal(dataset, expected_dataset)

    def f(x: ndarray) -> ndarray:
        """Return the opposite of an array.

        Args:
            x: The original array.

        Returns:
            The opposite of the original array.
        """
        return -x

    dataset.update_data = MagicMock()
    dataset.transform_data(f, "a", "x", [0], [0])
    dataset.update_data.assert_called_with(array([[1.0]]), "a", "x", [0], [0])


@pytest.mark.parametrize(
    (
        "group_name",
        "data",
        "variables",
        "variable_names_to_n_components",
        "expected_variable",
        "expected_components",
    ),
    [
        ("Toto", [[4], [4], [8]], "my_var", None, ["my_var"], {"my_var": 1}),
        ("parameters", [[5, 5, 5]], (), None, ["x"], {"x": 3}),
        (
            "Foo",
            [[1, 2, 3]],
            ["my_var", "your_var"],
            {"my_var": 2, "your_var": 1},
            ["my_var", "your_var"],
            {"my_var": 2, "your_var": 1},
        ),
        ("parameters", [[5, 5, 5]], (), {"y_1": 2, "y_2": 1}, ["x"], {"x": 3}),
        ("", [[5, 5, 5]], (), {"y_1": 2, "y_2": 1}, ["x"], {"x": 3}),
    ],
)
def test_add_groups(
    group_name,
    data,
    variables,
    variable_names_to_n_components,
    expected_variable,
    expected_components,
):
    """Test ``add_groups``."""
    dataset = Dataset()
    dataset.add_group(group_name, data, variables, variable_names_to_n_components)
    assert dataset.group_names == [group_name]
    assert np.isin(dataset.variable_names, expected_variable).all()
    for variable in dataset.variable_names:
        assert (
            dataset.get_variable_components(group_name, variable)
            == arange(expected_components.get(variable))
        ).all()


def test_add_group_error(dataset):
    """Test the method add_group with a group already defined."""
    with pytest.raises(
        ValueError, match=re.escape("The group 'parameters' is already defined.")
    ):
        dataset.add_group(dataset.PARAMETER_GROUP, 1)


@pytest.mark.parametrize("delimiter", [",", "/"])
def test_from_csv(tmp_wd, io_dataset, delimiter):
    """Test the ``from_csv`` method."""
    io_dataset.to_csv("io_dataset.csv", sep=delimiter, index=False)
    dataset = Dataset.from_csv("io_dataset.csv", delimiter=delimiter)
    assert_frame_equal(io_dataset.astype("int32"), dataset.astype("int32"))


@from_array_parameters
def test_from_txt(
    small_file_dataset,
    small_data,
    variable_names,
    variable_names_to_n_components,
    variable_names_to_group_names,
    expected_column_multi_index,
):
    """Test the ``from_txt`` method."""
    dataset = Dataset.from_txt(
        small_file_dataset,
        variable_names,
        variable_names_to_n_components,
        variable_names_to_group_names,
        header=False,
    ).astype("int32")

    columns = MultiIndex.from_tuples(
        expected_column_multi_index,
        names=["GROUP", "VARIABLE", "COMPONENT"],
    )
    dataframe = DataFrame(small_data, columns=columns).astype("int32")
    dataframe._metadata = ["name"]
    dataframe.name = Dataset.__name__
    assert_frame_equal(dataset, dataframe)


@pytest.mark.parametrize(
    ("variables", "as_tuple", "expected_output"),
    [
        ((), False, ["i1[0]", "i1[1]", "i2[0]", "i2[1]", "i2[2]", "o1[0]", "o1[1]"]),
        ("o1", False, ["o1[0]", "o1[1]"]),
        ("o1", True, [("g2", "o1", 0), ("g2", "o1", 1)]),
    ],
)
def test_get_columns(io_dataset, variables, as_tuple, expected_output):
    """Test the method get_columns."""
    assert (
        io_dataset.get_columns(variable_names=variables, as_tuple=as_tuple)
        == expected_output
    )


def test_name_default():
    """Test the property name with a default value."""
    assert Dataset().name == Dataset.__name__


def test_name_custom():
    """Test the property name with a custom value."""
    assert Dataset(dataset_name="foo").name == "foo"


@pytest.mark.parametrize("reverse_column_order", [False, True])
def test_variable_names(dataset, reverse_column_order):
    """Test the property variable_names."""
    if reverse_column_order:
        # Reverse the column order to check the alphabetical order.
        dataset = dataset[dataset.columns[::-1]]
    assert dataset.variable_names == ["var_1", "var_2"]


@pytest.mark.parametrize("reverse_column_order", [False, True])
def test_group_names(dataset, reverse_column_order):
    """Test the property group_names."""
    if reverse_column_order:
        dataset = dataset[dataset.columns[::-1]]
    assert dataset.group_names == ["foo", "parameters"]


@pytest.mark.parametrize("reverse_column_order", [False, True])
def test_variable_identifiers(dataset, reverse_column_order):
    """Test the property variable_identifiers."""
    if reverse_column_order:
        dataset = dataset[dataset.columns[::-1]]
    assert dataset.variable_identifiers == [
        ("foo", "var_2"),
        ("parameters", "var_1"),
    ]


def test_variable_names_to_n_components(dataset):
    """Test the property variable_names_to_n_components."""
    assert dataset.variable_names_to_n_components == {"var_1": 1, "var_2": 2}


def test_group_names_to_n_components(io_dataset):
    """Test the property group_names_to_n_components."""
    assert io_dataset.group_names_to_n_components == {"g1": 5, "g2": 2}


def test_get_variable_names(dataset):
    """Test the method get_variable_names."""
    assert dataset.get_variable_names("parameters") == ["var_1"]
    assert dataset.get_variable_names("foo") == ["var_2"]
    assert dataset.get_variable_names("not_existing_group") == []


def test_get_variable_components(dataset):
    """Test the method get_variable_components."""
    assert dataset.get_variable_components("parameters", "var_1") == [0]
    assert dataset.get_variable_components("foo", "var_2") == [0, 1]


def test_get_variable_components_custom():
    """Test the method get_variable_components with custom components."""
    dataset = Dataset()
    dataset.add_variable("x", array([[1, 1]]), components=[3, 5])
    assert dataset.get_variable_components("parameters", "x") == [3, 5]


@pytest.mark.parametrize("new_group", ["foo2", ""])
def test_rename_group(dataset, data, new_group):
    """The the method rename_group."""
    dataset_to_update = dataset.copy()
    dataset_to_update.rename_group("parameters", new_group)
    expected_dataset = Dataset.from_array(
        data,
        ["var_1", "var_2"],
        {"var_1": 1, "var_2": 2},
        {"var_1": new_group, "var_2": "foo"},
    )
    assert_frame_equal(dataset_to_update, expected_dataset)


def test_rename_variable(io_dataset, io_data):
    """Test the method rename_variable."""
    io_dataset.rename_variable("i1", "x")
    dataset = Dataset.from_array(
        io_data,
        ["x", "i2", "o1"],
        {"x": 2, "i2": 3, "o1": 2},
        {"x": "g1", "i2": "g1", "o1": "g2"},
    )
    assert_frame_equal(io_dataset, dataset)


def test_rename_variable_group_name(data):
    """Test the method rename_variable according to group_name."""
    dataset = Dataset()
    dataset.add_variable("x", 1, "a")
    dataset.add_variable("x", 1, "b")
    dataset.rename_variable("x", "z")

    expected_dataset = Dataset()
    expected_dataset.add_variable("z", 1, "a")
    expected_dataset.add_variable("z", 1, "b")
    assert_frame_equal(dataset, expected_dataset)

    dataset.rename_variable("z", "w", "a")

    expected_dataset = Dataset()
    expected_dataset.add_variable("w", 1, "a")
    expected_dataset.add_variable("z", 1, "b")
    assert_frame_equal(dataset, expected_dataset)


@pytest.mark.skipif(
    version.parse(pandas_version) >= version.parse("2.0.0"),
    reason="DataFrame does not get the append method in Pandas >= 2.0.0",
)
def test_append(dataset, data):
    """Test the method DataFrame.append."""
    new_dataset = dataset.append(dataset, ignore_index=True)
    expected_dataset = Dataset.from_array(
        vstack((data, data)),
        ["var_1", "var_2"],
        {"var_1": 1, "var_2": 2},
        {"var_2": "foo"},
    )
    assert_frame_equal(new_dataset, expected_dataset)


def test_concat(dataset, data):
    """Test the function concat of pandas."""
    new_dataset = concat([dataset, dataset], ignore_index=True)
    expected_dataset = Dataset.from_array(
        vstack((data, data)),
        ["var_1", "var_2"],
        {"var_1": 1, "var_2": 2},
        {"var_2": "foo"},
    )
    assert_frame_equal(new_dataset, expected_dataset)


@pytest.mark.parametrize(
    ("group_names", "variable_names", "components", "indices", "expected_array"),
    [
        (
            (),
            (),
            (),
            (),
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ],
        ),
        (
            ["parameters", "foo"],
            (),
            (),
            (),
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ],
        ),
        (
            ["foo", "parameters"],
            (),
            (),
            (),
            [
                [1, 2, 0],
                [4, 5, 3],
                [7, 8, 6],
            ],
        ),
        (
            (),
            ["var_2", "var_1"],
            (),
            (),
            [
                [1, 2, 0],
                [4, 5, 3],
                [7, 8, 6],
            ],
        ),
        (
            "parameters",
            (),
            (),
            (),
            [
                [0],
                [3],
                [6],
            ],
        ),
        (
            (),
            "var_1",
            (),
            (),
            [
                [0],
                [3],
                [6],
            ],
        ),
        (
            (),
            "var_2",
            [0, 1],
            [0, 2],
            [
                [1, 2],
                [7, 8],
            ],
        ),
        (
            (),
            "var_2",
            [1, 0],
            [0, 2],
            [
                [2, 1],
                [8, 7],
            ],
        ),
        (
            (),
            "var_2",
            [1, 0],
            [2, 0],
            [
                [8, 7],
                [2, 1],
            ],
        ),
        (
            (),
            "var_2",
            1,
            [0],
            [
                [2],
            ],
        ),
    ],
)
def test_get_view(
    small_dataset, group_names, variable_names, components, indices, expected_array
):
    """Test get_view with many options."""
    view = small_dataset.get_view(
        group_names=group_names,
        variable_names=variable_names,
        components=components,
        indices=indices,
    )
    assert isinstance(view, Dataset)
    assert (view.to_numpy() == expected_array).all()


@pytest.mark.skipif(
    version.parse(pandas_version) >= version.parse("2.0.0"),
    reason="Does not work with Pandas >= 2.0.0",
)
@pytest.mark.parametrize("arg", ["group_names", "variable_names", "components"])
def test_get_view_empty(dataset, arg):
    """Test that asking for a unknown column returns an empty dataset."""
    assert dataset.get_view(**{arg: "x"}).empty


@pytest.mark.skipif(
    version.parse(pandas_version) < version.parse("2.0.0"),
    reason="Does not work with Pandas < 2.0.0",
)
@pytest.mark.parametrize("arg", ["group_names", "variable_names", "components"])
def test_get_view_key_error(dataset, arg):
    """Test that asking for a unknown column raises a KeyError."""
    with pytest.raises(
        KeyError,
    ):
        dataset.get_view(**{arg: "x"})


def test_add_variable():
    """Test the method add_variable."""
    dataset = Dataset()
    dataset.add_variable("x", array([[1, 2], [3, 4]]))
    dataset.add_variable("x", array([[-1, -2], [-3, -4]]), "g", [1, 2])
    dataset.add_variable("y", array([[5], [6]]), components=1)
    dataset.add_variable("z", array([[5], [6]]), group_name="Foo")

    columns = MultiIndex.from_tuples(
        [
            ("parameters", "x", 0),
            ("parameters", "x", 1),
            ("g", "x", 1),
            ("g", "x", 2),
            ("parameters", "y", 1),
            ("Foo", "z", 0),
        ],
        names=["GROUP", "VARIABLE", "COMPONENT"],
    )
    data = array([[1, 2, -1, -2, 5, 5], [3, 4, -3, -4, 6, 6]])
    dataframe = DataFrame(data, columns=columns)
    dataframe._metadata = ["name"]
    dataframe.name = Dataset.__name__

    assert_frame_equal(dataset, dataframe)


def test_add_variable_twice():
    """Test the method add_variable to raise ValueError, when needed."""
    dataset = Dataset()
    dataset.add_variable("x", array([[1, 2], [3, 4]]))
    dataset.add_variable("x", array([[5], [6]]), components=2)
    with pytest.raises(
        ValueError,
        match=re.escape("The group 'parameters' has already a variable 'x' defined."),
    ):
        dataset.add_variable("x", array([[10, 20], [30, 40]]))


def test_data_shape_inconsistency():
    """Check that an exception is raised when data is inconsistent."""
    dataset = Dataset()
    dataset.add_variable("x", [[1, 1], [1, 1]])

    with pytest.raises(
        ValueError,
        match=re.escape("The data shape must be (2, 3) or (1, 3); got (3, 3) instead."),
    ):
        dataset.add_variable("y", [[1, 1, 1]] * 3)

    with pytest.raises(
        ValueError,
        match=re.escape("The data shape must be (2, 2) or (1, 2); got (3, 3) instead."),
    ):
        dataset.add_variable("y", [[1, 1, 1]] * 3, components=[1, 2])


def test_add_variable_constant():
    """Test adding a constant variable with data shaped as (1, d).

    This data is added to all the existing entries.
    """
    dataset = Dataset()
    dataset.add_variable("x", [[1, 1], [1, 1]])
    dataset.add_variable("y", [[2, 2, 2]])

    expected_dataset = Dataset()
    expected_dataset.add_variable("x", [[1, 1], [1, 1]])
    expected_dataset.add_variable("y", [[2, 2, 2], [2, 2, 2]])

    assert_frame_equal(dataset, expected_dataset)


def test_add_variable_number():
    """Test adding a constant variable with data as number."""
    dataset = Dataset()
    dataset.add_variable("x", [[1, 1], [1, 1]])
    dataset.add_variable("y", 2)

    expected_dataset = Dataset()
    expected_dataset.add_variable("x", [[1, 1], [1, 1]])
    expected_dataset.add_variable("y", [[2], [2]])

    assert_frame_equal(dataset, expected_dataset)

    dataset.add_variable("z", 3, components=[4, 5])
    expected_dataset.add_variable("z", [[3, 3], [3, 3]], components=[4, 5])

    assert_frame_equal(dataset, expected_dataset)


def test_slice(dataset):
    """Test the use of slice."""
    view = dataset.get_view(indices=slice(1, 3))
    assert_frame_equal(view, dataset.iloc[[1, 2, 3]])


def test_get_group_names():
    """Test the ``get_group_names`` method."""
    dataset = Dataset()
    dataset.add_variable("x", array([[1, 2], [3, 4]]))
    dataset.add_variable("x", array([[-1, -2], [-3, -4]]), "g", [1, 2])
    dataset.add_variable("y", array([[5], [6]]), components=1)
    dataset.add_variable("z", array([[5], [6]]), group_name="Foo")

    assert dataset.get_group_names("x") == ["g", "parameters"]
    assert dataset.get_group_names("y") == ["parameters"]
    assert dataset.get_group_names("z") == ["Foo"]


@pytest.fixture(scope="module")
def dataset_for_to_dict_of_arrays() -> Dataset:
    """A dataset to test the method to_dict_of_arrays."""
    dataset = Dataset()
    dataset.add_variable("x", array([[1, 1], [-1, -1]]), "A")
    dataset.add_variable("y", array([[2], [-2]]), "A")
    dataset.add_variable("x", array([[3, 3, 3], [-3, -3, -3]]), "B")
    dataset.add_variable("z", array([[4], [-4]]), "B")
    return dataset


def test_to_dict_of_arrays(dataset_for_to_dict_of_arrays):
    """Test the method to_dict_of_arrays with default options."""
    result = dataset_for_to_dict_of_arrays.to_dict_of_arrays()
    expected = {
        "A": {"x": array([[1, 1], [-1, -1]]), "y": array([[2], [-2]])},
        "B": {"x": array([[3, 3, 3], [-3, -3, -3]]), "z": array([[4], [-4]])},
    }
    assert_equal(result, expected)


def test_to_dict_of_arrays_by_variable_name(dataset_for_to_dict_of_arrays):
    """Test the method to_dict_of_arrays without sorting by group."""
    result = dataset_for_to_dict_of_arrays.to_dict_of_arrays(False)
    expected = {
        "y": array([[2], [-2]]),
        "z": array([[4], [-4]]),
        "A:x": array([[1, 1], [-1, -1]]),
        "B:x": array([[3, 3, 3], [-3, -3, -3]]),
    }
    assert_equal(result, expected)


def test_to_dict_of_arrays_by_entry_by_variable_name(dataset_for_to_dict_of_arrays):
    """Test the method to_dict_of_arrays with sorting by entry and by variable name."""
    result = dataset_for_to_dict_of_arrays.to_dict_of_arrays(False, True)
    expected = [
        {
            "y": array([2]),
            "z": array([4]),
            "A:x": array([1, 1]),
            "B:x": array([3, 3, 3]),
        },
        {
            "y": array([-2]),
            "z": array([-4]),
            "A:x": array([-1, -1]),
            "B:x": array([-3, -3, -3]),
        },
    ]
    assert_equal(result, expected)


def test_summary():
    """Test the property summary."""
    assert create_iris_dataset().summary == (
        "Iris\n"
        "   Class: Dataset\n"
        "   Number of entries: 150\n"
        "   Number of variable identifiers: 5\n"
        "   Variables names and sizes by group:\n"
        "      labels: specy (1)\n"
        "      parameters: petal_length (1), petal_width (1), sepal_length (1) and "
        "sepal_width (1)\n"
        "   Number of dimensions (total = 5) by group:\n"
        "      labels: 1\n"
        "      parameters: 4"
    )
    assert create_iris_dataset(True).summary == (
        "Iris\n"
        "   Class: IODataset\n"
        "   Number of entries: 150\n"
        "   Number of variable identifiers: 5\n"
        "   Variables names and sizes by group:\n"
        "      inputs: petal_length (1), petal_width (1), sepal_length (1) and "
        "sepal_width (1)\n"
        "      outputs: specy (1)\n"
        "   Number of dimensions (total = 5) by group:\n"
        "      inputs: 4\n"
        "      outputs: 1"
    )


def test_create_empty_dataset():
    """Check the high-level function create_dataset to create an empty dataset."""
    dataset = create_dataset("foo")
    assert dataset.empty
    assert dataset.name == "foo"


def test_create_dataset_with_wrong_data():
    """Check the high-level function create_dataset from a wrong type."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dataset can be created from an array or a .csv or .txt file, "
            "not a <class 'int'>."
        ),
    ):
        create_dataset("foo", 123)


def test_create_dataset_from_wrong_file_extension():
    """Check the high-level function create_dataset from a .png file."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dataset can be created from a file with a .csv or .txt extension, "
            "not .png."
        ),
    ):
        create_dataset("foo", "file_name.png")


def test_create_dataset_from_csv_file(tmp_wd):
    """Check the high-level function create_dataset from a .csv file."""
    dataset = create_dataset("foo", array([[1], [2]]))
    dataset.to_csv("bar.csv", sep="#", index=False)
    other_dataset = create_dataset("foo", "bar.csv", delimiter="#")
    assert_frame_equal(dataset.astype("int32"), other_dataset.astype("int32"))


@from_array_parameters
def test_create_dataset_from_txt_file(
    small_file_dataset,
    small_data,
    variable_names,
    variable_names_to_n_components,
    variable_names_to_group_names,
    expected_column_multi_index,
):
    """Check the high-level function create_dataset from a .txt file."""
    dataset = Dataset.from_txt(
        small_file_dataset,
        variable_names,
        variable_names_to_n_components,
        variable_names_to_group_names,
        header=False,
    )
    other_dataset = create_dataset(
        "Dataset",
        small_file_dataset,
        variable_names,
        variable_names_to_n_components,
        variable_names_to_group_names,
        header=False,
    )
    assert_frame_equal(dataset, other_dataset)
