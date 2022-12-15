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
#                           documentation
#        :author: Matthias De Lozzo
"""Test the dataset module."""
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.core.dataset import Dataset
from gemseo.core.dataset import LOGICAL_OPERATORS
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.utils.string_tools import MultiLineString
from numpy import allclose
from numpy import arange
from numpy import array
from numpy import concatenate
from numpy import nan
from numpy import ones
from numpy import savetxt
from numpy import zeros
from numpy.testing import assert_equal


@pytest.fixture
def data():
    return arange(30).reshape(10, 3)


@pytest.fixture
def dataset(data):
    variables = ["var_1", "var_2"]
    sizes = {"var_1": 1, "var_2": 2}
    tmp = Dataset(name="my_dataset")
    tmp.set_from_array(data, variables, sizes)
    return tmp


@pytest.fixture
def ungroup_dataset(data):
    variables = ["var_1", "var_2"]
    sizes = {"var_1": 1, "var_2": 2}
    tmp = Dataset(by_group=False)
    tmp.set_from_array(data, variables, sizes)
    return tmp


@pytest.fixture
def io_dataset():
    inputs = arange(50).reshape(10, 5)
    outputs = arange(20).reshape(10, 2)
    data = concatenate([inputs, outputs], axis=1)
    variables = ["in_1", "in_2", "out_1"]
    sizes = {"in_1": 2, "in_2": 3, "out_1": 2}
    groups = {"in_1": "inputs", "in_2": "inputs", "out_1": "outputs"}
    tmp = Dataset()
    tmp.set_from_array(data, variables, sizes, groups)
    return tmp


@pytest.fixture
def file_dataset(tmp_wd):
    inputs = arange(50).reshape(10, 5)
    outputs = arange(20).reshape(10, 2)
    data = concatenate([inputs, outputs], axis=1)
    variables = ["in_1", "in_2", "out_1"]
    sizes = {"in_1": 2, "in_2": 3, "out_1": 2}
    groups = {"in_1": "inputs", "in_2": "inputs", "out_1": "outputs"}
    filename = "dataset.txt"
    savetxt(filename, data, delimiter=",")
    return filename, data, variables, sizes, groups


def test_is_empty():
    dataset = Dataset()
    assert dataset.is_empty()
    assert not dataset
    dataset.set_from_array(array([[1, 2]]))
    assert not dataset.is_empty()
    assert dataset


def test_str(io_dataset):
    assert "inputs" in str(io_dataset)
    expected = MultiLineString()
    expected.add("Dataset")
    expected.indent()
    expected.add("Number of samples: 10")
    expected.add("Number of variables: 3")
    expected.add("Variables names and sizes by group:")
    expected.indent()
    expected.add("inputs: in_1 (2), in_2 (3)")
    expected.add("outputs: out_1 (2)")
    expected.dedent()
    expected.add("Number of dimensions (total = 7) by group:")
    expected.indent()
    expected.add("inputs: 5")
    expected.add("outputs: 2")
    assert str(expected) == str(io_dataset)


def test_set_from_raw_array(data):
    dataset = Dataset()
    dataset.set_from_array(data)
    assert dataset.get_names("parameters") == ["x_0", "x_1", "x_2"]
    assert dataset.sizes["x_0"] == 1
    assert dataset.sizes["x_1"] == 1
    assert dataset.sizes["x_2"] == 1
    assert dataset.data["parameters"].shape[0] == 10
    assert dataset.data["parameters"].shape[1] == 3
    dataset = Dataset()
    with pytest.raises(TypeError):
        dataset.set_from_array(data, variables=["x", "y", "z"], groups="outputs")
    dataset.set_from_array(data, variables=["x", "y", "z"], groups={"z": "outputs"})
    assert "outputs" in dataset.groups
    assert "parameters" in dataset.groups
    assert dataset.data["parameters"].shape[1] == 2
    assert dataset.data["outputs"].shape[1] == 1


def test_set_metadata(data):
    dataset = Dataset()
    dataset.set_metadata("toto", 5)
    assert dataset.metadata["toto"] == 5


def test_remove(io_dataset):
    assert io_dataset.n_samples == 10
    io_dataset.remove([0, 3])
    assert io_dataset.data["inputs"].shape == (8, 5)
    assert io_dataset.data["outputs"].shape == (8, 2)
    assert io_dataset.n_samples == 8


def test_logical_operators():
    assert set(LOGICAL_OPERATORS.keys()) == {"==", "<", "<=", ">", ">=", "!="}
    assert LOGICAL_OPERATORS["=="](1, 1)
    assert not LOGICAL_OPERATORS["=="](1, 2)
    assert LOGICAL_OPERATORS["!="](1, 2)
    assert not LOGICAL_OPERATORS["!="](1, 1)
    assert LOGICAL_OPERATORS["<"](1, 2)
    assert not LOGICAL_OPERATORS["<"](2, 2)
    assert not LOGICAL_OPERATORS["<"](1, 0)
    assert LOGICAL_OPERATORS["<="](1, 2)
    assert LOGICAL_OPERATORS["<="](2, 2)
    assert not LOGICAL_OPERATORS["<="](1, 0)
    assert not LOGICAL_OPERATORS[">"](1, 2)
    assert not LOGICAL_OPERATORS[">"](2, 2)
    assert LOGICAL_OPERATORS[">"](1, 0)
    assert not LOGICAL_OPERATORS[">="](1, 2)
    assert LOGICAL_OPERATORS[">="](2, 2)
    assert LOGICAL_OPERATORS[">="](1, 0)


def test_find_and_compare(io_dataset):
    data = io_dataset

    with pytest.raises(ValueError):
        comparison = data.compare(0, "==", 0)

    expected = r"\+ is not a logical operator: use either '==', '<', '<=', '>' or '>='"
    with pytest.raises(ValueError, match=expected):
        comparison = data.compare("in_1", "+", 0)

    comparison = data.compare("in_1", "==", 0)
    assert_equal(comparison, array([True] + [False] * 9))

    comparison = data.compare("in_1", "==", 0)
    indices = data.find(comparison)
    assert indices == [0]

    comparison = data.compare("in_1", "==", 10)
    indices = data.find(comparison)
    assert indices == [2]

    comparison = data.compare("in_1", "==", 10) & data.compare("in_2", "==", 12)
    indices = data.find(comparison)
    assert indices == [2]

    comparison = data.compare("in_1", "==", 0) | data.compare("in_1", "==", 10)
    indices = data.find(comparison)
    assert indices == [0, 2]


def test_isnan():
    dataset = Dataset()
    dataset.set_from_array(array([[1.0, 2.0], [3.0, nan], [5.0, 6.0], [nan, 8.0]]))
    is_nan = dataset.is_nan()
    assert_equal(is_nan, [False, True, False, True])
    assert len(dataset) == 4
    dataset.remove(is_nan)
    assert len(dataset) == 2


def test_find_and_remove(io_dataset):
    data = io_dataset
    assert 0 in data["in_1"][:, 0]
    assert len(data) == 10
    data.remove(data.compare("in_1", "==", 0))
    assert 0 not in data["in_1"][:, 0]
    assert len(data) == 9


def test_is_variable(dataset, io_dataset):
    assert dataset.is_variable("var_1")
    assert dataset.is_variable("var_2")
    assert not dataset.is_variable("dummy")
    assert io_dataset.is_variable("in_1")
    assert io_dataset.is_variable("in_2")
    assert io_dataset.is_variable("out_1")
    assert not io_dataset.is_variable("dummy")


def test_is_group(dataset, io_dataset):
    assert dataset.is_group("parameters")
    assert not dataset.is_group("inputs")
    assert io_dataset.is_group("inputs")


def test_get_names(dataset, io_dataset):
    assert dataset.get_names("parameters") == ["var_1", "var_2"]
    assert io_dataset.get_names("inputs") == ["in_1", "in_2"]
    assert io_dataset.get_names("outputs") == ["out_1"]


def test_groups(dataset, io_dataset):
    assert dataset.groups == ["parameters"]
    assert io_dataset.groups == ["inputs", "outputs"]


def test_add_variable(dataset, ungroup_dataset):
    dataset.add_variable("x", arange(10).reshape(10, 1))
    assert "x" in dataset.variables
    assert dataset.sizes["x"] == 1
    assert dataset._groups["x"] == "parameters"
    assert dataset.data["parameters"].shape[0] == 10
    assert dataset.data["parameters"].shape[1] == 4
    assert "x" in dataset._cached_inputs
    dataset.add_variable("y", arange(20).reshape(10, 2), group="new_group")
    assert "y" in dataset.variables
    assert "new_group" in dataset.groups
    assert dataset.get_names("new_group") == ["y"]
    assert dataset.sizes["y"] == 2
    assert dataset._groups["y"] == "new_group"
    assert dataset.data["parameters"].shape[0] == 10
    assert dataset.data["parameters"].shape[1] == 4
    assert dataset.data["new_group"].shape[0] == 10
    assert dataset.data["new_group"].shape[1] == 2

    dataset.add_variable("z", arange(30).reshape(10, 3), cache_as_input=False)
    assert "z" in dataset._cached_outputs

    ungroup_dataset.add_variable("x", arange(10).reshape(10, 1))
    assert "x" in ungroup_dataset.data

    with pytest.raises(TypeError):
        ungroup_dataset.add_variable(1, arange(10).reshape(10, 1))

    with pytest.raises(ValueError):
        ungroup_dataset.add_variable("x", arange(10).reshape(10, 1))

    with pytest.raises(TypeError):
        ungroup_dataset.add_variable("z", arange(10))


def test_add_group(dataset, ungroup_dataset):
    dataset.add_group("grp1", arange(30).reshape(10, 3))
    assert "grp1" in dataset.groups
    assert dataset.data["grp1"].shape[0] == 10
    assert dataset.data["grp1"].shape[1] == 3
    assert "grp1_0" in dataset.variables
    assert "grp1_1" in dataset.variables
    assert "grp1_2" in dataset.variables
    assert "grp1_0" in dataset._cached_inputs
    assert "grp1_1" in dataset._cached_inputs
    assert "grp1_2" in dataset._cached_inputs
    dataset.add_group("grp2", arange(30).reshape(10, 3), ["x", "y"], {"x": 1, "y": 2})
    assert "grp2" in dataset.groups
    assert dataset.data["grp2"].shape[0] == 10
    assert dataset.data["grp2"].shape[1] == 3
    assert "x" in dataset.variables
    assert "y" in dataset.variables
    assert dataset.sizes["x"] == 1
    assert dataset.sizes["y"] == 2
    assert "x" in dataset.get_names("grp2")
    assert "y" in dataset.get_names("grp2")
    dataset.add_group("functions", arange(30).reshape(10, 3))
    assert "functions" in dataset.groups
    assert dataset.data["functions"].shape[0] == 10
    assert dataset.data["functions"].shape[1] == 3
    assert "func_0" in dataset.variables
    assert "func_1" in dataset.variables
    assert "func_2" in dataset.variables
    dataset.add_group("grp3", arange(30).reshape(10, 3), pattern="x")
    assert "grp3" in dataset.groups
    assert dataset.data["grp3"].shape[0] == 10
    assert dataset.data["grp3"].shape[1] == 3
    assert "x_0" in dataset.variables
    assert "x_1" in dataset.variables
    assert "x_2" in dataset.variables

    dataset.add_group("grp4", arange(30).reshape(10, 3), cache_as_input=False)
    assert "grp4_0" in dataset._cached_outputs
    assert "grp4_1" in dataset._cached_outputs
    assert "grp4_2" in dataset._cached_outputs

    ungroup_dataset.add_group("grp1", arange(30).reshape(10, 3))
    assert "grp1_0" in ungroup_dataset.data
    assert "grp1_1" in ungroup_dataset.data
    assert "grp1_2" in ungroup_dataset.data

    with pytest.raises(TypeError):
        ungroup_dataset.add_group(1, arange(30).reshape(10, 3))

    with pytest.raises(TypeError):
        ungroup_dataset.add_group("grp", arange(30).reshape(10, 3), ["x", ["y1", "y2"]])

    with pytest.raises(TypeError):
        ungroup_dataset.add_group("grp", arange(30).reshape(10, 3), ["x", "y"], 10)

    with pytest.raises(TypeError):
        ungroup_dataset.add_group(
            "grp", arange(30).reshape(10, 3), ["x", "y"], {"x": "a", "y": "b"}
        )

    with pytest.raises(ValueError):
        ungroup_dataset.add_group("grp1", arange(10).reshape(10, 1))

    with pytest.raises(ValueError):
        ungroup_dataset.add_group("grp10", arange(5).reshape(5, 1))


def test_export_to_dataframe(dataset):
    """Check the dataframe resulting from a dataset export."""
    variables = ["i1", "o2", "o1", "i2"]
    sizes = {"i1": 1, "i2": 2, "o1": 1, "o2": 2}
    dataset = Dataset()
    dataset.set_from_array(arange(12).reshape(2, 6), variables, sizes)
    df = dataset.export_to_dataframe()
    for column, expected_column in zip(
        df.columns.values,
        [
            ("parameters", "i1", "0"),
            ("parameters", "i2", "0"),
            ("parameters", "i2", "1"),
            ("parameters", "o1", "0"),
            ("parameters", "o2", "0"),
            ("parameters", "o2", "1"),
        ],
    ):
        assert column == expected_column

    assert_equal(df.values, array([[0, 4, 5, 3, 1, 2], [6, 10, 11, 9, 7, 8]]))


def test_get_columns_names():
    """Check the default names of the dataset columns."""
    dataset = Dataset()
    dataset.set_from_array(array([[1.0], [1.0]]))
    assert dataset.get_column_names() == ["x_0"]


def test_get_data_by_group(io_dataset, ungroup_dataset):
    inputs = io_dataset.get_data_by_group("inputs")
    assert inputs.shape[0] == 10
    assert inputs.shape[1] == 5
    inputs = io_dataset.get_data_by_group("inputs", True)
    assert "in_1" in inputs
    assert "in_2" in inputs

    data = ungroup_dataset.get_data_by_group("parameters")
    assert data.shape[0] == 10
    assert data.shape[1] == 3
    data = ungroup_dataset.get_data_by_group("parameters", True)
    assert "var_1" in data
    assert "var_2" in data

    with pytest.raises(ValueError):
        io_dataset.get_data_by_group("not_a_group")


def test_get_data_by_names(io_dataset):
    data = io_dataset.get_data_by_names(["in_1", "out_1"])
    assert "in_1" in data
    assert "out_1" in data
    data = io_dataset.get_data_by_names("in_1")
    assert "in_1" in data
    data = io_dataset.get_data_by_names("in_1", False)
    assert data.shape[0] == 10
    assert data.shape[1] == 2


@pytest.mark.parametrize(
    "by_group,expected_keys",
    [(False, ["in_1", "in_2", "out_1"]), (True, ["inputs", "outputs"])],
)
def test_get_all_data(io_dataset, by_group, expected_keys):
    """Check that get_all_data returns a dictionary correctly indexed."""
    data = io_dataset.get_all_data(by_group=by_group, as_dict=True)
    for key in expected_keys:
        assert key in data


def test_n_variables(io_dataset):
    assert io_dataset.n_variables == 3
    assert io_dataset.n_variables_by_group("outputs") == 1


def test_n_samples(io_dataset):
    assert io_dataset.n_samples == 10


@pytest.mark.parametrize("cache_type", ["MemoryFullCache", "HDF5Cache"])
@pytest.mark.parametrize("inputs", [None, ["var_1"]])
@pytest.mark.parametrize("outputs", [None, ["var_2"]])
def test_export_dataset_to_cache(dataset, tmp_wd, cache_type, inputs, outputs):
    """Check that a dataset is correctly exported to a cache."""
    input_names = inputs or ["var_1", "var_2"]
    output_names = outputs or []
    cache = dataset.export_to_cache(
        cache_type=cache_type,
        cache_hdf_file="cache.hdf5",
        inputs=inputs,
        outputs=outputs,
    )
    assert len(cache) == 10
    for name, value in cache.last_entry.inputs.items():
        assert (dataset[9][name] == value).all()

    for input_name in cache.input_names:
        assert input_name in input_names

    for output_name in cache.output_names:
        assert output_name in output_names


@pytest.mark.parametrize(
    "filename,header,expected_names",
    [
        ("data_without_header.csv", False, ["x_0", "x_1", "x_2", "x_3"]),
        ("data_with_header.csv", True, ["a", "b", "c", "d"]),
    ],
)
def test_dataset_from_file(filename, header, expected_names):
    """Check the construction of a dataset from a CSV file."""
    file_path = Path(__file__).parent / "data" / "dataset" / filename
    dataset = Dataset()
    dataset.set_from_file(file_path, header=header)
    assert_equal(dataset.data[dataset.PARAMETER_GROUP], [[1, 2, 3, 4], [-1, -2, -3, 4]])
    assert dataset.columns_names == expected_names


@pytest.fixture(scope="module")
def x_dataset():
    dataset = Dataset()
    dataset.add_variable("x", array([[1.0], [-1.0]]))
    dataset.add_variable("y", array([[2.0, 3.0], [-2.0, -3.0]]))
    return dataset


@pytest.mark.parametrize(
    "item,expected",
    [
        (0, {"x": array([1.0]), "y": array([2.0, 3.0])}),
        ([0], {"x": array([[1.0]]), "y": array([[2.0, 3.0]])}),
        ([1, 0], {"x": array([[-1.0], [1.0]]), "y": array([[-2.0, -3.0], [2.0, 3.0]])}),
        (
            slice(0, 2),
            {"x": array([[1.0], [-1.0]]), "y": array([[2.0, 3.0], [-2.0, -3.0]])},
        ),
        (
            Ellipsis,
            {"x": array([[1.0], [-1.0]]), "y": array([[2.0, 3.0], [-2.0, -3.0]])},
        ),
        ("x", array([[1.0], [-1.0]])),
        (["x"], {"x": array([[1.0], [-1.0]])}),
        (
            ["x", "y"],
            {"x": array([[1.0], [-1.0]]), "y": array([[2.0, 3.0], [-2.0, -3.0]])},
        ),
        ((0, "x"), array([1.0])),
        (([0], "x"), array([[1.0]])),
        ((0, ["x"]), {"x": array([1.0])}),
        ((0, ["x", "y"]), {"x": array([1.0]), "y": array([2.0, 3.0])}),
        (([0], ["x"]), {"x": array([[1.0]])}),
        (
            ([1, 0], ["x", "y"]),
            {"x": array([[-1.0], [1.0]]), "y": array([[-2.0, -3.0], [2.0, 3.0]])},
        ),
        (
            (slice(0, 2), ["x", "y"]),
            {"x": array([[1.0], [-1.0]]), "y": array([[2.0, 3.0], [-2.0, -3.0]])},
        ),
        (
            (Ellipsis, ["x", "y"]),
            {"x": array([[1.0], [-1.0]]), "y": array([[2.0, 3.0], [-2.0, -3.0]])},
        ),
    ],
)
def test_getitem(x_dataset, item, expected):
    """Check the access to an item."""
    assert_equal(x_dataset[item], expected)


@pytest.mark.parametrize(
    "item,error,msg",
    [
        ("dummy", KeyError, "There is not variable named 'dummy' in the dataset."),
        (1000, KeyError, "Entries must be integers between -2 and 1; got 1000."),
        ((0, 1), TypeError, Dataset._Dataset__GETITEM_ERROR_MESSAGE),
        ([1, "x"], TypeError, Dataset._Dataset__GETITEM_ERROR_MESSAGE),
        (1.0, TypeError, Dataset._Dataset__GETITEM_ERROR_MESSAGE),
    ],
)
def test_getitem_raising_error(x_dataset, item, error, msg):
    """Check that accessing an unknown item raises an error."""
    with pytest.raises(error, match=re.escape(msg)):
        x_dataset[item]


def test_plot(dataset, tmp_wd):
    post = dataset.plot("ScatterMatrix", show=False, save=True, file_path="scatter")
    assert len(post.output_files) > 0
    assert "ScatterMatrix" in dataset.get_available_plots()


def test_export_to_dataset():
    disc = AnalyticDiscipline({"obj": "x1+x2", "cstr": "x1-x2"})
    disc.set_cache_policy(disc.MEMORY_FULL_CACHE)
    d_s = DesignSpace()
    d_s.add_variable("x1", 1, "float", 0, 1, 0.5)
    d_s.add_variable("x2", 1, "float", 0, 1, 0.5)
    scn = DOEScenario([disc], "DisciplinaryOpt", "obj", d_s)
    scn.execute({"algo": "lhs", "n_samples": 10})
    dataset = disc.cache.export_to_dataset()
    assert "x1" in dataset.get_names(dataset.INPUT_GROUP)
    assert "x2" in dataset.get_names(dataset.INPUT_GROUP)
    assert "obj" in dataset.get_names(dataset.OUTPUT_GROUP)
    assert "cstr" in dataset.get_names(dataset.OUTPUT_GROUP)
    dataset = disc.cache.export_to_dataset(input_names=["x1"])
    assert "x1" in dataset.get_names(dataset.INPUT_GROUP)
    assert "x2" not in dataset.get_names(dataset.INPUT_GROUP)
    assert "obj" in dataset.get_names(dataset.OUTPUT_GROUP)
    assert "cstr" in dataset.get_names(dataset.OUTPUT_GROUP)
    dataset = disc.cache.export_to_dataset(output_names=["obj"])
    assert "x1" in dataset.get_names(dataset.INPUT_GROUP)
    assert "x2" in dataset.get_names(dataset.INPUT_GROUP)
    assert "obj" in dataset.get_names(dataset.OUTPUT_GROUP)
    assert "cstr" not in dataset.get_names(dataset.OUTPUT_GROUP)


def test_row_names(io_dataset):
    """Check row_names property and setter."""
    assert io_dataset.row_names == [str(val) for val in range(len(io_dataset))]
    io_dataset.row_names = [f"sample_{index}" for index in range(len(io_dataset))]
    assert io_dataset.row_names == io_dataset.row_names


@pytest.mark.parametrize(
    "excluded_variables", [None, ["in_1"], ["in_2"], ["out_1"], ["in_1", "out_1"]]
)
@pytest.mark.parametrize(
    "excluded_groups", [None, ["inputs"], ["outputs"], ["inputs", "outputs"]]
)
def test_get_normalized_dataset(io_dataset, excluded_groups, excluded_variables):
    """Check the normalization of a dataset.

    Args:
        io_dataset (Dataset): The original dataset.
        excluded_groups (Optional[Sequence[str]]): The groups not to be normalized.
        excluded_variables (Optional[Sequence[str]]): The names not to be normalized.
    """
    excluded_groups = excluded_groups or []
    excluded_variables = excluded_variables or []
    for group in excluded_groups:
        excluded_variables += io_dataset._names[group]

    dataset = io_dataset.get_normalized_dataset(excluded_variables, excluded_groups)
    all_data = dataset.get_all_data(by_group=False, as_dict=True)

    for name, data in all_data.items():
        if name in excluded_variables:
            assert not allclose(np.min(data, 0), zeros(data.shape[1]))
            assert not allclose(np.max(data, 0), ones(data.shape[1]))
        else:
            assert allclose(np.min(data, 0), zeros(data.shape[1]))
            assert allclose(np.max(data, 0), ones(data.shape[1]))


def test_malformed_groups():
    """Check that passing malformed groups raises an error."""
    dataset = Dataset()
    with pytest.raises(
        TypeError,
        match="groups must be a dictionary of the form {variable_name: group_name}.",
    ):
        dataset.set_from_array(array([[1]]), variables=["x"], groups={"x": 1})


def test_malformed_variables():
    """Check that passing malformed variables raises an error."""
    dataset = Dataset()
    with pytest.raises(
        TypeError,
        match="variables must be a list of string variable names.",
    ):
        dataset.set_from_array(array([[1]]), variables=1)


def test_malformed_sizes():
    """Check that passing malformed sizes raises an error."""
    dataset = Dataset()
    with pytest.raises(
        ValueError,
        match=(
            re.escape(
                "The sum of the variable sizes (2) "
                "must be equal to the data dimension (1)."
            )
        ),
    ):
        dataset.set_from_array(array([[1]]), variables=["x"], sizes={"x": 2})


@pytest.mark.parametrize("by_group", [False, True])
def test_rename_variable(by_group):
    """Check the renaming of a variable."""
    data = array([[1]])
    dataset = Dataset(by_group=by_group)
    dataset.add_variable("x", data)
    dataset.add_variable("a", data, cache_as_input=False)
    dataset.rename_variable("x", "y")
    dataset.rename_variable("a", "b")
    assert dataset["y"] == data
    assert dataset.sizes["y"] == 1
    assert dataset._positions["y"] == [0, 0]
    assert "x" not in dataset.sizes
    if by_group:
        assert "x" not in dataset.data[dataset.DEFAULT_GROUP]
    else:
        assert "x" not in dataset.data
    assert "x" not in dataset._positions
    assert "x" not in dataset._names[dataset.DEFAULT_GROUP]
    assert "y" in dataset._names[dataset.DEFAULT_GROUP]
    assert "x" not in dataset._cached_inputs
    assert "y" in dataset._cached_inputs
    assert "a" not in dataset._cached_outputs
    assert "b" in dataset._cached_outputs


@pytest.mark.parametrize("by_group", [False, True])
def test_transform_variable(by_group):
    """Check the transformation of a variable."""
    dataset = Dataset(by_group=by_group)
    dataset.add_variable("x", array([[1]]))
    dataset.transform_variable("x", lambda x: -x)
    assert dataset["x"] == array([[-1]])
