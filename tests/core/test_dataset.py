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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
"""Test the dataset module."""
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import division, unicode_literals

from os.path import join

import numpy as np
import pytest
from numpy import allclose, arange, array, concatenate, nan, ones, savetxt, zeros

from gemseo.algos.design_space import DesignSpace
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.dataset import LOGICAL_OPERATORS, Dataset
from gemseo.core.doe_scenario import DOEScenario
from gemseo.utils.string_tools import MultiLineString


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
def file_dataset(tmp_path):
    inputs = arange(50).reshape(10, 5)
    outputs = arange(20).reshape(10, 2)
    data = concatenate([inputs, outputs], axis=1)
    variables = ["in_1", "in_2", "out_1"]
    sizes = {"in_1": 2, "in_2": 3, "out_1": 2}
    groups = {"in_1": "inputs", "in_2": "inputs", "out_1": "outputs"}
    filename = join(str(tmp_path), "dataset.txt")
    savetxt(filename, data, delimiter=",")
    return filename, data, variables, sizes, groups


@pytest.fixture
def header_file_dataset(tmp_path):
    inputs = arange(50).reshape(10, 5)
    outputs = arange(20).reshape(10, 2)
    data = concatenate([inputs, outputs], axis=1)
    filename = join(str(tmp_path), "dataset_with_header.txt")
    header = ",".join(["a", "b", "c", "d", "e", "f", "g"])
    savetxt(filename, data, delimiter=",", header=header, comments="")
    return filename, data


@pytest.fixture
def txt_io_dataset():
    inpt = array([["1", "2"], ["3", "4"]])
    out = array([["a"], ["b"]])
    data = concatenate([inpt, out], axis=1)
    variables = ["var_1", "var_2"]
    sizes = {"var_1": 2, "var_2": 1}
    groups = {"var_1": "input", "var_2": "output"}
    dataset = Dataset()
    dataset.set_from_array(data, variables, sizes, groups)
    return dataset


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
    assert set(LOGICAL_OPERATORS.keys()) == set(["==", "<", "<=", ">", ">=", "!="])
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
    assert (comparison == array([True] + [False] * 9)).all()

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
    assert (is_nan == array([False, True, False, True])).all()
    assert len(dataset) == 4
    dataset.remove(is_nan)
    assert len(dataset) == 2


def test_find_and_remove(io_dataset):
    data = io_dataset
    assert 0 in data["in_1"]["in_1"][:, 0]
    assert len(data) == 10
    data.remove(data.compare("in_1", "==", 0))
    assert 0 not in data["in_1"]["in_1"][:, 0]
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
    assert len(dataset.export_to_dataframe()) == len(dataset)


def test_get_columns_names():
    dataset = Dataset()
    dataset.set_from_array(array([[1.0], [1.0]]))
    assert dataset._get_columns_names() == ["x_0"]


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


def test_get_all_data(io_dataset):
    data = io_dataset.get_all_data(by_group=False, as_dict=True)
    assert "in_1" in data
    assert "in_2" in data
    assert "out_1" in data


def test_n_variables(io_dataset):
    assert io_dataset.n_variables == 3
    assert io_dataset.n_variables_by_group("outputs") == 1


def test_n_samples(io_dataset):
    assert io_dataset.n_samples == 10


# def test_dataset_with_txt_array(txt_io_dataset):
#     assert_allclose(txt_io_dataset.data['output'], array([[0.], [1.]]))
#
#
# def test_export_io_dataset_to_cache(io_dataset):
#     cache = io_dataset.export_to_cache()
#     first_data = cache.get_data(1)
#     assert_allclose(first_data['inputs']['in_1'], array([0, 1]))
#     assert_allclose(first_data['inputs']['in_2'], array([2, 3, 4]))
#     assert_allclose(first_data['outputs']['out_1'], array([0, 1]))
#     assert_allclose(cache.get_last_cached_inputs()['in_1'],
#                     array([45, 46]))
#     assert_allclose(cache.get_last_cached_inputs()['in_2'],
#                     array([47, 48, 49]))
#     assert_allclose(cache.get_last_cached_outputs()['out_1'],
#                     array([18, 19]))


def test_export_dataset_to_cache(dataset, tmp_path):
    cache = dataset.export_to_cache()
    assert cache.name == "my_dataset"
    assert cache.get_length() == 10
    for name, value in cache.get_last_cached_inputs().items():
        assert (dataset[9][name] == value).all()

    filename = join(str(tmp_path), "cache.hdf5")
    cache = dataset.export_to_cache(
        cache_type=dataset.HDF5_CACHE, cache_hdf_file=filename
    )
    assert cache.get_length() == 10
    for name, value in cache.get_last_cached_inputs().items():
        assert (dataset[9][name] == value).all()


#
# def test_dataset_with_file(file_dataset, header_file_dataset):
#     grp = Dataset.DEFAULT_GROUP
#     filename, data, variables, sizes, groups = file_dataset
#     dataset = Dataset()
#     dataset.set_from_file(filename, variables, sizes, header=False)
#     assert_allclose(dataset.data[grp], data)
#     assert dataset.get_names(grp) == ['in_1', 'in_2', 'out_1']
#     dataset = Dataset()
#     dataset.set_from_file(filename, header=False)
#     assert dataset.get_names(grp) == ['x_' + str(i) for i in range(7)]
#     filename, data = header_file_dataset
#     dataset = Dataset()
#     dataset.set_from_file(filename, header=True)
#     assert dataset.get_names(grp) == ['a', 'b', 'c', 'd', 'e', 'f', 'g']
#     assert_allclose(dataset.data[grp], data)


def test_getitem(dataset, data):
    with pytest.raises(TypeError):
        # item is a bad item
        dataset[{}]
    with pytest.raises(TypeError):
        # item is a list with bad elements
        dataset[[{}]]
    with pytest.raises(TypeError):
        # item is a tuple whose first element is a list with bad elements
        dataset[([{}], "var_1")]
    with pytest.raises(TypeError):
        # item is a tuple whose first element is a bad element
        dataset[({}, "var_1")]
    with pytest.raises(TypeError):
        # item is a tuple whose second element is a list with bad elements
        dataset[(1, [{}])]
    with pytest.raises(TypeError):
        # item is a tuple whose second element is a bad element
        dataset[(1, {})]
    with pytest.raises(ValueError):
        dataset["dummy"]
    with pytest.raises(ValueError):
        dataset[1000]
    res = dataset["var_1"]
    assert allclose(res["var_1"], data[:, 0:1])
    res = dataset["var_2"]
    assert allclose(res["var_2"], data[:, 1:3])
    res = dataset[["var_1", "var_2"]]
    assert allclose(res["var_1"], data[:, 0:1])
    assert allclose(res["var_2"], data[:, 1:3])
    res = dataset[2]
    assert allclose(res["var_1"], data[2:3, 0:1])
    assert allclose(res["var_2"], data[2:3, 1:3])
    res = dataset[[2, 3]]
    assert allclose(res["var_1"], data[2:4, 0:1])
    assert allclose(res["var_2"], data[2:4, 1:3])
    res = dataset[2:4]
    assert allclose(res["var_1"], data[2:4, 0:1])
    assert allclose(res["var_2"], data[2:4, 1:3])
    res = dataset[(2, "var_1")]
    assert allclose(res["var_1"], data[2:3, 0:1])
    res = dataset[(2, ["var_1", "var_2"])]
    assert allclose(res["var_1"], data[2:3, 0:1])
    assert allclose(res["var_2"], data[2:3, 1:3])
    res = dataset[([2, 3], "var_1")]
    assert allclose(res["var_1"], data[2:4, 0:1])
    res = dataset[([2, 3], ["var_1", "var_2"])]
    assert allclose(res["var_1"], data[2:4, 0:1])
    assert allclose(res["var_2"], data[2:4, 1:3])
    res = dataset[(slice(2, 4), ["var_1", "var_2"])]
    assert allclose(res["var_1"], data[2:4, 0:1])
    assert allclose(res["var_2"], data[2:4, 1:3])


def test_plot(dataset, tmp_path):
    fpath = tmp_path / "scatter"
    post = dataset.plot("ScatterMatrix", show=False, save=True, file_path=fpath)
    assert len(post.output_files) > 0
    assert "ScatterMatrix" in dataset.get_available_plots()


def test_export_to_dataset():
    disc = AnalyticDiscipline(expressions_dict={"obj": "x1+x2", "cstr": "x1-x2"})
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
    dataset = disc.cache.export_to_dataset(inputs_names=["x1"])
    assert "x1" in dataset.get_names(dataset.INPUT_GROUP)
    assert "x2" not in dataset.get_names(dataset.INPUT_GROUP)
    assert "obj" in dataset.get_names(dataset.OUTPUT_GROUP)
    assert "cstr" in dataset.get_names(dataset.OUTPUT_GROUP)
    dataset = disc.cache.export_to_dataset(outputs_names=["obj"])
    assert "x1" in dataset.get_names(dataset.INPUT_GROUP)
    assert "x2" in dataset.get_names(dataset.INPUT_GROUP)
    assert "obj" in dataset.get_names(dataset.OUTPUT_GROUP)
    assert "cstr" not in dataset.get_names(dataset.OUTPUT_GROUP)


def test_row_names(io_dataset):
    """Check row_names property and setter."""
    assert io_dataset.row_names == [str(val) for val in range(len(io_dataset))]
    io_dataset.row_names = [
        "sample_{}".format(index) for index in range(len(io_dataset))
    ]
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
