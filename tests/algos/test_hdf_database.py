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
from __future__ import annotations

import h5py
import pytest
from numpy import array
from numpy import bytes_
from numpy import ndarray

from gemseo.algos._hdf_database import HDFDatabase
from gemseo.algos.database import HashableNdarray


@pytest.fixture
def h5_file(tmp_wd):
    return h5py.File("test.h5", "w")


def test_create_hdf_input_dataset(h5_file) -> None:
    """Test that design variable values are correctly added to the hdf5 group of design
    variables."""
    hdf_database = HDFDatabase()

    design_vars_grp = h5_file.require_group("x")

    input_val_1 = HashableNdarray(array([0, 1, 2]))
    hdf_database._HDFDatabase__add_hdf_input_dataset(0, design_vars_grp, input_val_1)
    assert array(design_vars_grp["0"]) == pytest.approx(input_val_1.unwrap())

    hdf_database._HDFDatabase__add_hdf_input_dataset(1, design_vars_grp, input_val_1)
    assert array(design_vars_grp["1"]) == pytest.approx(input_val_1.unwrap())

    input_val_2 = HashableNdarray(array([3, 4, 5]))
    with pytest.raises(ValueError):
        hdf_database._HDFDatabase__add_hdf_input_dataset(
            0, design_vars_grp, input_val_2
        )


def test_add_hdf_name_output(h5_file) -> None:
    """Test that output names are correctly added to the hdf5 group of output names."""
    hdf_database = HDFDatabase()

    keys_group = h5_file.require_group("k")

    hdf_database._HDFDatabase__add_hdf_name_output(0, keys_group, ["f1"])
    assert array(keys_group["0"]) == array(["f1"], dtype=bytes_)

    hdf_database._HDFDatabase__add_hdf_name_output(0, keys_group, ["f2", "f3", "f4"])
    assert (
        array(keys_group["0"]) == array(["f1", "f2", "f3", "f4"], dtype=bytes_)
    ).all()

    hdf_database._HDFDatabase__add_hdf_name_output(1, keys_group, ["f2", "f3", "f4"])
    assert (array(keys_group["1"]) == array(["f2", "f3", "f4"], dtype=bytes_)).all()

    hdf_database._HDFDatabase__add_hdf_name_output(1, keys_group, ["@-y_1"])
    assert (
        array(keys_group["1"]) == array(["f2", "f3", "f4", "@-y_1"], dtype=bytes_)
    ).all()


def test_add_hdf_scalar_output(h5_file) -> None:
    """Test that scalar values are correctly added to the group of output values."""
    hdf_database = HDFDatabase()

    values_group = h5_file.require_group("v")

    hdf_database._HDFDatabase__add_hdf_scalar_output(0, values_group, [10])
    assert array(values_group["0"]) == pytest.approx(array([10]))

    hdf_database._HDFDatabase__add_hdf_scalar_output(0, values_group, [20])
    assert array(values_group["0"]) == pytest.approx(array([10, 20]))

    hdf_database._HDFDatabase__add_hdf_scalar_output(0, values_group, [30, 40, 50, 60])
    assert array(values_group["0"]) == pytest.approx(array([10, 20, 30, 40, 50, 60]))

    hdf_database._HDFDatabase__add_hdf_scalar_output(1, values_group, [100, 200])
    assert array(values_group["1"]) == pytest.approx(array([100, 200]))


def test_add_hdf_vector_output(h5_file) -> None:
    """Test that a vector (array and/or list) of outputs is correctly added to the group
    of output values."""
    hdf_database = HDFDatabase()

    values_group = h5_file.require_group("v")

    hdf_database._HDFDatabase__add_hdf_vector_output(0, 0, values_group, [10, 20, 30])
    assert array(values_group["arr_0"]["0"]) == pytest.approx(array([10, 20, 30]))

    hdf_database._HDFDatabase__add_hdf_vector_output(
        0, 1, values_group, array([100, 200])
    )
    assert array(values_group["arr_0"]["1"]) == pytest.approx(array([100, 200]))

    hdf_database._HDFDatabase__add_hdf_vector_output(
        1, 2, values_group, array([[0.1, 0.2, 0.3, 0.4]])
    )
    assert array(values_group["arr_1"]["2"]) == pytest.approx(
        array([[0.1, 0.2, 0.3, 0.4]])
    )

    with pytest.raises(ValueError):
        hdf_database._HDFDatabase__add_hdf_vector_output(1, 2, values_group, [1, 2])


def test_add_hdf_output_dataset(h5_file) -> None:
    """Test that output datasets are correctly added to the hdf groups of output."""
    hdf_database = HDFDatabase()

    values_group = h5_file.require_group("v")
    keys_group = h5_file.require_group("k")

    values = {"f": 10, "g": array([1, 2]), "Iter": [3], "@f": array([[1, 2, 3]])}
    hdf_database._HDFDatabase__add_hdf_output_dataset(
        10, keys_group, values_group, values
    )
    assert sorted(keys_group["10"]) == sorted(array(list(values.keys()), dtype=bytes_))
    assert array(values_group["10"]) == pytest.approx(array([10]))
    dataset_names = values_group["arr_10"].keys()
    for key, dataset_name in zip(sorted(values.keys()), dataset_names):
        val = values[key]
        if isinstance(val, (ndarray, list)):
            assert array(values_group["arr_10"][dataset_name]) == pytest.approx(
                array(val)
            )

    values = {
        "@j": array([[1, 2, 3]]),
        "Iter": 1,
        "i": array([1, 2]),
        "k": 99,
        "l": 100,
    }
    hdf_database._HDFDatabase__add_hdf_output_dataset(
        100, keys_group, values_group, values
    )
    assert sorted(keys_group["100"]) == sorted(array(list(values.keys()), dtype=bytes_))
    assert array(values_group["100"]) == pytest.approx(array([1, 99, 100]))
    assert array(values_group["arr_100"]["0"]) == pytest.approx(array([[1, 2, 3]]))
    assert array(values_group["arr_100"]["2"]) == pytest.approx(array([1, 2]))


def test_get_missing_hdf_output_dataset(h5_file) -> None:
    """Test that missing values in the hdf  output datasets are correctly found."""
    hdf_database = HDFDatabase()

    values_group = h5_file.require_group("v")
    keys_group = h5_file.require_group("k")

    values = {"f": 0.1, "g": array([1, 2])}
    hdf_database._HDFDatabase__add_hdf_output_dataset(
        10, keys_group, values_group, values
    )

    with pytest.raises(ValueError):
        hdf_database._HDFDatabase__get_missing_hdf_output_dataset(0, keys_group, values)

    values = {"f": 0.1, "g": array([1, 2]), "h": [10]}
    new_values, idx_mapping = hdf_database._HDFDatabase__get_missing_hdf_output_dataset(
        10, keys_group, values
    )
    assert new_values == {"h": [10]}
    assert idx_mapping == {"h": 2}

    values = {"f": 0.1, "g": array([1, 2])}
    new_values, idx_mapping = hdf_database._HDFDatabase__get_missing_hdf_output_dataset(
        10, keys_group, values
    )
    assert new_values == {}
    assert idx_mapping == {}

    values = {"f": 0.1, "g": array([1, 2]), "h": [2, 3], "i": 20}
    new_values, idx_mapping = hdf_database._HDFDatabase__get_missing_hdf_output_dataset(
        10, keys_group, values
    )
    assert new_values == {"h": [2, 3], "i": 20}
    assert idx_mapping == {"h": 2, "i": 3}
