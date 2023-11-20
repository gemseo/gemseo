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
"""Test the class IODataset."""

from __future__ import annotations

import pytest
from numpy import arange
from numpy import concatenate
from numpy.testing import assert_equal
from pandas.testing import assert_frame_equal

from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset


@pytest.fixture(scope="module")
def dataset() -> IODataset:
    """An input-output dataset.

    The inputs are "in_1" (2 components) and "in_2" (3 components)
    and the output name is "out_1" (2 components).

    The input values are: 0   1  2  3  4 5   6  7  8  9 ... 45 46 47 48 49

    The output values are: 0 1 2 3 ... 8 9
    """
    return IODataset.from_array(
        concatenate([arange(50).reshape(10, 5), arange(20).reshape(10, 2)], axis=1),
        ["in_1", "in_2", "out_1"],
        {"in_1": 2, "in_2": 3, "out_1": 2},
        {"in_1": "inputs", "in_2": "inputs", "out_1": "outputs"},
    )


def test_input_names(dataset):
    """Test the property input_names."""
    assert dataset.input_names == ["in_1", "in_2"]


def test_output_names(dataset):
    """Test the property input_names."""
    assert dataset.output_names == ["out_1"]


def test_n_samples(dataset):
    """Test the property n_samples."""
    assert dataset.n_samples == 10


def test_samples(dataset):
    """Test the property samples."""
    assert_equal(dataset.samples, arange(10))


def test_add_input_variable():
    """Test the method add_input_variable."""
    io_dataset = IODataset()
    io_dataset.add_input_variable("x", [[1.0], [2.0]])

    dataset = Dataset()
    dataset.name = io_dataset.__class__.__name__
    dataset.add_variable("x", [[1.0], [2.0]], group_name=IODataset.INPUT_GROUP)

    assert_frame_equal(io_dataset, dataset)


def test_add_output_variable():
    """Test the method add_input_variable."""
    io_dataset = IODataset()
    io_dataset.add_output_variable("x", [[1.0], [2.0]])

    dataset = Dataset()
    dataset.name = io_dataset.__class__.__name__
    dataset.add_variable("x", [[1.0], [2.0]], group_name=IODataset.OUTPUT_GROUP)

    assert_frame_equal(io_dataset, dataset)


def test_add_input_group():
    """Test the method add_input_group."""
    io_dataset = IODataset()
    io_dataset.add_input_group([[1.0], [2.0]], ["x"])

    dataset = Dataset()
    dataset.name = io_dataset.__class__.__name__
    dataset.add_group(IODataset.INPUT_GROUP, [[1.0], [2.0]], ["x"])

    assert_frame_equal(io_dataset.input_dataset, dataset)


def test_add_output_group():
    """Test the method add_output_group."""
    io_dataset = IODataset()
    io_dataset.add_output_group([[1.0], [2.0]], ["x"])

    dataset = Dataset()
    dataset.name = io_dataset.__class__.__name__
    dataset.add_group(IODataset.OUTPUT_GROUP, [[1.0], [2.0]], ["x"])

    assert_frame_equal(io_dataset.output_dataset, dataset)


def test_input_dataset(dataset):
    """Test the method input_dataset."""
    input_dataset = dataset.get_view(group_names=dataset.INPUT_GROUP)
    assert_frame_equal(dataset.input_dataset, input_dataset)


def test_output_dataset(dataset):
    """Test the property output_dataset."""
    output_dataset = dataset.get_view(group_names=dataset.OUTPUT_GROUP)
    assert_frame_equal(dataset.output_dataset, output_dataset)
