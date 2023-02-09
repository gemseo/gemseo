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
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from os.path import exists

import numpy as np
import pytest
from gemseo.core.dataset import Dataset
from gemseo.problems.scalable.data_driven.diagonal import ScalableDiagonalModel
from numpy import newaxis


def f_1(x_1, x_2, x_3):
    return np.sin(2 * np.pi * x_1) + np.cos(2 * np.pi * x_2) + x_3


def f_2(x_1, x_2, x_3):
    return np.sin(2 * np.pi * x_1) * np.cos(2 * np.pi * x_2) - x_3


@pytest.fixture
def dataset():
    data = Dataset("sinus")
    x1_val = x2_val = x3_val = np.linspace(0.0, 1.0, 10)[:, newaxis]
    data.add_variable("x1", x1_val, data.INPUT_GROUP)
    data.add_variable("x2", x2_val, data.INPUT_GROUP)
    data.add_variable("x3", x2_val, data.INPUT_GROUP)
    data.add_variable("y1", f_1(x1_val, x2_val, x3_val), data.OUTPUT_GROUP)
    data.add_variable("y2", f_2(x1_val, x2_val, x3_val), data.OUTPUT_GROUP)
    return data


def test_constructor(dataset):
    ScalableDiagonalModel(dataset)
    with pytest.raises(TypeError):
        ScalableDiagonalModel(dataset, fill_factor="dummy")


def test_scalable_function(dataset):
    model = ScalableDiagonalModel(dataset)
    output = model.scalable_function()
    assert "y1" in output
    assert "y2" in output
    assert len(output["y1"].shape) == 1
    assert len(output["y1"] == 3)
    assert len(output["y2"].shape) == 1
    assert len(output["y2"] == 3)


def test_scalable_derivative(dataset):
    model = ScalableDiagonalModel(dataset)
    output = model.scalable_derivatives()
    assert "y1" in output
    assert "y2" in output
    assert len(output["y1"].shape) == 2
    assert output["y1"].shape[0] == 1
    assert output["y1"].shape[1] == 3
    assert len(output["y2"].shape) == 2
    assert output["y2"].shape[0] == 1
    assert output["y2"].shape[1] == 3


def test_plot(dataset, tmp_wd):
    model = ScalableDiagonalModel(dataset)
    model.plot_1d_interpolations(save=True)
    assert exists("sdm_sinus_y1_1D_interpolation_0.pdf")
    assert exists("sdm_sinus_y2_1D_interpolation_0.pdf")
    model.plot_dependency()
    assert exists("sdm_sinus_dependency.pdf")


def test_force_io_dependency(dataset):
    ScalableDiagonalModel(dataset, force_input_dependency=True)


def test_force_allow_unusedinputs(dataset):
    ScalableDiagonalModel(dataset, allow_unused_inputs=False)


def test_wrong_fill_factor(dataset):
    with pytest.raises(TypeError):
        ScalableDiagonalModel(dataset, fill_factor=-2)


def test_group_dep(dataset):
    model = ScalableDiagonalModel(dataset, group_dep={"y2": ["x1", "x2"]})
    assert model.model.io_dependency["y2"]["x3"][0] == 0.0
