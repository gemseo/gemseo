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
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

import pytest
from numpy import array

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot
from gemseo.post.dataset.yvsx import YvsX


def test_empty_dataset():

    dataset = Dataset()
    with pytest.raises(ValueError):
        YvsX(dataset)


def test_plot_notimplementederror():
    dataset = Dataset()
    dataset.set_from_array(array([[1, 2]]))
    post = DatasetPlot(dataset)
    with pytest.raises(NotImplementedError):
        post._plot({})


def test_get_label():
    dataset = Dataset()
    dataset.set_from_array(array([[1, 2]]), variables=["x"], sizes={"x": 2})
    post = DatasetPlot(dataset)
    label, varname = post._get_label(["parameters", "x", 0])
    assert label == "x(0)"
    assert varname == ("parameters", "x", "0")
    with pytest.raises(TypeError):
        label, varname = post._get_label(123)


def test_custom():
    dataset = Dataset()
    dataset.set_from_array(array([[1, 2]]), variables=["x"], sizes={"x": 2})
    plot = DatasetPlot(dataset)
    assert plot.font_size == 10
    assert plot.title is None
    assert plot.xlabel is None
    assert plot.ylabel is None
    assert plot.zlabel is None
    plot.title = "title"
    assert plot.title == "title"
    plot.xlabel = "xlabel"
    assert plot.xlabel == "xlabel"
    plot.ylabel = "ylabel"
    assert plot.ylabel == "ylabel"
    plot.zlabel = "zlabel"
    assert plot.zlabel == "zlabel"
    plot.font_size = 2
    assert plot.font_size == 2
    plot.font_size *= 2
    assert plot.font_size == 4
