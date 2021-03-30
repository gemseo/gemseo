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
from __future__ import absolute_import, division, unicode_literals

from os.path import join

import pytest
from future import standard_library
from numpy import array

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.scatter import Scatter
from gemseo.problems.dataset.rosenbrock import RosenbrockDataset

standard_library.install_aliases()


def test_constructor(tmp_path):

    dataset = Dataset()
    dataset.add_group(
        "inputs", array([[0, 0], [1, 1], [0, 1], [1, 0]]), ["x", "y"], {"x": 1, "y": 1}
    )
    plot = Scatter(dataset)
    color = ["b", "b", "r", "r"]
    plot.execute(
        x="x",
        y="y",
        color=color,
        save=True,
        show=False,
        file_path=join(str(tmp_path), "scatter"),
    )

    dataset = Dataset()
    dataset.add_group(
        "inputs",
        array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0]]),
        ["x", "y"],
        {"x": 2, "y": 2},
    )
    plot = Scatter(dataset)
    color = ["b", "b", "r", "r"]
    plot.execute(
        x="x",
        y="y",
        color=color,
        save=True,
        show=False,
        file_path=join(str(tmp_path), "scatter"),
    )
