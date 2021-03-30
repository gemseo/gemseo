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
from gemseo.post.dataset.yvsx import YvsX
from gemseo.problems.dataset.rosenbrock import RosenbrockDataset

standard_library.install_aliases()


def test_constructor(tmp_path):

    dataset = RosenbrockDataset()
    plot = YvsX(dataset)
    plot.execute(
        x="x",
        x_comp=0,
        y="rosen",
        save=True,
        show=False,
        file_path=join(str(tmp_path), "yvsx"),
    )
    assert len(plot.output_files) == 1

    with pytest.raises(ValueError):
        plot.execute(
            x="foo",
            x_comp=0,
            y="rosen",
            save=True,
            show=False,
            file_path=join(str(tmp_path), "yvsx"),
        )

    dataset = Dataset()
    dataset.add_group("inputs", array([[0], [1], [2], [3]]), ["x"], {"x": 1})
    dataset.add_group(
        "outputs", array([[0.0, 0], [1.0, 1], [2.0, 2.0], [3.0, 3]]), ["y"], {"y": 2}
    )
    plot = YvsX(dataset)
    plot.execute(
        x="x",
        y="y",
        y_comp=0,
        save=True,
        show=False,
        file_path=join(str(tmp_path), "yvsx"),
    )
