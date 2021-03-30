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

from gemseo.post.dataset.scatter_plot_matrix import ScatterMatrix
from gemseo.problems.dataset.iris import IrisDataset

standard_library.install_aliases()


def test_constructor(tmp_path):

    dataset = IrisDataset()
    plot = ScatterMatrix(dataset)
    plot.execute(save=True, show=False, file_path=join(str(tmp_path), "scatter"))
    assert len(plot.output_files) == 1

    plot.execute(
        kde=True, save=True, show=False, file_path=join(str(tmp_path), "scatter")
    )
    assert len(plot.output_files) == 1

    plot.execute(
        classifier="specy",
        save=True,
        show=False,
        file_path=join(str(tmp_path), "scatter"),
    )
    assert len(plot.output_files) == 1

    with pytest.raises(ValueError):
        plot.execute(
            classifier="dummy",
            save=True,
            show=False,
            file_path=join(str(tmp_path), "scatter"),
        )
