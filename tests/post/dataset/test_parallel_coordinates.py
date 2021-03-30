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

from gemseo.post.dataset.parallel_coordinates import ParallelCoordinates
from gemseo.problems.dataset.iris import IrisDataset

standard_library.install_aliases()


def test_constructor(tmp_path):
    file_path = join(str(tmp_path), "parallel")
    dataset = IrisDataset()
    plot = ParallelCoordinates(dataset)

    with pytest.raises(ValueError):
        plot.execute(classifier="dummy", save=True, show=False, file_path=file_path)

    plot.execute(classifier="specy", save=True, show=False, file_path=file_path)
    assert len(plot.output_files) == 1

    plot.execute(
        classifier="specy", save=True, show=False, lower=1.0, file_path=file_path
    )
    assert len(plot.output_files) == 1
