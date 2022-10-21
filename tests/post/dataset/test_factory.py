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

from gemseo.post.dataset.factory import DatasetPlotFactory
from gemseo.problems.dataset.rosenbrock import RosenbrockDataset


def test_instantiate_factory():
    DatasetPlotFactory()


def test_plots():
    factory = DatasetPlotFactory()
    plots = factory.plots
    assert "ScatterMatrix" in plots
    assert "ParallelCoordinates" in plots


def test_is_available():
    factory = DatasetPlotFactory()
    assert factory.is_available("ScatterMatrix")
    assert not factory.is_available("DummyPlot")


def test_create():
    factory = DatasetPlotFactory()
    dataset = RosenbrockDataset()
    dataset = factory.create("ScatterMatrix", dataset=dataset)
    assert not dataset.output_files
