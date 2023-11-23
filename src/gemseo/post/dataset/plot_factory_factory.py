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
"""A factory of factories of plots."""

from __future__ import annotations

from gemseo.core.base_factory import BaseFactory
from gemseo.post.dataset.plot_factory import PlotFactory


class PlotFactoryFactory(BaseFactory):
    """A factory of factories of plots.

    A factory of plots is used to create plots from a visualization library,
    a.k.a. plot engine,
    e.g. :class:`.MatplotlibPlotFactory` for matplotlib-based plots
    and :class:`.PlotlyPlotFactory` for plotly-based plots.

    :class:`.PlotsFactoryFactory` is used to create any factory of plots.
    This mechanism is used by :class:`.DatasetPlot`
    by associating one plot engine per file format.
    """

    _CLASS = PlotFactory
    _MODULE_NAMES = ("gemseo.post.dataset._plotly", "gemseo.post.dataset._matplotlib")
