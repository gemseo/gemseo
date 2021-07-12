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
"""Draw lines from a :class:`.Dataset`.

A :class:`.Lines` plot represents variables vs samples using lines.
"""
from __future__ import division, unicode_literals

from typing import List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from gemseo.post.dataset.dataset_plot import DatasetPlot, DatasetPlotPropertyType


class Lines(DatasetPlot):
    """Plot sampled variables as lines."""

    def _plot(
        self,
        properties,  # type: Mapping[str,DatasetPlotPropertyType]
        variables=None,  # type: Optional[Sequence[str]]
    ):  # type: (...) -> List[Figure]
        """
        Args:
            variables: The names of the variables to plot.
        """
        x_data = range(len(self.dataset))
        if variables is None:
            y_data = self.dataset.get_all_data(False, True)
            variables = y_data.keys()
        else:
            y_data = self.dataset[variables]

        plt.figure(figsize=self.figsize)
        self._set_color(properties, len(variables))
        self._set_linestyle(properties, len(variables), "-")
        index = 0
        for name, value in y_data.items():
            plt.plot(
                x_data,
                value,
                linestyle=self.linestyle[index],
                color=self.color[index],
                label=name,
            )
            index += 1
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.legend(loc=self.legend_location)
        return [plt.gcf()]
