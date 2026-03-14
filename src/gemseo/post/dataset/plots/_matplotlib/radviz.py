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
"""Radar visualization based on matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pandas.plotting import radviz

from gemseo.post.dataset.plots._matplotlib.plot import MatplotlibPlot
from gemseo.post.dataset.radviz_settings import RadViz_Settings

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gemseo.datasets.dataset import Dataset


class RadViz(MatplotlibPlot[RadViz_Settings]):
    """RadViz visualization based on matplotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        ax: Axes | None,
        dataset: Dataset,
        classifier_name: str,
    ) -> list[Figure]:
        """
        Args:
            dataset: The dataset.
            classifier_name: The name of the classifier.
        """  # noqa: D205, D212, D415
        settings = self._settings
        fig, ax = self._get_figure_and_axes(fig, ax)
        radviz(dataset, classifier_name, ax=ax)
        ax.set_axisbelow(True)
        ax.grid(visible=settings.grid)
        ax.set_xlabel(settings.xlabel)
        ax.set_ylabel(settings.ylabel)
        ax.set_title(settings.title)
        return [fig]
