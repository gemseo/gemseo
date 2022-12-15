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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Evolution of the variables by means of a color scale."""
from __future__ import annotations

from typing import Iterable

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import arange

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot
from gemseo.utils.compatibility.matplotlib import SymLogNorm


class ColorEvolution(DatasetPlot):
    """Evolution of the variables by means of a color scale.

    Based on the matplotlib function :meth:`imshow`.

    Tip:
        Use :attr:`.colormap` to set a matplotlib colormap, e.g. ``"seismic"``.
    """

    def __init__(
        self,
        dataset: Dataset,
        variables: Iterable[str] | None = None,
        use_log: bool = False,
        opacity: float = 0.6,
        **options: bool | float | str | None,
    ) -> None:
        """
        Args:
            variables: The variables of interest
                If ``None``, use all the variables.
            use_log: Whether to use a symmetric logarithmic scale.
            opacity: The level of opacity (0 = transparent; 1 = opaque).
            **options: The options for the matplotlib function :meth:`imshow`.
        """  # noqa: D205, D212, D415
        options_ = {
            "interpolation": "nearest",
            "aspect": "auto",
        }
        options_.update(options)
        super().__init__(
            dataset,
            variables=variables,
            use_log=use_log,
            opacity=opacity,
            options=options_,
        )

    def _plot(
        self,
        fig: None | Figure = None,
        axes: None | Axes = None,
    ) -> list[Figure]:
        variables = self._param.variables or self.dataset.variables
        data = self.dataset.get_data_by_names(variables, False).T

        if self._param.use_log:
            maximum = abs(data).max()
            norm = SymLogNorm(linthresh=1.0, vmin=-maximum, vmax=maximum)
        else:
            norm = None

        fig, axes = self._get_figure_and_axes(fig=fig, axes=axes)
        img_ = axes.imshow(
            data,
            cmap=self.colormap,
            norm=norm,
            alpha=self._param.opacity,
            **self._param.options,
        )
        names = self.dataset.get_column_names(variables)
        axes.set_yticks(arange(len(names)))
        axes.set_yticklabels(names)
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_title(self.title)
        fig.colorbar(img_, ax=axes)
        return [fig]
