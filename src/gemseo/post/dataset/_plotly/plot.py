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
"""A base plot class relying on plotly."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import Final
from typing import NamedTuple

from gemseo.post.dataset.base_plot import BasePlot

if TYPE_CHECKING:
    from pathlib import Path

    from plotly.graph_objs import Figure

    from gemseo.datasets.dataset import Dataset
    from gemseo.post.dataset.plot_settings import PlotSettings


class PlotlyPlot(BasePlot):
    """A base plot class relying on matplotlib."""

    _PLOTLY_LINESTYLES: Final[dict[str, str]] = {
        "-": "solid",
        ":": "dot",
        "--": "dash",
        "-.": "dashdot",
    }

    __figure: Figure
    """The plotly figure."""

    def __init__(
        self,
        dataset: Dataset,
        common_settings: PlotSettings,
        specific_settings: NamedTuple,
        *specific_data: Any,
    ) -> None:
        """
        Args:
            *args: The data to be plotted.
        """  # noqa: D205 D212 D415
        super().__init__(dataset, common_settings, specific_settings)
        self.__figure = self._create_figure(*specific_data)

    @abstractmethod
    def _create_figure(self, *specific_data: Any) -> Figure:
        """Create the plotly figure.

        Args:
            *specific_data: The data specific to this plot class.

        Returns:
            The plotly figure.
        """

    def show(self) -> None:  # noqa: D102
        self.__figure.show()

    def _save(self, file_path: Path) -> tuple[str]:
        file_format = file_path.suffix[1:]
        if file_format == "html":
            self.__figure.write_html(file_path)
        else:
            self.__figure.write_image(file=file_path, format=file_format)

        return (str(file_path),)

    @property
    def figures(self) -> list[Figure]:  # noqa: D102
        return [self.__figure]
