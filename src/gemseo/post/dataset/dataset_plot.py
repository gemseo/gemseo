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
"""An abstract class to plot data from a :class:`.Dataset`.

The :mod:`~gemseo.post.dataset.dataset_plot` module implements the abstract
:class:`.DatasetPlot` class whose purpose is to build a graphical representation of a
:class:`.Dataset` and to display it on screen or save it to a file.

This abstract class has to be overloaded by concrete ones implementing at least method
:meth:`!DatasetPlot._run`.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from inspect import getfullargspec
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import NamedTuple
from typing import Union

from numpy import linspace
from strenum import StrEnum

from gemseo.post.dataset.plot_factory_factory import PlotFactoryFactory
from gemseo.post.dataset.plot_settings import PlotSettings
from gemseo.utils.compatibility.matplotlib import get_color_map
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from gemseo.datasets.dataset import Dataset
    from gemseo.utils.matplotlib_figure import FigSizeType


DatasetPlotPropertyType = Union[str, int, float, Sequence[Union[str, int, float]]]

VariableType = Union[str, tuple[str, int]]


class DatasetPlot(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Abstract class for plotting a dataset."""

    _common_settings: PlotSettings
    """The settings common to many plot classes."""

    _n_items: int
    """The number of items to plot.

    This notion is specific to the class deriving from :class:`.DatasetPlot`.
    """

    _specific_settings: NamedTuple
    """The settings specific to this plot class."""

    __common_dataset: Dataset
    """The dataset to be plotted."""

    __figure_file_paths: list[str]
    """The figure file paths."""

    __names_to_labels: Mapping[str, str]
    """The variable names bound to the variable labels."""

    __specific_data: tuple
    """The data pre-processed specifically for this :class:`.DatasetPlot`."""

    class PlotEngine(StrEnum):
        """An engine of plots."""

        MATPLOTLIB = "MatplotlibPlotFactory"
        PLOTLY = "PlotlyPlotFactory"

    DEFAULT_PLOT_ENGINE: ClassVar[PlotEngine] = PlotEngine.MATPLOTLIB
    """The default engine of plots."""

    FILE_FORMATS_TO_PLOT_ENGINES: ClassVar[dict[str, PlotEngine]] = {
        "html": PlotEngine.PLOTLY
    }
    """The file formats bound to the engines of plots.

    The method :meth:`.execute` uses this dictionary
    to select the engine of plots associated with its ``file_format`` argument.
    If missing, the method uses the :attr:`.DEFAULT_PLOT_ENGINE`.
    """

    def __init__(
        self,
        dataset: Dataset,
        **parameters: Any,
    ) -> None:
        """
        Args:
            dataset: The dataset containing the data to plot.
            **parameters: The parameters of the visualization.

        Raises:
            ValueError: If the dataset is empty.
        """  # noqa: D205, D212, D415
        if dataset.empty:
            raise ValueError("Dataset is empty.")

        annotations = getfullargspec(self.__init__).annotations
        parameter_names_to_types = [(name, annotations[name]) for name in parameters]

        specific_settings = NamedTuple("SpecificSettings", parameter_names_to_types)
        self._common_settings = PlotSettings()
        self._n_items = 0
        self._specific_settings = specific_settings(**parameters)
        self.__common_dataset = dataset
        self.__figure_file_paths = []
        self.__names_to_labels = {}
        self.__specific_data = self._create_specific_data_from_dataset()

    @property
    def dataset(self) -> Dataset:
        """The dataset."""
        return self.__common_dataset

    @property
    def color(self) -> str | list[str]:
        """The colors for the series; if empty, use a default one."""
        return self._common_settings.color

    @color.setter
    def color(self, value: str | list[str]) -> None:
        if isinstance(value, str) and self._n_items:
            self._common_settings.color = [value] * self._n_items
        else:
            self._common_settings.color = value

    @property
    def colormap(self) -> str:
        """The color map."""
        return self._common_settings.colormap

    @colormap.setter
    def colormap(self, value: str) -> None:
        self._common_settings.colormap = value

    @property
    def font_size(self) -> int:
        """The font size."""
        return self._common_settings.font_size

    @font_size.setter
    def font_size(self, value: int) -> None:
        self._common_settings.font_size = value

    @property
    def legend_location(self) -> str:
        """The location of the legend."""
        return self._common_settings.legend_location

    @legend_location.setter
    def legend_location(self, value: str) -> None:
        self._common_settings.legend_location = value

    @property
    def linestyle(self) -> str | list[str]:
        """The line style for the series; if empty, use a default one."""
        return self._common_settings.linestyle

    @linestyle.setter
    def linestyle(self, value: str | list[str]) -> None:
        if isinstance(value, str) and self._n_items:
            self._common_settings.linestyle = [value] * self._n_items
        else:
            self._common_settings.linestyle = value

    @property
    def marker(self) -> str | list[str]:
        """The marker for the series; if empty, use a default one."""
        return self._common_settings.marker

    @marker.setter
    def marker(self, value: str | list[str]) -> None:
        if isinstance(value, str) and self._n_items:
            self._common_settings.marker = [value] * self._n_items
        else:
            self._common_settings.marker = value

    @property
    def title(self) -> str:
        """The title of the plot."""
        return self._common_settings.title

    @title.setter
    def title(self, value: str) -> None:
        self._common_settings.title = value

    @property
    def xtick_rotation(self) -> str:
        """The rotation angle in degrees for the x-ticks."""
        return self._common_settings.xtick_rotation

    @xtick_rotation.setter
    def xtick_rotation(self, value: str) -> None:
        self._common_settings.xtick_rotation = value

    @property
    def rmin(self) -> float | None:
        """The minimum value on the r-axis; if ``None``, compute it from data."""
        return self._common_settings.rmin

    @rmin.setter
    def rmin(self, value: float | None) -> None:
        self._common_settings.rmin = value

    @property
    def rmax(self) -> float | None:
        """The maximum value on the r-axis; if ``None``, compute it from data."""
        return self._common_settings.rmax

    @rmax.setter
    def rmax(self, value: float | None) -> None:
        self._common_settings.rmax = value

    @property
    def xmin(self) -> float | None:
        """The minimum value on the x-axis; if ``None``, compute it from data."""
        return self._common_settings.xmin

    @xmin.setter
    def xmin(self, value: float | None) -> None:
        self._common_settings.xmin = value

    @property
    def xmax(self) -> float | None:
        """The maximum value on the x-axis; if ``None``, compute it from data."""
        return self._common_settings.xmax

    @xmax.setter
    def xmax(self, value: float | None) -> None:
        self._common_settings.xmax = value

    @property
    def ymin(self) -> float | None:
        """The minimum value on the y-axis; if ``None``, compute it from data."""
        return self._common_settings.ymin

    @ymin.setter
    def ymin(self, value: float | None) -> None:
        self._common_settings.ymin = value

    @property
    def ymax(self) -> float | None:
        """The maximum value on the y-axis; if ``None``, compute it from data."""
        return self._common_settings.ymax

    @ymax.setter
    def ymax(self, value):
        self._common_settings.ymax = value

    @property
    def zmin(self) -> float | None:
        """The minimum value on the z-axis; if ``None``, compute it from data."""
        return self._common_settings.zmin

    @zmin.setter
    def zmin(self, value: float | None):
        self._common_settings.zmin = value

    @property
    def zmax(self) -> float | None:
        """The maximum value on the z-axis; if ``None``, compute it from data."""
        return self._common_settings.zmax

    @zmax.setter
    def zmax(self, value: float | None) -> None:
        self._common_settings.zmax = value

    @property
    def xlabel(self) -> str:
        """The label for the x-axis."""
        return self._common_settings.xlabel

    @xlabel.setter
    def xlabel(self, value: str) -> None:
        self._common_settings.xlabel = value

    @property
    def ylabel(self) -> str:
        """The label for the y-axis."""
        return self._common_settings.ylabel

    @ylabel.setter
    def ylabel(self, value: str) -> None:
        self._common_settings.ylabel = value

    @property
    def zlabel(self) -> str:
        """The label for the z-axis."""
        return self._common_settings.zlabel

    @zlabel.setter
    def zlabel(self, value: str) -> None:
        self._common_settings.zlabel = value

    @property
    def fig_size(self) -> FigSizeType:
        """The figure size."""
        return self._common_settings.fig_size

    @fig_size.setter
    def fig_size(self, value: FigSizeType) -> None:
        self._common_settings.fig_size = value

    @property
    def fig_size_x(self) -> float:
        """The x-component of figure size."""
        return self.fig_size[0]

    @fig_size_x.setter
    def fig_size_x(self, value: float) -> None:
        self.fig_size = (value, self.fig_size_y)

    @property
    def fig_size_y(self) -> float:
        """The y-component of figure size."""
        return self.fig_size[1]

    @fig_size_y.setter
    def fig_size_y(self, value: float) -> None:
        self.fig_size = (self.fig_size_x, value)

    @property
    def output_files(self) -> list[str]:
        """The paths to the output files."""
        return self.__figure_file_paths

    def execute(
        self,
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_format: str = "png",
        file_name_suffix: str = "",
        **engine_parameters: Any,
    ) -> list[Figure]:
        """Execute the post-processing.

        Args:
            save: If ``True``, save the plot.
            show: If ``True``, display the plot.
            file_path: The path of the file to save the figures.
                If empty,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_format``.
            directory_path: The path of the directory to save the figures.
                If ``None``, use the current working directory.
            file_name: The name of the file to save the figures.
                If ``None``, use a default one generated by the post-processing.
            file_format: A file format, e.g. 'png', 'pdf', 'svg', ...
            file_name_suffix: The suffix to be added to the file name.
            **engine_parameters: The parameters specific to the plot engine.

        Returns:
            The figures.
        """
        engine = self.FILE_FORMATS_TO_PLOT_ENGINES.get(
            file_format, self.DEFAULT_PLOT_ENGINE
        )
        plot_factory = PlotFactoryFactory().create(engine)
        plot = plot_factory.create(
            self.__class__.__name__,
            self.dataset,
            self._common_settings,
            self._specific_settings,
            *self.__specific_data,
            **engine_parameters,
        )
        if show:
            plot.show()

        if save:
            args = (file_path, directory_path, file_name, file_format, file_name_suffix)
            self.__figure_file_paths = list(plot.save(*args))

        return plot.figures

    def _create_specific_data_from_dataset(self) -> tuple:
        """Pre-process the dataset specifically for this type of :class:`.DatasetPlot`.

        Returns:
            The data resulting from the pre-processing of the dataset.
        """
        return ()

    def _get_variable_names(
        self,
        dataframe_columns: Iterable[tuple],
    ) -> list[str]:
        """Return the names of the variables from the columns of a pandas DataFrame.

        Args:
            dataframe_columns: The columns of a pandas DataFrame.

        Returns:
            The names of the variables.
        """
        new_columns = []
        for column in dataframe_columns:
            name = self._get_component_name(
                column[1], column[2], self.dataset.variable_names_to_n_components
            )
            new_columns.append(name)

        return new_columns

    @staticmethod
    def _get_component_name(
        name: str, component: int, names_to_sizes: Mapping[str, int]
    ) -> str:
        """Return the name of a variable component.

        Args:
            name: The name of the variable.
            component: The component of the variable.
            names_to_sizes: The sizes of the variables.

        Returns:
            The name of the variable component.
        """
        if names_to_sizes[name] == 1:
            return name
        return f"{name}({component})"

    def _get_label(
        self,
        variable: str | tuple[str, str, int],
    ) -> tuple[str, tuple[str, str, int]]:
        """Return the label related to a variable name and a refactored variable name.

        Args:
            variable: The name of a variable,
                either a string (e.g. "x")
                or a tuple formatted as ``(group_name, variable_name, component)``.

        Returns:
            The label related to a variable, e.g. "x[0]",
            as well as the refactored variable name, e.g. (group_name, "x", 0).
        """
        if isinstance(variable, str):
            return variable, (self.dataset.get_group_names(variable)[0], variable, 0)

        return repr_variable(variable[1], variable[2]), variable

    def _set_color(
        self,
        n_items: int,
    ) -> None:
        """Set the colors of the items to be plotted.

        Args:
            n_items: The number of items to be plotted.
        """
        color_map = get_color_map(self.colormap)
        self.color = self.color or [color_map(c) for c in linspace(0, 1, n_items)]
        if isinstance(self.color, str):
            self.color = [self.color] * n_items

    def _set_linestyle(self, n_items: int, linestyle: str | Sequence[str]) -> None:
        """Set the line style of the items to be plotted.

        Args:
            n_items: The number of items to be plotted.
            linestyle: The default line style to use
                when :attr:`.linestyle` is ``None`.
        """
        self.linestyle = self.linestyle or linestyle
        if isinstance(self.linestyle, str):
            self.linestyle = [self.linestyle] * n_items

    def _set_marker(
        self,
        n_items: int,
        marker: str | Sequence[str] | None,
    ) -> None:
        """Set the marker of the items to be plotted.

        Args:
            n_items: The number of items to be plotted.
            marker: The default marker to use when :attr:`.marker` is ``None``.
        """
        self.marker = self.marker or marker
        if isinstance(self.marker, str):
            self.marker = [self.marker] * n_items

    @property
    def labels(self) -> Mapping[str, str]:
        """The labels for the variables."""
        return self._common_settings.labels

    @labels.setter
    def labels(self, names_to_labels: Mapping[str, str]) -> None:
        self._common_settings.labels = names_to_labels

    @staticmethod
    def _force_variable_to_tuple(variable: VariableType) -> tuple[str, int]:
        """Return a variable as a tuple ``(variable_name, variable_component)``.

        Args:
            variable: The original variable.

        Returns:
            The variable as ``(variable_name, variable_component)``.
        """
        if isinstance(variable, str):
            variable = (variable, 0)

        return variable
