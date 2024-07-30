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

from strenum import StrEnum

from gemseo.post.dataset.plot_settings import PlotSettings
from gemseo.post.dataset.plots.factory_factory import PlotFactoryFactory
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.datasets.dataset import Dataset
    from gemseo.post.dataset.plots.base_plot import BasePlot
    from gemseo.post.dataset.plots.factory import PlotFactory
    from gemseo.utils.matplotlib_figure import FigSizeType


DatasetPlotPropertyType = Union[str, int, float, Sequence[Union[str, int, float]]]


class DatasetPlot(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Abstract class for plotting a dataset."""

    _common_settings: PlotSettings
    """The settings common to many plot classes."""

    _specific_settings: NamedTuple
    """The settings specific to this plot class."""

    __common_dataset: Dataset
    """The dataset to be plotted."""

    __figure_file_paths: list[str]
    """The figure file paths."""

    __figures: list[BasePlot]
    """The figures."""

    __names_to_labels: Mapping[str, str]
    """The variable names bound to the variable labels."""

    __specific_data: tuple[Any, ...]
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
            msg = "Dataset is empty."
            raise ValueError(msg)

        annotations = getfullargspec(self.__class__.__init__).annotations
        parameter_names_to_types = [(name, annotations[name]) for name in parameters]

        specific_settings = NamedTuple("specific_settings", parameter_names_to_types)
        self._common_settings = PlotSettings()
        self._specific_settings = specific_settings(**parameters)
        self.__common_dataset = dataset
        self.__figure_file_paths = []
        self.__figures = []
        self.__names_to_labels = {}
        self.__specific_data = self._create_specific_data_from_dataset()

    @property
    def dataset(self) -> Dataset:
        """The dataset."""
        return self.__common_dataset

    @property
    def color(self) -> str | list[str]:
        """The color.

        Either a global one or one per item if ``n_items`` is non-zero.
        If empty, use a default one.
        """
        return self._common_settings.color

    @color.setter
    def color(self, value: str | list[str]) -> None:
        self._common_settings.set_colors(value)

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
    def linestyle(self) -> str | Sequence[str]:
        """The line style.

        Either a global one or one per item if ``n_items`` is non-zero.
        If empty, use a default one.
        """
        return self._common_settings.linestyle

    @linestyle.setter
    def linestyle(self, value: str | Sequence[str]) -> None:
        self._common_settings.set_linestyles(value)

    @property
    def marker(self) -> str | Sequence[str]:
        """The marker.

        Either a global one or one per item if ``n_items`` is non-zero.
        If empty, use a default one.
        """
        return self._common_settings.marker

    @marker.setter
    def marker(self, value: str | list[str]) -> None:
        self._common_settings.set_markers(value)

    @property
    def title(self) -> str:
        """The title of the plot."""
        return self._common_settings.title

    @title.setter
    def title(self, value: str) -> None:
        self._common_settings.title = value

    @property
    def xtick_rotation(self) -> float:
        """The rotation angle in degrees for the x-ticks."""
        return self._common_settings.xtick_rotation

    @xtick_rotation.setter
    def xtick_rotation(self, value: float) -> None:
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
    def ymax(self, value: float) -> None:
        self._common_settings.ymax = value

    @property
    def zmin(self) -> float | None:
        """The minimum value on the z-axis; if ``None``, compute it from data."""
        return self._common_settings.zmin

    @zmin.setter
    def zmin(self, value: float | None) -> None:
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
    def grid(self) -> bool:
        """Whether to add a grid."""
        return self._common_settings.grid

    @grid.setter
    def grid(self, value: bool) -> None:
        self._common_settings.grid = value

    @property
    def output_files(self) -> list[str]:
        """The paths to the output files."""
        return self.__figure_file_paths

    @property
    def figures(self) -> list[BasePlot]:
        """The figures."""
        return self.__figures

    def execute(
        self,
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_format: str = "png",
        file_name_suffix: str = "",
        **engine_parameters: Any,
    ) -> list[BasePlot]:
        """Execute the post-processing.

        Args:
            save: Whether to save the plot.
            show: Whether to display the plot.
            file_path: The path of the file to save the figures.
                If empty,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_format``.
            directory_path: The path of the directory to save the figures.
                If empty, use the current working directory.
            file_name: The name of the file to save the figures.
                If empty, use a default one generated by the post-processing.
            file_format: A file format, e.g. 'png', 'pdf', 'svg', ...
            file_name_suffix: The suffix to be added to the file name.
            **engine_parameters: The parameters specific to the plot engine.

        Returns:
            The figures.
        """
        engine = self.FILE_FORMATS_TO_PLOT_ENGINES.get(
            file_format, self.DEFAULT_PLOT_ENGINE
        )
        plot_factory: PlotFactory[Any] = PlotFactoryFactory().create(engine)
        plot: BasePlot = plot_factory.create(
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
            self.__figure_file_paths = list(
                plot.save(
                    file_path, directory_path, file_name, file_format, file_name_suffix
                )
            )

        self.__figures = plot.figures

        return self.__figures

    def _create_specific_data_from_dataset(self) -> tuple[Any, ...]:
        """Pre-process the dataset specifically for this type of :class:`.DatasetPlot`.

        Returns:
            The data resulting from the pre-processing of the dataset.
        """
        return ()

    def _get_variable_names(
        self,
        dataframe_columns: Iterable[tuple[str, str, int]],
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

    @property
    def labels(self) -> Mapping[str, str]:
        """The labels for the variables."""
        return self._common_settings.labels

    @labels.setter
    def labels(self, names_to_labels: Mapping[str, str]) -> None:
        self._common_settings.labels = names_to_labels

    @property
    def _n_items(self) -> int:
        """The number of items to plot.

        The item definition is specific to the plot type and is used to define
        properties, e.g. color and line style, for each item.

        For example, items can correspond to curves or series of points.

        By default, a graph has no item.
        """
        return self._common_settings.n_items

    @_n_items.setter
    def _n_items(self, n_items: int) -> None:
        self._common_settings.n_items = n_items
