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

The :mod:`~gemseo.post.dataset.dataset_plot` module
implements the abstract :class:`.DatasetPlot` class
whose purpose is to build a graphical representation of a :class:`.Dataset`
and to display it on screen or save it to a file.

This abstract class has to be overloaded by concrete ones
implementing at least method :meth:`!DatasetPlot._run`.
"""
from __future__ import annotations

from collections import namedtuple
from numbers import Number
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Union

from docstring_inheritance import GoogleDocstringInheritanceMeta
from matplotlib.axes import Axes
from numpy import linspace

from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.file_path_manager import FileType
from gemseo.utils.matplotlib_figure import save_show_figure

if TYPE_CHECKING:
    from gemseo.core.dataset import Dataset

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pathlib import Path

DatasetPlotPropertyType = Union[str, int, float, Sequence[Union[str, int, float]]]


class DatasetPlot(metaclass=GoogleDocstringInheritanceMeta):
    """Abstract class for plotting a dataset."""

    color: str | list[str]
    """The color(s) for the series.

    If empty, use a default one.
    """

    colormap: str
    """The color map."""

    dataset: Dataset
    """The dataset to be plotted."""

    fig_size: tuple[float, float]
    """The figure size."""

    font_size: int
    """The font size."""

    legend_location: str
    """The location of the legend."""

    linestyle: str | list[str]
    """The line style(s) for the series.

    If empty, use a default one.
    """

    marker: str | list[str]
    """The marker(s) for the series.

    If empty, use a default one.
    """

    title: str
    """The title of the plot."""

    xlabel: str
    """The label for the x-axis."""

    xmin: float | None
    """The minimum value on the x-axis.

    If ``None``, compute it from data.
    """

    xmax: float | None
    """The maximum value on the x-axis.".

    If ``None``, compute it from data.
    """

    ylabel: str
    """The label for the y-axis."""

    ymin: float | None
    """The minimum value on the y-axis.

    If ``None``, compute it from data.
    """

    ymax: float | None
    """The maximum value on the y-axis.

    If ``None``, compute it from data.
    """

    zlabel: str
    """The label for the z-axis."""

    zmin: float | None
    """The minimum value on the z-axis.

    If ``None``, compute it from data.
    """

    zmax: float | None
    """The maximum value on the z-axis.

    If ``None``, compute it from data.
    """

    def __init__(
        self,
        dataset: Dataset,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            dataset: The dataset containing the data to plot.

        Raises:
            ValueError: If the dataset is empty.
        """  # noqa: D205, D212, D415
        param = namedtuple(f"{self.__class__.__name__}Parameters", kwargs.keys())
        self._param = param(**kwargs)

        if dataset.is_empty():
            raise ValueError("Dataset is empty.")

        self.color = ""
        self.colormap = "rainbow"
        self.dataset = dataset
        self.font_size = 10
        self.legend_location = "best"
        self.linestyle = ""
        self.marker = ""
        self.title = ""
        self.rmin = None
        self.rmax = None
        self.xlabel = ""
        self.xmin = None
        self.xmax = None
        self.ylabel = ""
        self.ymin = None
        self.ymax = None
        self.zlabel = ""
        self.zmin = None
        self.zmax = None
        self.fig_size = (6.4, 4.8)
        self.__file_path_manager = FilePathManager(
            FileType.FIGURE,
            default_name=FilePathManager.to_snake_case(self.__class__.__name__),
        )
        self.__output_files = []
        self.__names_to_labels = {}

    @property
    def output_files(self) -> list[str]:
        """The paths to the output files."""
        return self.__output_files

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

    def execute(
        self,
        save: bool = True,
        show: bool = False,
        file_path: str | Path | None = None,
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_format: str | None = None,
        properties: Mapping[str, DatasetPlotPropertyType] | None = None,
        fig: None | Figure = None,
        axes: None | Axes = None,
        **plot_options,
    ) -> list[Figure]:
        """Execute the post-processing.

        Args:
            save: If True, save the plot.
            show: If True, display the plot.
            file_path: The path of the file to save the figures.
                If None,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_format``.
            directory_path: The path of the directory to save the figures.
                If None, use the current working directory.
            file_name: The name of the file to save the figures.
                If None, use a default one generated by the post-processing.
            file_format: A file format, e.g. 'png', 'pdf', 'svg', ...
                If None, use a default file extension.
            properties: The general properties of a :class:`.DatasetPlot`.
            fig: The figure to plot the data.
                If ``None``, create a new one.
            axes: The axes to plot the data.
                If ``None``, create new ones.
            **plot_options: The options of the current class
                inheriting from :class:`.DatasetPlot`.

        Returns:
            The figures.

        Raises:
            AttributeError: When the name of a property is not the name of an attribute.
        """
        properties = properties or {}
        for name, value in properties.items():
            if not hasattr(self, name):
                raise AttributeError(
                    f"{name} is not an attribute of {self.__class__.__name__}."
                )
            setattr(self, name, value)

        if file_path is not None:
            file_path = Path(file_path)

        file_path = self.__file_path_manager.create_file_path(
            file_path=file_path,
            directory_path=directory_path,
            file_name=file_name,
            file_extension=file_format,
        )
        return self._run(save, show, file_path, fig, axes, **plot_options)

    def _run(
        self,
        save: bool,
        show: bool,
        file_path: Path,
        fig: None | Figure,
        axes: None | Axes,
        **plot_options,
    ) -> list[Figure]:
        """Create the post-processing and save or display it.

        Args:
            save: If True, save the plot on the disk.
            show: If True, display the plot.
            file_path: The file path.
            fig: The figure to plot the data.
                If ``None``, create a new one.
            axes: The axes to plot the data.
                If ``None``, create new ones.
            **plot_options: The options of the current class
                inheriting from :class:`.DatasetPlot`.

        Returns:
            The figures.
        """
        if plot_options:
            self._param = self._param._replace(**plot_options)

        figures = self._plot(fig=fig, axes=axes)
        if fig or axes:
            return []

        for index, sub_figure in enumerate(figures):
            if save:
                if len(figures) > 1:
                    fig_file_path = self.__file_path_manager.add_suffix(
                        file_path, index
                    )
                else:
                    fig_file_path = file_path
                self.__output_files.append(str(fig_file_path))

            else:
                fig_file_path = None

            save_show_figure(
                sub_figure,
                show,
                fig_file_path,
            )

        return figures

    def _plot(
        self,
        fig: None | Figure = None,
        axes: None | Axes = None,
    ) -> list[Figure]:
        """Define the way as the dataset is plotted.

        Args:
            fig: The figure to plot the data.
                If ``None``, create a new one.
            axes: The axes to plot the data.
                If ``None``, create new ones.

        Returns:
            The figures.
        """
        raise NotImplementedError

    def _get_variables_names(
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
            name = self._get_component_name(column[1], column[2], self.dataset.sizes)
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
        else:
            return f"{name}({component})"

    def _get_label(
        self,
        variable: str | tuple[str, int],
    ) -> tuple[str, tuple[str, int]]:
        """Return the label related to a variable name and a refactored variable name.

        Args:
            variable: The name of a variable,
                either a string (e.g. "x") or a (name, component) tuple (e.g. ("x", 0)).

        Returns:
            The label related to a variable, e.g. "x(0)",
            as well as the refactored variable name, e.g. (x,0).
        """
        error_message = (
            "'variable' must be either a string or a tuple"
            " whose first component is a string and second"
            " one is an integer"
        )
        if isinstance(variable, str):
            label = variable
            variable = (self.dataset.get_group(variable), variable, "0")
        elif hasattr(variable, "__len__") and len(variable) == 3:
            is_string = isinstance(variable[0], str)
            is_string = is_string and isinstance(variable[1], str)
            is_number = isinstance(variable[2], Number)
            if is_string and is_number:
                label = f"{variable[1]}({variable[2]})"
                variable[2] = str(variable[2])
                variable = tuple(variable)
            else:
                raise TypeError(error_message)
        else:
            raise TypeError(error_message)
        return label, variable

    def _set_color(
        self,
        n_items: int,
    ) -> None:
        """Set the colors of the items to be plotted.

        Args:
            n_items: The number of items to be plotted.
        """
        colormap = plt.cm.get_cmap(self.colormap)
        color = [colormap(color) for color in linspace(0, 1, n_items)]
        self.color = self.color or color
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
        """The labels of the variables."""
        return self.__names_to_labels

    @labels.setter
    def labels(self, names_to_labels: Mapping[str, str]) -> None:
        self.__names_to_labels = names_to_labels

    def _get_figure_and_axes(
        self,
        fig: Figure | None,
        axes: Axes | None,
        fig_size: tuple[float, float] | None = None,
    ) -> tuple[Figure, Axes]:
        """Return the figure and axes to plot the data.

        Args:
            fig: The figure to plot the data.
                If ``None``, create a new one.
            axes: The axes to plot the data.
                If ``None``, create new ones.
            fig_size: The width and height of the figure in inches.
                If ``None``, use the default ``fig_size``.

        Returns:
            The figure and axis to plot the data.
        """
        if fig is None:
            if axes is not None:
                raise ValueError(
                    "The figure associated with the given axes is missing."
                )

            return plt.subplots(figsize=fig_size or self.fig_size)

        if axes is None:
            raise ValueError("The axes associated with the given figure are missing.")

        return fig, axes
