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
"""An abstract class to plot data from a [Dataset][gemseo.datasets.dataset.Dataset].

The [gemseo.post.dataset.base][gemseo.post.dataset.base] module
implements the abstract
[DatasetPlot][gemseo.post.dataset.base.BaseDatasetPlot] class
whose purpose is to build a graphical representation of a
[Dataset][gemseo.datasets.dataset.Dataset]
and to display it on screen or save it to a file.

This abstract class has to be overloaded by concrete ones implementing at least method
`DatasetPlot._run()`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import TypeVar

from strenum import StrEnum

from gemseo.post.dataset.base_settings import BaseDatasetPlotSettings
from gemseo.post.dataset.plots.factory_factory import PlotFactoryFactory
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from pathlib import Path

    from gemseo.datasets.dataset import Dataset
    from gemseo.post.dataset.plots.base import BasePlot
    from gemseo.post.dataset.plots.factory import PlotFactory

DatasetPlotPropertyType = str | int | float | Sequence[str | int | float]

T = TypeVar("T", bound=BaseDatasetPlotSettings)


class BaseDatasetPlot(Generic[T], metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base class for dataset visualizations."""

    settings: T

    __common_dataset: Dataset
    """The dataset to be plotted."""

    __figure_file_paths: list[str]
    """The figure file paths."""

    __figures: list[BasePlot]
    """The figures."""

    __name_to_label: Mapping[str, str]
    """The map from a variable name to its label."""

    __specific_data: tuple[Any, ...]
    """The data pre-processed specifically for this `DatasetPlot`."""

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

    The method
    [execute()][gemseo.post.dataset.base.BaseDatasetPlot.execute]
    uses this dictionary
    to select the engine of plots associated with its `file_format` argument.
    If missing, the method uses the
    [DEFAULT_PLOT_ENGINE][gemseo.post.dataset.base.BaseDatasetPlot.DEFAULT_PLOT_ENGINE].
    """

    settings_class: ClassVar[type[T]]

    def __init__(self, dataset: Dataset, settings: T | None = None) -> None:
        """
        Args:
            dataset: The dataset.
            settings: The settings.
                If `None`, use the default settings if any.

        Raises:
            ValueError: If the dataset is empty.
        """  # noqa: D205, D212, D415
        if dataset.empty:
            msg = "Dataset is empty."
            raise ValueError(msg)

        if settings is None:
            settings = self.settings_class()

        self.settings = settings
        self.__common_dataset = dataset
        self.__figure_file_paths = []
        self.__figures = []
        self.__name_to_label = {}
        self.__specific_data = self._create_specific_data_from_dataset()

    @property
    def dataset(self) -> Dataset:
        """The dataset."""
        return self.__common_dataset

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
                from `directory_path`, `file_name` and `file_format`.
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
            self.settings,
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
        """Pre-process the dataset specifically for this type of `DatasetPlot`.

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
                column[1], column[2], self.dataset.variable_name_to_n_components
            )
            new_columns.append(name)

        return new_columns

    @staticmethod
    def _get_component_name(
        name: str, component: int, name_to_size: Mapping[str, int]
    ) -> str:
        """Return the name of a variable component.

        Args:
            name: The name of the variable.
            component: The component of the variable.
            name_to_size: The sizes of the variables.

        Returns:
            The name of the variable component.
        """
        if name_to_size[name] == 1:
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
                or a tuple formatted as `(group_name, variable_name, component)`.

        Returns:
            The label related to a variable, e.g. "x[0]",
            as well as the refactored variable name, e.g. (group_name, "x", 0).
        """
        if isinstance(variable, str):
            return variable, (self.dataset.get_group_names(variable)[0], variable, 0)

        return repr_variable(variable[1], variable[2]), variable
