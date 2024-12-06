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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base class for optimization history post-processing."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import TypeVar
from typing import Union

from gemseo.post.base_post_settings import BasePostSettings
from gemseo.post.dataset.dataset_plot import DatasetPlot
from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.matplotlib_figure import FigSizeType
from gemseo.utils.matplotlib_figure import save_show_figure
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.pydantic import create_model
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from gemseo.algos.database import Database
    from gemseo.algos.optimization_problem import OptimizationProblem

BasePostOptionType = Union[int, float, str, bool, Sequence[str], FigSizeType]


T = TypeVar("T", bound=BasePostSettings)


class BasePost(Generic[T], metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base class for optimization post-processing."""

    # Silencing mypy since the root cause does not seem legit,
    # and may be changed.
    # See https://github.com/python/mypy/issues/5144.
    Settings: ClassVar[type[T]]
    """The Pydantic model for the settings."""

    optimization_problem: OptimizationProblem
    """The optimization problem."""

    database: Database
    """The database generated by the optimization problem."""

    materials_for_plotting: dict[Any, Any]
    """The materials to eventually rebuild the plot in another framework."""

    _obj_name: str
    """The name of the objective function as passed by the user."""

    _standardized_obj_name: str
    """The name of the objective function stored in the database."""

    _neg_obj_name: str
    """The name of the objective function starting with a '-'."""

    _output_file_paths: list[Path]
    """Paths to the output files."""

    __figures: dict[str, Figure | DatasetPlot]
    """The mapping from figure names or nameless figure counters to figures."""

    def __init__(
        self,
        opt_problem: OptimizationProblem,
    ) -> None:
        """
        Args:
            opt_problem: The optimization problem to be post-processed.
        """  # noqa: D205, D212, D415
        self.optimization_problem = opt_problem
        self._obj_name = opt_problem.objective_name
        self._standardized_obj_name = opt_problem.standardized_objective_name
        self._neg_obj_name = f"-{self._obj_name}"
        self.database = opt_problem.database
        # The data required to eventually rebuild the plot in another framework.
        self.materials_for_plotting = {}
        default_file_name = FilePathManager.to_snake_case(self.__class__.__name__)
        self.__file_path_manager = FilePathManager(
            FilePathManager.FileType.FIGURE, default_file_name
        )
        self._output_file_paths = []
        self.__figures = {}
        self.__nameless_figure_counter = 0
        self._dataset_plots = []

    @property
    def _change_obj(self) -> bool:
        """Whether to change the objective value and names by using the opposite."""
        return not (
            self.optimization_problem.minimize_objective
            or self.optimization_problem.use_standardized_objective
        )

    @property
    def figures(self) -> dict[str, Figure | DatasetPlot]:
        """The figures indexed by a name, or the nameless figure counter."""
        return self.__figures

    @property
    def output_file_paths(self) -> list[Path]:
        """The paths to the output files."""
        return self._output_file_paths

    def _add_figure(
        self,
        figure: Figure | DatasetPlot,
        file_name: str = "",
    ) -> None:
        """Add a figure.

        Args:
            figure: The figure to be added.
            file_name: The default name of the file to save the figure.
                If empty, use the nameless figure counter.
        """
        if not file_name:
            self.__nameless_figure_counter += 1
            file_name = str(self.__nameless_figure_counter)

        self.__figures[file_name] = figure

    def execute(
        self, settings_model: BasePostSettings | None = None, **settings: Any
    ) -> dict[str, Figure | DatasetPlot]:
        """Post-process the optimization problem.

        Args:
            settings_model: The post-processor settings as a Pydantic model.
                If ``None``, use ``**settings``.
            **settings: The post-processor settings.
                This argument is ignored when ``settings_model`` is not ``None``.

        Returns:
            The figures, to be customized;
            in the case of a matplotlib ``Figure``, it must not be closed.

        Raises:
            ValueError: If the ``opt_problem.database`` is empty.
        """
        if not self.optimization_problem.database:
            msg = (
                f"The post-processor {self.__class__.__name__} cannot be solved "
                "because the optimization problem was not solved."
            )
            raise ValueError(msg)

        settings_ = create_model(
            self.Settings, settings_model=settings_model, **settings
        )
        self._plot(settings_)
        self.__render(settings_)
        return self.__figures

    def __render(self, settings: T) -> None:
        """Render the figures.

        Args:
            settings: The rendering settings.
        """
        file_extension = settings.file_extension
        file_path = self.__file_path_manager.create_file_path(
            file_path=settings.file_path,
            directory_path=settings.directory_path,
            file_name=settings.file_name,
            file_extension=file_extension,
        )
        file_extension = file_path.suffix[1:]
        for figure_name, figure in self.__figures.items():
            fig_file_path: str | Path
            if settings.save:
                if len(self.__figures) > 1:
                    fig_file_path = self.__file_path_manager.add_suffix(
                        file_path, figure_name
                    )
                else:
                    fig_file_path = file_path

                self._output_file_paths.append(fig_file_path)
            else:
                fig_file_path = ""

            if isinstance(figure, DatasetPlot):
                figure.fig_size = settings.fig_size
                figure.execute(
                    save=settings.save,
                    show=settings.show,
                    file_format=file_extension,
                    file_path=fig_file_path,
                )
            else:
                save_show_figure(
                    figure, settings.show, fig_file_path, settings.fig_size
                )

    @abstractmethod
    def _plot(self, settings: T) -> None:
        """Create the figures.

        Args:
            settings: The settings of the post-processor.
        """

    def _get_design_variable_names(
        self,
        variables: Iterable[str] = (),
        simplify: bool = False,
    ) -> list[str]:
        """Create the names of the components of design variables as ``"name[i]"``.

        Args:
            variables: The design variables of interest.
                If empty, use all the design variables.
            simplify: Whether to use ``"[i]"`` when ``i>0`` instead of ``"name[i]"``.

        Returns:
            The names of the components of the design variables.
        """
        if not variables:
            variables = self.optimization_problem.design_space.variable_names

        design_variable_names = []
        design_variable_sizes = self.optimization_problem.design_space.variable_sizes
        for variable in variables:
            design_variable_size = design_variable_sizes[variable]
            design_variable_names.extend([
                repr_variable(
                    variable, index, size=design_variable_size, simplify=simplify
                )
                for index in range(design_variable_size)
            ])

        return design_variable_names
