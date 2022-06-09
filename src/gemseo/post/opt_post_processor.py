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

import inspect
from os.path import abspath
from os.path import dirname
from os.path import exists
from os.path import join
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from docstring_inheritance import GoogleDocstringInheritanceMeta
from matplotlib.figure import Figure

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.grammars.errors import InvalidDataException
from gemseo.core.json_grammar import JSONGrammar
from gemseo.post.dataset.dataset_plot import DatasetPlot
from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.file_path_manager import FileType
from gemseo.utils.matplotlib_figure import save_show_figure
from gemseo.utils.source_parsing import get_options_doc

OptPostProcessorOptionType = Union[int, float, str, bool, Sequence[str]]
PlotOutputType = List[
    Tuple[Optional[str], Union[Figure, DatasetPlot], Optional[Dict[str, Sequence[str]]]]
]


class OptPostProcessor(metaclass=GoogleDocstringInheritanceMeta):
    """Abstract class for optimization post-processing methods.

    Attributes:
        opt_problem (OptimizationProblem): The optimization problem.
        database (Database): The database generated by the optimization problem.
        materials_for_plotting (Dict[Any,Any]): The materials
            to eventually rebuild the plot in another framework.
    """

    DEFAULT_FIG_SIZE = (11.0, 11.0)
    """tuple(float, float): The default width and height of the figure, in inches."""

    def __init__(
        self,
        opt_problem: OptimizationProblem,
    ) -> None:
        """
        Args:
            opt_problem: The optimization problem to be post-processed.

        Raises:
            ValueError: If the JSON grammar file
                for the options of the post-processor does not exist.
        """
        self.opt_problem = opt_problem
        self.database = opt_problem.database
        comp_dir = abspath(dirname(inspect.getfile(OptPostProcessor)))
        self.opt_grammar = JSONGrammar("OptPostProcessor")
        self.opt_grammar.update_from_file(join(comp_dir, "OptPostProcessor.json"))
        self.opt_grammar.set_descriptions(get_options_doc(self.execute))

        cls_name = self.__class__.__name__
        name = cls_name + "_options"
        f_class = inspect.getfile(self.__class__)
        comp_dir = abspath(dirname(f_class))
        schema_file = join(comp_dir, f"{name}.json")
        if not exists(schema_file):
            schema_file = join(comp_dir, "options", f"{name}.json")
        if not exists(schema_file):
            raise ValueError(
                "Options grammar for optimization post-processor does not exist, "
                "expected: {} or {}".format(schema_file, join(comp_dir, name + ".json"))
            )

        descriptions = {}

        if hasattr(self.__class__, "_run"):
            descriptions.update(get_options_doc(self.__class__._run))

        if hasattr(self.__class__, "_plot"):
            descriptions.update(get_options_doc(self.__class__._plot))

        self.opt_grammar.update_from_file(schema_file)
        self.opt_grammar.set_descriptions(descriptions)

        # The data required to eventually rebuild the plot in another framework.
        self.materials_for_plotting = {}
        default_file_name = FilePathManager.to_snake_case(self.__class__.__name__)
        self.__file_path_manager = FilePathManager(FileType.FIGURE, default_file_name)
        self.__output_files = []
        self.__figures = {}
        self.__nameless_figure_counter = 0

    @property
    def figures(self) -> dict[str, Figure]:
        """The Matplotlib figures indexed by a name, or the nameless figure counter."""
        return self.__figures

    @property
    def output_files(self) -> list[str]:
        """The paths to the output files."""
        return self.__output_files

    def _add_figure(
        self,
        figure: Figure,
        file_name: str | None = None,
    ) -> None:
        """Add a figure.

        Args:
            figure: The figure to be added.
            file_name: The default name of the file to save the figure.
                If None, use the nameless figure counter.
        """
        if file_name is None:
            self.__nameless_figure_counter += 1
            file_name = str(self.__nameless_figure_counter)

        self.__figures[file_name] = figure

    def execute(
        self,
        save: bool = True,
        show: bool = False,
        file_path: str | Path | None = None,
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_extension: str | None = None,
        fig_size: tuple[float, float] | None = None,
        **options: OptPostProcessorOptionType,
    ) -> dict[str, Figure]:
        """Post-process the optimization problem.

        Args:
            save: If True, save the figure.
            show: If True, display the figure.
            file_path: The path of the file to save the figures.
                If the extension is missing, use ``file_extension``.
                If None,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_extension``.
            directory_path: The path of the directory to save the figures.
                If None, use the current working directory.
            file_name: The name of the file to save the figures.
                If None, use a default one generated by the post-processing.
            file_extension: A file extension, e.g. 'png', 'pdf', 'svg', ...
                If None, use a default file extension.
            fig_size: The width and height of the figure in inches, e.g. `(w, h)`.
                If None, use the :attr:`.OptPostProcessor.DEFAULT_FIG_SIZE`
                of the post-processor.
            **options: The options of the post-processor.

        Returns:
            The figures, to be customized if not closed.

        Raises:
            ValueError: If the `opt_problem.database` is empty.
        """
        # convert file_path to string before grammar-based options checking
        if isinstance(file_path, Path):
            file_path_to_be_checked = str(file_path)
        else:
            file_path_to_be_checked = file_path

        if isinstance(directory_path, Path):
            directory_path_to_be_checked = str(directory_path)
        else:
            directory_path_to_be_checked = directory_path

        if file_path is not None:
            file_path = Path(file_path)

        if directory_path is not None:
            directory_path = Path(directory_path)

        self.check_options(
            save=save,
            show=show,
            file_path=file_path_to_be_checked,
            file_name=file_name,
            directory_path=directory_path_to_be_checked,
            file_extension=file_extension,
            fig_size=fig_size,
            **options,
        )
        if not self.opt_problem.database:
            raise ValueError(
                "Optimization problem was not solved, "
                "cannot run post processing {}".format(self.__class__.__name__)
            )

        self.__figures = self._run(
            save=save,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_extension=file_extension,
            directory_path=directory_path,
            **options,
        )
        return self.__figures

    def check_options(self, **options: OptPostProcessorOptionType) -> None:
        """Check the options of the post-processor.

        Args:
            **options: The options of the post-processor.

        Raises:
            InvalidDataException: If an option is invalid according to the grammar.
        """
        try:
            self.opt_grammar.validate(options)
        except InvalidDataException:
            raise InvalidDataException(
                "Invalid options for post-processor {}; "
                "got: {}".format(self.__class__.__name__, options)
            )

    def _run(
        self,
        save: bool = True,
        show: bool = False,
        file_path: Path | None = None,
        directory_path: Path | None = None,
        file_name: str | None = None,
        file_extension: str | None = None,
        fig_size: tuple[float, float] | None = None,
        **options: OptPostProcessorOptionType,
    ) -> dict[str, Figure]:
        """Run the post-processor.

        Args:
            save: If True, save the figure.
            show: If True, display the figure.
            file_path: The path of the file to save the figures.
                If the extension is missing, use ``file_extension``.
                If None,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_extension``.
            directory_path: The path of the directory to save the figures.
                If None, use the current working directory.
            file_name: The name of the file to save the figures.
                If None, use a default one generated by the post-processing.
            file_extension: A file extension, e.g. 'png', 'pdf', 'svg', ...
                If None, use a default file extension.
            fig_size: The width and height of the figure in inches, e.g. `(w, h)`.
                If None, use the :attr:`.OptPostProcessor.DEFAULT_FIG_SIZE`
                of the post-processor.
            **options: The options of the post-processor.

        Returns:
            The figures resulting from the post-processing of the optimization problem.
        """
        self._plot(**options)

        file_path = self.__file_path_manager.create_file_path(
            file_path=file_path,
            directory_path=directory_path,
            file_name=file_name,
            file_extension=file_extension,
        )

        for figure_name, figure in self.__figures.items():
            if save:
                if len(self.__figures) > 1:
                    fig_file_path = self.__file_path_manager.add_suffix(
                        file_path, figure_name
                    )
                else:
                    fig_file_path = file_path

                self.__output_files.append(str(fig_file_path))

            else:
                fig_file_path = None

            save_show_figure(figure, show, fig_file_path, fig_size)

        return self.__figures

    def _plot(self, **options: OptPostProcessorOptionType) -> None:
        """Create the figures.

        Args:
            **options: The post-processor options.
        """
        raise NotImplementedError()

    def _generate_x_names(self, variables: Iterable[str] | None = None) -> list[str]:
        """Create the design variables names for the plot.

        Args:
            variables: The variables to create the names. If None, use all
                the design variables.

        Returns:
            The design variables names.
        """
        if not variables:
            variables = self.opt_problem.get_design_variable_names()

        x_names = []
        for d_v in variables:
            dv_size = self.opt_problem.design_space.variables_sizes[d_v]
            if dv_size == 1:
                x_names.append(d_v)
            else:
                for k in range(dv_size):
                    x_names.append(f"{d_v}_{k}")
        return x_names
