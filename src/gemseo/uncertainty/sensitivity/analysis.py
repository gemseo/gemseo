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
"""Abstract class for the computation and analysis of sensitivity indices.

The purpose of a sensitivity analysis is to
qualify or quantify how the model's uncertain inputs impact its outputs.

This analysis relies on :class:`.SensitivityAnalysis`
computed from a :class:`.MDODiscipline` representing the model,
a :class:`.ParameterSpace` describing the uncertain parameters
and options associated with a particular concrete class
inheriting from :class:`.SensitivityAnalysis` which is an abstract one.
"""
from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Sequence
from typing import Tuple
from typing import Union

from docstring_inheritance import GoogleDocstringInheritanceMeta
from matplotlib.figure import Figure
from numpy import array
from numpy import linspace
from numpy import ndarray
from numpy import vstack

from gemseo.algos.doe.doe_lib import DOELibraryOptionType
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.utils import get_all_outputs
from gemseo.post.dataset.bars import BarPlot
from gemseo.post.dataset.curves import Curves
from gemseo.post.dataset.dataset_plot import DatasetPlotPropertyType
from gemseo.post.dataset.radar_chart import RadarChart
from gemseo.post.dataset.surfaces import Surfaces
from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.file_path_manager import FileType
from gemseo.utils.matplotlib_figure import save_show_figure

OutputsType = Union[str, Tuple[str, int], Sequence[Union[str, Tuple[str, int]]]]
IndicesType = Dict[str, List[Dict[str, ndarray]]]


class SensitivityAnalysis(metaclass=GoogleDocstringInheritanceMeta):
    """Sensitivity analysis.

    The :class:`.SensitivityAnalysis` class provides both
    the values of :attr:`.SensitivityAnalysis.indices`
    and their graphical representations, from either
    the :meth:`.SensitivityAnalysis.plot` method,
    the :meth:`.SensitivityAnalysis.plot_radar` method or
    the :meth:`.SensitivityAnalysis.plot_bar` method.

    It is also possible to use :meth:`.SensitivityAnalysis.sort_parameters`
    to get the parameters sorted according to :attr:`.SensitivityAnalysis.main_method`.
    The :attr:`.SensitivityAnalysis.main_indices` are indices computed with the latter.

    Lastly, the :meth:`.SensitivityAnalysis.plot_comparison` method allows
    to compare the current :class:`.SensitivityAnalysis` with another one.
    """

    default_output: list[str]
    """The default outputs of interest."""

    dataset: Dataset
    """The dataset containing the discipline evaluations."""

    DEFAULT_DRIVER = None

    def __init__(
        self,
        disciplines: Collection[MDODiscipline],
        parameter_space: ParameterSpace,
        n_samples: int | None = None,
        output_names: Iterable[str] | None = None,
        algo: str | None = None,
        algo_options: Mapping[str, DOELibraryOptionType] | None = None,
        formulation: str = "MDF",
        **formulation_options: Any,
    ) -> None:
        """
        Args:
            disciplines: The discipline or disciplines to use for the analysis.
            parameter_space: A parameter space.
            n_samples: A number of samples.
                If ``None``, the number of samples is computed by the algorithm.
            output_names: The disciplines' outputs to be considered for the analysis.
                If ``None``, use all the outputs.
            algo: The name of the DOE algorithm.
                If ``None``, use the :attr:`.SensitivityAnalysis.DEFAULT_DRIVER`.
            algo_options: The options of the DOE algorithm.
            formulation: The name of the :class:`.MDOFormulation` to sample the disciplines.
            **formulation_options: The options of the :class:`.MDOFormulation`.
        """  # noqa: D205, D212, D415
        disciplines = list(disciplines)
        self._algo_name = algo or self.DEFAULT_DRIVER
        self._output_names = output_names or get_all_outputs(disciplines)
        self.default_output = self._output_names
        self.dataset = self.__sample_disciplines(
            disciplines,
            parameter_space,
            n_samples,
            algo_options or {},
            formulation,
            **(formulation_options or {}),
        ).export_to_dataset(opt_naming=False)
        self._main_method = self.__class__.__name__
        self._file_path_manager = FilePathManager(
            FileType.FIGURE,
            default_name=FilePathManager.to_snake_case(self.__class__.__name__),
        )

    def save(self, file_path: str | Path) -> None:
        """Save the current sensitivity analysis on the disk.

        Args:
            file_path: The path to the file.
        """
        with Path(file_path).open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: str | Path) -> SensitivityAnalysis:
        """Load a sensitivity analysis from the disk.

        Args:
            file_path: The path to the file.

        Returns:
            The sensitivity analysis.
        """
        with Path(file_path).open("rb") as f:
            return pickle.load(f)

    def __sample_disciplines(
        self,
        disciplines: Sequence[MDODiscipline],
        parameter_space: ParameterSpace,
        n_samples: int | None,
        algo_options: Mapping[str, DOELibraryOptionType],
        formulation: str,
        **formulation_options: Any,
    ) -> DOEScenario:
        """Sample the disciplines and return the scenario after evaluation.

        Args:
            disciplines: The disciplines to sample.
            parameter_space: A parameter space.
            n_samples: A number of samples.
                If ``None``, the number of samples is computed by the algorithm.
            algo_options: The options for the DOE algorithm.
            formulation: The name of the :class:`.MDOFormulation` to sample the disciplines.
            **formulation_options: The options of the :class:`.MDOFormulation`.

        Returns:
            The DOE scenario after evaluation.
        """
        scenario = self._create_scenario(
            disciplines,
            self._output_names,
            formulation,
            formulation_options,
            parameter_space,
        )
        scenario.execute(
            {
                "algo": self._algo_name,
                "n_samples": n_samples,
                "algo_options": algo_options,
            }
        )
        return scenario

    @staticmethod
    def _create_scenario(
        disciplines: Iterable[MDODiscipline],
        observable_names: Sequence[str],
        formulation: str,
        formulation_options: Mapping[str, Any],
        parameter_space: ParameterSpace,
    ) -> DOEScenario:
        """Create a DOE scenario to sample the disciplines.

        Args:
            disciplines: The disciplines to sample.
            observable_names: The names of the observables.
            formulation: The name of the :class:`.MDOFormulation` to sample the disciplines.
            formulation_options: The options of the :class:`.MDOFormulation`.
            parameter_space: A parameter space.

        Returns:
            The DOE scenario to be used to sample the disciplines.
        """
        scenario = DOEScenario(
            disciplines,
            formulation,
            observable_names[0],
            parameter_space,
            **formulation_options,
        )
        for discipline in disciplines:
            for output_name in discipline.get_output_data_names():
                if output_name in observable_names[1:]:
                    scenario.add_observable(output_name)
        return scenario

    @property
    def inputs_names(self) -> list[str]:
        """The names of the inputs."""
        return self.dataset.get_names(self.dataset.INPUT_GROUP)

    def compute_indices(
        self, outputs: Sequence[str] | None = None
    ) -> dict[str, IndicesType]:
        """Compute the sensitivity indices.

        Args:
            outputs: The outputs
                for which to display sensitivity indices.
                If None, use the default outputs, that are all the discipline outputs.

        Returns:
            The sensitivity indices.

            With the following structure:

            .. code-block:: python

                {
                    "method_name": {
                        "output_name": [
                            {
                                "input_name": data_array,
                            }
                        ]
                    }
                }
        """
        raise NotImplementedError

    @property
    def indices(self) -> dict[str, IndicesType]:
        """The sensitivity indices.

        With the following structure:

        .. code-block:: python

            {
                "method_name": {
                    "output_name": [
                        {
                            "input_name": data_array,
                        }
                    ]
                }
            }
        """
        raise NotImplementedError

    @property
    def main_method(self) -> str:
        """The name of the main method."""
        return self._main_method

    @main_method.setter
    def main_method(
        self,
        name: str,
    ) -> NoReturn:
        raise NotImplementedError("You cannot change the main method.")

    @property
    def main_indices(self) -> IndicesType:
        """The main sensitivity indices.

        With the following structure:

        .. code-block:: python

            {
                "output_name": [
                    {
                        "input_name": data_array,
                    }
                ]
            }
        """
        raise NotImplementedError

    def _outputs_to_tuples(
        self,
        outputs: OutputsType,
    ) -> list[tuple[str, int]]:
        """Convert the outputs to a list of tuple(str,int).

        Args:
            outputs: The outputs
                for which to display sensitivity indices,
                either a name,
                a list of names,
                a (name, component) tuple,
                a list of such tuples or
                a list mixing such tuples and names.
                When a name is specified, all its components are considered.
                If None, use the default outputs.

        Returns:
            The outputs.

            The outputs are formatted as tuples of the form (name, component),
            where name is the output name and component is the output component.
        """
        if not isinstance(outputs, list):
            outputs = [outputs]

        def get_all(output):
            return [(output, index) for index in range(len(self.main_indices[output]))]

        result = [
            output if isinstance(output, tuple) else get_all(output)
            for output in outputs
        ]
        return [item for sublist in result for item in sublist]

    def sort_parameters(self, output: str | tuple[str, int]) -> list[str]:
        """Return the parameters sorted in descending order.

        Args:
            output: An output of the form :code:`(name, component)`,
                where name is the output name and component is the output component.
                If a string is passed,
                the tuple :code:`(name, 0)` will be considered
                corresponding to the first component of the output :code:`name`.

        Returns:
            The input parameters sorted in descending order.
        """
        if not isinstance(output, tuple):
            output = (output, 0)
        output_name, output_component = output
        indices = self.main_indices[output_name][output_component]
        names = [
            name
            for name, _ in sorted(
                list(indices.items()), key=lambda item: item[1].sum(), reverse=True
            )
        ]
        return names

    def plot(
        self,
        output: str | tuple[str, int],
        inputs: Iterable[str] | None = None,
        title: str | None = None,
        save: bool = True,
        show: bool = False,
        file_path: str | Path | None = None,
        file_format: str | None = None,
    ) -> None:
        """Plot the sensitivity indices.

        Args:
            output: The output
                for which to display sensitivity indices,
                either a name or a tuple of the form (name, component).
                If name, its first component is considered.
            inputs: The inputs to display. If None, display all.
            title: The title of the plot. If None, no title.
            save: If True, save the figure.
            show: If True, show the figure.
            file_path: A file path.
                Either a complete file path, a directory name or a file name.
                If None, use a default file name and a default directory.
                The file extension is inferred from filepath extension, if any.
            file_format: A file format, e.g. 'png', 'pdf', 'svg', ...
                Used when ``file_path`` does not have any extension.
                If None, use a default file extension.
        """
        raise NotImplementedError

    def plot_field(
        self,
        output: str | tuple[str, int],
        mesh: ndarray | None = None,
        inputs: Iterable[str] | None = None,
        standardize: bool = False,
        title: str | None = None,
        save: bool = True,
        show: bool = False,
        file_path: str | Path | None = None,
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_format: str | None = None,
        properties: Mapping[str, DatasetPlotPropertyType] = None,
    ) -> Curves | Surfaces:
        """Plot the sensitivity indices related to a 1D or 2D functional output.

        The output is considered as a 1D or 2D functional variable,
        according to the shape of the mesh on which it is represented.

        Args:
            output: The output
                for which to display sensitivity indices,
                either a name or a tuple of the form (name, component)
                where (name, component) is used to sort the inputs.
                If it is a name, its first component is considered.
            mesh: The mesh on which the p-length output
                is represented. Either a p-length array for a 1D functional output
                or a (p, 2) array for a 2D one. If None, assume a 1D functional output.
            inputs: The inputs to display. If None, display all inputs.
            standardize: If True, standardize the indices between 0 and 1 for each output.
            title: The title of the plot. If None, no title is displayed.
            save: If True, save the figure.
            show: If True, show the figure.
            file_path: The path of the file to save the figures.
                If None,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_extension``.
            directory_path: The path of the directory to save the figures.
                If None, use the current working directory.
            file_name: The name of the file to save the figures.
                If None, use a default one generated by the post-processing.
            file_format: A file extension, e.g. 'png', 'pdf', 'svg', ...
                If None, use a default file extension.
            properties: The general properties of a :class:`.DatasetPlot`.

        Returns:
            A bar plot representing the sensitivity indices.

        Raises:
            NotImplementedError: If the dimension of the mesh is greater than 2.
        """
        if isinstance(output, str):
            output_name = output
            output_component = 0
        else:
            output_name, output_component = output

        dataset = Dataset()
        inputs_names = self._sort_and_filter_input_parameters(
            (output_name, output_component), inputs
        )
        if standardize:
            main_indices = self.standardize_indices(self.main_indices)
        else:
            main_indices = self.main_indices

        data = []
        for input_name in inputs_names:
            data.append(
                [main_index[input_name] for main_index in main_indices[output_name]]
            )

        data = array(data)[:, :, 0]
        dataset.set_from_array(data, [output_name], sizes={output_name: data.shape[1]})
        dataset.row_names = inputs_names
        mesh = linspace(0, 1, data.shape[1]) if mesh is None else mesh
        dataset.set_metadata("mesh", mesh)
        mesh_dimension = len(dataset.metadata["mesh"].shape)
        if mesh_dimension == 1:
            plot = Curves(dataset, mesh="mesh", variable=output_name)
            plot.title = title
        elif mesh_dimension == 2:
            plot = Surfaces(dataset, mesh="mesh", variable=output_name)
        else:
            raise NotImplementedError

        plot.execute(
            save=save,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
            properties=properties,
        )
        return plot

    def plot_bar(
        self,
        outputs: OutputsType,
        inputs: Iterable[str] | None = None,
        standardize: bool = False,
        title: str | None = None,
        save: bool = True,
        show: bool = False,
        file_path: str | Path | None = None,
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_format: str | None = None,
        **options: int,
    ) -> BarPlot:
        """Plot the sensitivity indices on a bar chart.

        This method may consider one or more outputs,
        as well as all inputs (default behavior) or a subset.

        Args:
            outputs: The outputs
                for which to display sensitivity indices,
                either a name,
                a list of names,
                a (name, component) tuple,
                a list of such tuples or
                a list mixing such tuples and names.
                When a name is specified, all its components are considered.
                If None, use the default outputs.
            inputs: The inputs to display. If None, display all.
            standardize: If True, standardize the indices between 0 and 1 for each output.
            title: The title of the plot. If None, no title.
            save: If True, save the figure.
            show: If True, show the figure.
            file_path: The path of the file to save the figures.
                If the extension is missing, use ``file_extension``.
                If None,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_extension``.
            directory_path: The path of the directory to save the figures.
                If None, use the current working directory.
            file_name: The name of the file to save the figures.
                If None, use a default one generated by the post-processing.
            file_format: A file extension, e.g. 'png', 'pdf', 'svg', ...
                If None, use a default file extension.

        Returns:
            A bar chart representing the sensitivity indices.
        """
        outputs = self._outputs_to_tuples(outputs)
        dataset = Dataset()
        inputs_names = self._sort_and_filter_input_parameters(outputs[0], inputs)
        data = {name: [] for name in inputs_names}
        if standardize:
            main_indices = self.standardize_indices(self.main_indices)
        else:
            main_indices = self.main_indices

        for output in outputs:
            for name in inputs_names:
                data[name].append(main_indices[output[0]][output[1]][name])

        for name in inputs_names:
            dataset.add_variable(name, vstack(data[name]))

        dataset.row_names = [f"{output[0]}({output[1]})" for output in outputs]
        plot = BarPlot(dataset)
        plot.title = title
        plot.execute(
            save=save,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
            **options,
        )
        return plot

    def plot_radar(
        self,
        outputs: OutputsType,
        inputs: Iterable[str] | None = None,
        standardize: bool = False,
        title: str | None = None,
        save: bool = True,
        show: bool = False,
        file_path: str | Path | None = None,
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_format: str | None = None,
        min_radius: float | None = None,
        max_radius: float | None = None,
        **options: bool,
    ) -> RadarChart:
        """Plot the sensitivity indices on a radar chart.

        This method may consider one or more outputs,
        as well as all inputs (default behavior) or a subset.

        For visualization purposes,
        it is also possible to change the minimum and maximum radius values.

        Args:
            outputs: The outputs
                for which to display sensitivity indices,
                either a name,
                a list of names,
                a (name, component) tuple,
                a list of such tuples or
                a list mixing such tuples and names.
                When a name is specified, all its components are considered.
                If None, use the default outputs.
            inputs: The inputs to display.
                If None, display all.
            standardize: If True, standardize the indices between 0 and 1 for each output.
            title: The title of the plot. If None, no title.
            save: If True, save the figure.
            show: If True, show the figure.
            file_path: The path of the file to save the figures.
                If the extension is missing, use ``file_extension``.
                If None,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_extension``.
            directory_path: The path of the directory to save the figures.
                If None, use the current working directory.
            file_name: The name of the file to save the figures.
                If None, use a default one generated by the post-processing.
            file_format: A file extension, e.g. 'png', 'pdf', 'svg', ...
                If None, use a default file extension.
            min_radius: The minimal radial value. If None, from data.
            max_radius: The maximal radial value. If None, from data.

        Returns:
            A radar chart representing the sensitivity indices.
        """
        outputs = self._outputs_to_tuples(outputs)
        dataset = Dataset()
        inputs_names = self._sort_and_filter_input_parameters(outputs[0], inputs)
        data = {name: [] for name in inputs_names}
        if standardize:
            main_indices = self.standardize_indices(self.main_indices)
        else:
            main_indices = self.main_indices

        for output in outputs:
            for name, value in main_indices[output[0]][output[1]].items():
                if name in inputs_names:
                    data[name].append(value)

        for name in inputs_names:
            dataset.add_variable(name, vstack(data[name]))

        dataset.row_names = [f"{output[0]}({output[1]})" for output in outputs]
        plot = RadarChart(dataset)
        plot.title = title
        plot.rmin = min_radius
        plot.rmax = max_radius
        plot.execute(
            save=save,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
            **options,
        )
        return plot

    @staticmethod
    def _filter_names(
        names: Iterable[str],
        names_to_keep: Iterable[str],
    ) -> list[str]:
        """Sort and filter the names.

        Args:
            names: The original names.
            names_to_keep: The names to keep. If None, keep all.

        Returns:
            The filtered names.
        """
        if names_to_keep is not None:
            names = [item for item in names if item in set(names_to_keep)]
        return names

    def _sort_and_filter_input_parameters(
        self,
        output: tuple[str, int],
        inputs_to_keep: Iterable[str],
    ) -> list[str]:
        """Sort and filter the input parameters.

        Args:
            output: An output for which to display sensitivity indices.
            inputs_to_keep: The inputs to keep. If None, keep all.

        Returns:
            The filtered names.
        """
        inputs = self.sort_parameters((output[0], output[1]))
        return self._filter_names(inputs, inputs_to_keep)

    def plot_comparison(
        self,
        indices: list[SensitivityAnalysis],
        output: str | tuple[str, int],
        inputs: Iterable[str] | None = None,
        title: str | None = None,
        use_bar_plot: bool = True,
        save: bool = True,
        show: bool = False,
        file_path: str | Path | None = None,
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_format: str | None = None,
        **options: bool,
    ) -> BarPlot | RadarChart:
        """Plot a comparison between the current sensitivity indices and other ones.

        This method allows to use either a bar chart (default option) or a radar one.

        Args:
            indices: The sensitivity indices.
            output: The output
                for which to display sensitivity indices,
                either a name or a tuple of the form (name, component).
                If name, its first component is considered.
            inputs: The inputs to display. If None, display all.
            title: The title of the plot. If None, no title.
            use_bar_plot: The type of graph.
                If True, use a bar plot. Otherwise, use a radar chart.
            save: If True, save the figure.
            show: If True, show the figure.
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
            **options: The options passed to the underlying :class:`.DatasetPlot`.

        Returns:
            A graph comparing sensitivity indices.
        """
        if not isinstance(output, tuple):
            output = (output, 0)
        if isinstance(indices, SensitivityAnalysis):
            indices = [indices]
        methods = [self] + indices
        dataset = Dataset()
        inputs_names = self._sort_and_filter_input_parameters(output, inputs)
        for name in inputs_names:
            data = abs(
                array(
                    [
                        method.main_indices[output[0]][output[1]][name]
                        for method in methods
                    ]
                )
            )
            dataset.add_variable(name, data)
        data = dataset.data[dataset.PARAMETER_GROUP]
        dataset.data[dataset.PARAMETER_GROUP] = data / data.max(axis=1)[:, None]
        dataset.row_names = [method.main_method for method in methods]
        if use_bar_plot:
            plot = BarPlot(dataset)
        else:
            plot = RadarChart(dataset)
            plot.rmin = 0.0
            plot.rmax = 1.0
        plot.title = title
        plot.execute(
            save, show, file_path, directory_path, file_name, file_format, **options
        )
        return plot

    def _save_show_plot(
        self,
        fig: Figure,
        save: bool = True,
        show: bool = False,
        file_path: str | Path | None = None,
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_format: str | None = None,
    ) -> Figure:
        """Save or show the plot.

        Args:
            fig: The figure to be processed.
            save: If True, save the figure.
            show: If True, show the figure.
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

        Returns:
            The figure.
        """
        if save:
            file_path = self._file_path_manager.create_file_path(
                file_path=file_path,
                directory_path=directory_path,
                file_name=file_name,
                file_extension=file_format,
            )
        else:
            file_path = None

        save_show_figure(fig, show, file_path)
        return fig

    def export_to_dataset(self) -> Dataset:
        """Convert :attr:`.SensitivityAnalysis.indices` into a :class:`.Dataset`.

        Returns:
            Dataset: The sensitivity indices.
        """
        sizes = self.dataset.sizes

        rows_names = []
        for input_name in self.inputs_names:
            for input_component in range(sizes[input_name]):
                rows_names.append(f"{input_name}({input_component})")

        dataset = Dataset(by_group=False)
        for method, indices in self.indices.items():
            variables = []
            sizes = {}
            data = []
            for output, components in indices.items():
                variables.append(output)
                sizes[output] = len(components)
                for component in components:
                    data.append(
                        [component[name].tolist() for name in self.inputs_names]
                    )
                    data[-1] = [item for sublist in data[-1] for item in sublist]
            data = array(data).T
            dataset.add_group(method, data, variables, sizes)
        dataset.row_names = rows_names
        return dataset

    @staticmethod
    def standardize_indices(
        indices: IndicesType,
    ) -> IndicesType:
        """Standardize the sensitivity indices for each output component.

        Each index is replaced by its absolute value divided by the largest index.
        Thus, the standardized indices belong to the interval :math:`[0,1]`.

        Args:
            indices: The indices to be standardized.

        Returns:
            The standardized indices.
        """
        new_indices = deepcopy(indices)
        for output_name, output_indices in indices.items():
            for output_component, output_component_indices in enumerate(output_indices):
                max_value = max(
                    abs(value)[0] for value in output_component_indices.values()
                )

                for input_name, input_indices in output_component_indices.items():
                    new_indices[output_name][output_component][input_name] = (
                        abs(input_indices) / max_value
                    )

        return new_indices
