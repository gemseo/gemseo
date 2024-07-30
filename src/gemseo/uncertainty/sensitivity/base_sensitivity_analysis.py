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
"""Base class for the computation and analysis of sensitivity indices."""

from __future__ import annotations

import pickle
from abc import abstractmethod
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Union

from numpy import array
from numpy import hstack
from numpy import linspace
from numpy import newaxis
from numpy import vstack
from pandas import MultiIndex
from strenum import StrEnum

from gemseo import sample_disciplines
from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset
from gemseo.disciplines.utils import get_all_outputs
from gemseo.post.dataset.bars import BarPlot
from gemseo.post.dataset.curves import Curves
from gemseo.post.dataset.radar_chart import RadarChart
from gemseo.post.dataset.surfaces import Surfaces
from gemseo.typing import RealArray
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.string_tools import convert_strings_to_iterable
from gemseo.utils.string_tools import filter_names
from gemseo.utils.string_tools import get_name_and_component
from gemseo.utils.string_tools import get_variables_with_components
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from gemseo.algos.base_driver_library import DriverLibraryOptionType
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline import MDODiscipline
    from gemseo.post.dataset.dataset_plot import DatasetPlot
    from gemseo.post.dataset.dataset_plot import DatasetPlotPropertyType
    from gemseo.scenarios.backup_settings import BackupSettings
    from gemseo.utils.string_tools import VariableType

OutputsType = Union[str, tuple[str, int], Sequence[Union[str, tuple[str, int]]]]
FirstOrderIndicesType = dict[str, list[dict[str, RealArray]]]
SecondOrderIndicesType = dict[str, list[dict[str, dict[str, RealArray]]]]


class BaseSensitivityAnalysis(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base class for sensitivity analysis.

    A sensitivity analysis aims to qualify or quantify
    how the model's uncertain input variables impact its output variables
    from input-output samples relying on a specific design of experiments (DOE).

    A :class:`.BaseSensitivityAnalysis` can be created from such samples
    (passed as an :class:`.IODataset`)
    or use its :meth:`.compute_samples` method to generate them,
    using a :class:`.MDODiscipline` representing the model,
    a :class:`.ParameterSpace` describing the uncertain input variables
    and a set of options.
    In the second case,
    the samples returned by :meth:`.compute_samples` can be saved on the disk
    for future use.
    """

    dataset: IODataset | None
    """The dataset containing the discipline evaluations.

    The samples must be
    either passed at instantiation
    or generated with :meth:`.compute_samples`.
    """

    class Method(StrEnum):
        """The names of the sensitivity methods."""

        NONE = "none"

    _INTERACTION_METHODS: ClassVar[tuple[str]] = ()
    """The names of the sensitivity methods considering interaction effects."""

    DEFAULT_DRIVER: ClassVar[str] = ""

    _DEFAULT_MAIN_METHOD: ClassVar[Method] = Method.NONE
    """The name of the default main sensitivity analysis method."""

    _input_names: list[str]
    """The names of the inputs in parameter space order."""

    _output_names: list[str]
    """The disciplines' outputs to be considered for the analysis."""

    _algo_name: str
    """The name of the DOE algorithm to sample the discipline."""

    _file_path_manager: FilePathManager
    """The file path manager for the figures."""

    main_method: Method
    """The name of the main sensitivity analysis method."""

    @dataclass(frozen=True)
    class SensitivityIndices:
        """The sensitivity indices.

        Given a sensitivity method, an input variable and an output variable,
        the sensitivity index which is a 1D NumPy array can be accessed through
        ``indices.method_name[output_name][output_component][input_name]``.
        """

    _indices: SensitivityIndices
    """The sensitivity indices computed by the :meth:`.compute_indices` method."""

    def __init__(self, samples: IODataset | str | Path | None = None) -> None:
        """
        Args:
            samples: The samples for the estimation of the sensitivity indices,
                either as an :class:`.IODataset`
                or as a pickle file path generated from
                the :class:`.IODataset.to_pickle` method.
                If ``None``, use :meth:`.compute_samples`.
        """  # noqa: D202, D205, D212
        if isinstance(samples, IODataset):
            self.dataset = samples
        elif samples not in {None, ""}:
            with Path(samples).open("rb") as f:
                samples = self.dataset = pickle.load(f)
        else:
            self.dataset = None

        self._algo_name = ""
        self._file_path_manager = FilePathManager(
            FilePathManager.FileType.FIGURE,
            default_name=FilePathManager.to_snake_case(self.__class__.__name__),
        )
        self.main_method = self._DEFAULT_MAIN_METHOD
        if samples is None:
            self._input_names = []
            self._output_names = []
        else:
            self._input_names = samples.input_names
            self._output_names = samples.output_names
        self._indices = self.SensitivityIndices()

    def compute_samples(
        self,
        disciplines: Collection[MDODiscipline],
        parameter_space: ParameterSpace,
        n_samples: int | None,
        output_names: Iterable[str] = (),
        algo: str = "",
        algo_options: Mapping[str, DriverLibraryOptionType] = READ_ONLY_EMPTY_DICT,
        backup_settings: BackupSettings | None = None,
        formulation: str = "MDF",
        **formulation_options: Any,
    ) -> IODataset:
        """Compute the samples for the estimation of the sensitivity indices.

        Args:
            disciplines: The discipline or disciplines to use for the analysis.
            parameter_space: A parameter space.
            n_samples: A number of samples.
                If ``None``, the number of samples is computed by the algorithm.
            output_names: The disciplines' outputs to be considered for the analysis.
                If empty, use all the outputs.
            algo: The name of the DOE algorithm.
                If empty, use the :attr:`.BaseSensitivityAnalysis.DEFAULT_DRIVER`.
            algo_options: The options of the DOE algorithm.
            backup_settings: The settings of the backup file to store the evaluations
                if any.
            formulation: The name of the :class:`.BaseMDOFormulation`
                to sample the disciplines.
            **formulation_options: The options of the :class:`.BaseMDOFormulation`.

        Returns:
            The samples for the estimation of the sensitivity indices.
        """  # noqa: D205, D212, D415
        disciplines = list(disciplines)
        self._algo_name = algo or self.DEFAULT_DRIVER
        self._output_names = list(output_names or get_all_outputs(disciplines))
        self._input_names = parameter_space.variable_names
        algo_options = dict(algo_options)
        algo_options["use_one_line_progress_bar"] = True
        self.dataset = sample_disciplines(
            disciplines,
            parameter_space,
            self._output_names,
            n_samples,
            self._algo_name,
            formulation=formulation,
            formulation_options=formulation_options or {},
            name=f"{self.__class__.__name__}SamplingPhase",
            backup_settings=backup_settings,
            **algo_options,
        )
        return self.dataset

    @property
    def default_output_names(self) -> list[str]:
        """The default outputs of interest."""
        return self._output_names

    @property
    def input_names(self) -> list[str]:
        """The names of the inputs."""
        return self._input_names

    @abstractmethod
    def compute_indices(
        self, output_names: str | Iterable[str] = ()
    ) -> dict[str, FirstOrderIndicesType | SecondOrderIndicesType]:
        """Compute the sensitivity indices.

        Args:
            output_names: The name(s) of the output(s)
                for which to compute the sensitivity indices.
                If empty,
                use the names of the outputs set at instantiation.

        Returns:
            The sensitivity indices.

            Given a sensitivity method, an input variable and an output variable,
            the sensitivity index which is a 1D NumPy array can be accessed through
            ``indices.method_name[output_name][output_component][input_name]``.
        """

    @property
    def indices(self) -> BaseSensitivityAnalysis.SensitivityIndices:
        """The sensitivity indices.

        Given a sensitivity method, an input variable and an output variable,
        the sensitivity index which is a 1D NumPy array can be accessed through
        ``indices.method_name[output_name][output_component][input_name]``.
        """
        return self._indices

    @property
    def main_indices(self) -> FirstOrderIndicesType:
        """The main sensitivity indices.

        Given an input variable and an output variable,
        the sensitivity index which is a 1D NumPy array can be accessed through
        ``main_indices[output_name][output_component][input_name]``.
        """
        return getattr(self.indices, str(self.main_method).lower())

    def sort_input_variables(self, output: VariableType) -> list[str]:
        """Return the input variables sorted in descending order.

        Args:
            output: Either a tuple as ``(output_name, output_component)``
                or an output name; in the second case, use the first output component.

        Returns:
            The names of the inputs sorted by cumulative sensitivity index,
            which is the sum of the absolute values of the sensitivity indices
            associated to the different components of an input.
        """
        output_name, output_component = get_name_and_component(output)
        return [
            input_name
            for input_name, _ in sorted(
                self.main_indices[output_name][output_component].items(),
                key=lambda indices: abs(indices[1]).sum(),
                reverse=True,
            )
        ]

    def plot(
        self,
        output: VariableType,
        input_names: Iterable[str] = (),
        title: str = "",
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        file_format: str = "",
    ) -> DatasetPlot | Figure:
        """Plot the sensitivity indices.

        Args:
            output: The output
                for which to display sensitivity indices,
                either a name or a tuple of the form (name, component).
                If name, its first component is considered.
            input_names: The input variables
                for which to display the sensitivity indices.
                If empty, display all the input variables.
            title: The title of the plot, if any.
            save: If ``True``, save the figure.
            show: If ``True``, show the figure.
            file_path: A file path.
                Either a complete file path, a directory name or a file name.
                If empty, use a default file name and a default directory.
                The file extension is inferred from filepath extension, if any.
            file_format: A file format, e.g. 'png', 'pdf', 'svg', ...
                Used when ``file_path`` does not have any extension.
                If empty, use a default file extension.

        Returns:
            The plot figure.
        """
        raise NotImplementedError

    def plot_field(
        self,
        output: VariableType,
        mesh: RealArray | None = None,
        input_names: Iterable[str] = (),
        standardize: bool = False,
        title: str = "",
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_format: str = "",
        properties: Mapping[str, DatasetPlotPropertyType] = READ_ONLY_EMPTY_DICT,
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
                or a (p, 2) array for a 2D one. If ``None``,
                assume a 1D functional output.
            input_names: The input variables
                for which to display the sensitivity indices.
                If empty, display all the input variables.
            standardize: Whether to scale the indices to :math:`[0,1]`.
            title: The title of the plot, if any.
            save: If ``True``, save the figure.
            show: If ``True``, show the figure.
            file_path: The path of the file to save the figures.
                If empty,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_extension``.
            directory_path: The path of the directory to save the figures.
                If empty, use the current working directory.
            file_name: The name of the file to save the figures.
                If empty, use a default one generated by the post-processing.
            file_format: A file extension, e.g. 'png', 'pdf', 'svg', ...
                If empty, use a default file extension.
            properties: The general properties of a :class:`.DatasetPlot`.

        Returns:
            A bar plot representing the sensitivity indices.

        Raises:
            NotImplementedError: If the dimension of the mesh is greater than 2.
        """
        output_name, output_component = get_name_and_component(output)
        input_names = self._filter_sorted_input_names(
            (output_name, output_component), input_names
        )
        if standardize:
            main_indices = self.standardize_indices(self.main_indices)
        else:
            main_indices = self.main_indices

        data = [
            [main_index[input_name] for main_index in main_indices[output_name]]
            for input_name in input_names
        ]

        data = array(data)[:, :, 0]
        dataset = Dataset.from_array(data, [output_name], {output_name: data.shape[1]})
        dataset.index = input_names
        mesh = linspace(0, 1, data.shape[1]) if mesh is None else mesh
        dataset.misc["mesh"] = mesh
        mesh_dimension = len(dataset.misc["mesh"].shape)
        if mesh_dimension == 1:
            plot = Curves(dataset, mesh="mesh", variable=output_name)
        elif mesh_dimension == 2:
            plot = Surfaces(dataset, mesh="mesh", variable=output_name)
        else:
            raise NotImplementedError

        for k, v in properties.items():
            setattr(plot, k, v)
        plot.title = title
        plot.execute(
            save=save,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
        )
        return plot

    def plot_bar(
        self,
        outputs: OutputsType = (),
        input_names: Iterable[str] = (),
        standardize: bool = False,
        title: str = "",
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_format: str = "",
        sort: bool = True,
        sorting_output: VariableType = "",
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
                If empty, use the default outputs.
            input_names: The input variables
                for which to display the sensitivity indices.
                If empty, display all the input variables.
            standardize: Whether to scale the indices to :math:`[0,1]`.
            title: The title of the plot, if any.
            save: If ``True``, save the figure.
            show: If ``True``, show the figure.
            file_path: The path of the file to save the figures.
                If the extension is missing, use ``file_extension``.
                If empty,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_extension``.
            directory_path: The path of the directory to save the figures.
                If empty, use the current working directory.
            file_name: The name of the file to save the figures.
                If empty, use a default one generated by the post-processing.
            file_format: A file extension, e.g. 'png', 'pdf', 'svg', ...
                If None, use a default file extension.
            sort: Whether to sort the input variables
                by decreasing order of the sensitivity indices
                associated with the sorting output variable.
            sorting_output: The sorting output variable
                If empty, use the first one.
            **options: The options to instantiate the :class:`.BarPlot`.

        Returns:
            A bar chart representing the sensitivity indices.
        """
        _options = {"n_digits": 2}
        _options.update(options)
        bar_plot = BarPlot(
            self.__create_dataset_to_plot(
                input_names,
                outputs or self._output_names,
                standardize,
                sort,
                sorting_output,
            ),
            **_options,
        )
        bar_plot.title = title
        bar_plot.execute(
            save=save,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
        )
        return bar_plot

    def __create_dataset_to_plot(
        self,
        input_names: Iterable[str],
        outputs: OutputsType,
        standardize: bool,
        sort: True,
        sorting_output: VariableType,
    ) -> Dataset:
        r"""Create the dataset to plot.

        Args:
            input_names: The names of the input variables
                for which to display the sensitivity indices.
                If empty, display all the input variables.
            outputs: The outputs
                for which to display sensitivity indices,
                either a name,
                a list of names,
                a (name, component) tuple,
                a list of such tuples or
                a list mixing such tuples and names.
                When a name is specified, all its components are considered.
            standardize: Whether to scale the indices to :math:`[0,1]`.
            sort: Whether to sort the input variables
                by decreasing order of the sensitivity indices
                associated with the sorting output variable.
            sorting_output: The sorting output variable
                If empty, use the first one.

        Returns:
            The dataset to plot.
        """
        sizes = {k: len(v) for k, v in self.main_indices.items()}
        if standardize:
            main_indices = self.standardize_indices(self.main_indices)
        else:
            main_indices = self.main_indices

        outputs = list(get_variables_with_components(outputs, sizes))
        input_names = self._filter_sorted_input_names(outputs[0], input_names)
        data = {name: [] for name in input_names}
        for output_name, output_component in outputs:
            indices = main_indices[output_name][output_component]
            for input_name in input_names:
                data[input_name].append(indices[input_name])

        dataset = Dataset(
            hstack([vstack(data[input_name]) for input_name in input_names]),
            columns=MultiIndex.from_tuples(
                [
                    (Dataset.PARAMETER_GROUP, input_name, index)
                    for input_name in input_names
                    for index in range(
                        self.dataset.variable_names_to_n_components[input_name]
                    )
                ],
                names=Dataset.COLUMN_LEVEL_NAMES,
            ),
        )

        dataset.index = [
            repr_variable(
                name, component, size=self.dataset.variable_names_to_n_components[name]
            )
            for name, component in outputs
        ]
        if sort:
            if sorting_output:
                name, component = get_name_and_component(sorting_output)
                by = repr_variable(
                    name,
                    component,
                    size=self.dataset.variable_names_to_n_components[name],
                )
            else:
                by = dataset.index[0]
            dataset = dataset.sort_values(by=by, ascending=False, axis=1)

        return dataset

    def plot_radar(
        self,
        outputs: OutputsType = (),
        input_names: Iterable[str] = (),
        standardize: bool = False,
        title: str = "",
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_format: str = "",
        min_radius: float | None = None,
        max_radius: float | None = None,
        sort: bool = True,
        sorting_output: VariableType = "",
        **options: bool | int,
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
                If empty, use the default outputs.
            input_names: The input variables
                for which to display the sensitivity indices.
                If empty, display all the input variables.
            standardize: Whether to scale the indices to :math:`[0,1]`.
            title: The title of the plot, if any.
            save: If ``True``, save the figure.
            show: If ``True``, show the figure.
            file_path: The path of the file to save the figures.
                If the extension is missing, use ``file_extension``.
                If empty,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_extension``.
            directory_path: The path of the directory to save the figures.
                If empty, use the current working directory.
            file_name: The name of the file to save the figures.
                If empty, use a default one generated by the post-processing.
            file_format: A file extension, e.g. 'png', 'pdf', 'svg', ...
                If empty, use a default file extension.
            min_radius: The minimal radial value. If ``None``, from data.
            max_radius: The maximal radial value. If ``None``, from data.
            sort: Whether to sort the input variables
                by decreasing order of the sensitivity indices
                associated with the sorting output variable.
            sorting_output: The sorting output variable
                If empty, use the first one.
            **options: The options to instantiate the :class:`.RadarChart`.

        Returns:
            A radar chart representing the sensitivity indices.
        """
        radar_chart = RadarChart(
            self.__create_dataset_to_plot(
                input_names,
                outputs or self._output_names,
                standardize,
                sort,
                sorting_output,
            ),
            **options,
        )
        radar_chart.title = title
        radar_chart.rmin = min_radius or radar_chart.rmin
        radar_chart.rmax = max_radius or radar_chart.rmax
        radar_chart.execute(
            save=save,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
        )
        return radar_chart

    def _filter_sorted_input_names(
        self, output: tuple[str, int], inputs_to_keep: Iterable[str]
    ) -> Iterable[str]:
        """Filter the input names sorted in descending order of influence.

        Args:
            output: An output for which to display the sensitivity indices.
            inputs_to_keep: The inputs to keep. If ``None``, keep all.

        Returns:
            The filtered input names sorted in descending order of influence.
        """
        return filter_names(self.sort_input_variables(output), inputs_to_keep)

    def plot_comparison(
        self,
        indices: list[BaseSensitivityAnalysis],
        output: VariableType,
        input_names: Iterable[str] = (),
        title: str = "",
        use_bar_plot: bool = True,
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_format: str = "",
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
            input_names: The input variables
                for which to display the sensitivity indices.
                If empty, display all the input variables.
            title: The title of the plot, if any.
            use_bar_plot: The type of graph.
                If ``True``, use a bar plot. Otherwise, use a radar chart.
            save: If ``True``, save the figure.
            show: If ``True``, show the figure.
            file_path: The path of the file to save the figures.
                If empty,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_format``.
            directory_path: The path of the directory to save the figures.
                If empty, use the current working directory.
            file_name: The name of the file to save the figures.
                If empty, use a default one generated by the post-processing.
            file_format: A file format, e.g. 'png', 'pdf', 'svg', ...
                If empty, use a default file extension.
            **options: The options passed to the underlying :class:`.DatasetPlot`.

        Returns:
            A graph comparing sensitivity indices.
        """
        output = get_name_and_component(output)
        if isinstance(indices, BaseSensitivityAnalysis):
            indices = [indices]
        methods = [self, *indices]
        dataset = Dataset()
        input_names = self._filter_sorted_input_names(output, input_names)
        for input_name in input_names:
            data = abs(
                array([
                    method.main_indices[output[0]][output[1]][input_name]
                    for method in methods
                ])
            )
            dataset.add_variable(input_name, data)
        data = dataset.get_view(group_names=dataset.PARAMETER_GROUP).to_numpy()
        dataset.update_data(
            data / data.max(axis=1)[:, newaxis], group_names=dataset.PARAMETER_GROUP
        )
        dataset.index = [method.main_method for method in methods]
        if use_bar_plot:
            plot = BarPlot(dataset, n_digits=2)
        else:
            plot = RadarChart(dataset)
            plot.rmin = 0.0
            plot.rmax = 1.0
        plot.title = title
        plot.execute(
            save, show, file_path, directory_path, file_name, file_format, **options
        )
        return plot

    def to_dataset(self) -> Dataset:
        """Convert :attr:`.BaseSensitivityAnalysis.indices` into a :class:`.Dataset`.

        Returns:
            The sensitivity indices.
        """
        sizes = self.dataset.variable_names_to_n_components

        row_names = []
        for input_name in self.input_names:
            for input_component in range(sizes[input_name]):
                row_names.append(  # noqa: PERF401
                    repr_variable(
                        input_name,
                        input_component,
                        size=self.dataset.variable_names_to_n_components[input_name],
                    )
                )

        dataset = Dataset()
        for method, indices in asdict(self.indices).items():
            if method in self._INTERACTION_METHODS:
                dataset.misc[method] = indices
                continue

            variables = []
            sizes = {}
            data = []
            for output, components in indices.items():
                variables.append(output)
                sizes[output] = len(components)
                for component in components:
                    data.append([component[name].tolist() for name in self.input_names])
                    data[-1] = [item for sublist in data[-1] for item in sublist]
            data = array(data).T
            dataset.add_group(
                method,
                data,
                [f"{v}" for v in variables],
                {f"{v}": s for v, s in sizes.items()},
            )
        dataset.index = row_names
        return dataset

    @staticmethod
    def standardize_indices(
        indices: FirstOrderIndicesType,
    ) -> FirstOrderIndicesType:
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

    def _get_output_names(self, output_names: str | Iterable[str]) -> Iterable[str]:
        """Return the output names.

        Args:
            output_names: The initial output name(s).
                If empty, return the default output names.

        Returns:
            The output names.
        """
        if not output_names:
            return self.default_output_names

        return convert_strings_to_iterable(output_names)
