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

The purpose of a sensitivity analysis is to qualify or quantify how the model's
uncertain inputs impact its outputs.

This analysis relies on :class:`.SensitivityAnalysis` computed from a
:class:`.MDODiscipline` representing the model,
a :class:`.ParameterSpace` describing the
uncertain parameters and options associated with a particular concrete class inheriting
from :class:`.SensitivityAnalysis` which is an abstract one.
"""

from __future__ import annotations

import pickle
from abc import abstractmethod
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Union

from numpy import array
from numpy import hstack
from numpy import linspace
from numpy import ndarray
from numpy import newaxis
from numpy import vstack
from pandas import MultiIndex

from gemseo.core.doe_scenario import DOEScenario
from gemseo.datasets.dataset import Dataset
from gemseo.disciplines.utils import get_all_outputs
from gemseo.post.dataset.bars import BarPlot
from gemseo.post.dataset.curves import Curves
from gemseo.post.dataset.radar_chart import RadarChart
from gemseo.post.dataset.surfaces import Surfaces
from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.matplotlib_figure import save_show_figure
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from strenum import StrEnum

    from gemseo.algos.doe.doe_library import DOELibraryOptionType
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline import MDODiscipline
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.post.dataset.dataset_plot import DatasetPlotPropertyType
    from gemseo.post.dataset.dataset_plot import VariableType

OutputsType = Union[str, tuple[str, int], Sequence[Union[str, tuple[str, int]]]]
FirstOrderIndicesType = dict[str, list[dict[str, ndarray]]]
SecondOrderIndicesType = dict[str, list[dict[str, dict[str, ndarray]]]]


class SensitivityAnalysis(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Sensitivity analysis.

    The :class:`.SensitivityAnalysis` class provides both the values of
    :attr:`.SensitivityAnalysis.indices` and their graphical representations,
    from either the :meth:`.SensitivityAnalysis.plot` method, the
    :meth:`.SensitivityAnalysis.plot_radar` method or the
    :meth:`.SensitivityAnalysis.plot_bar` method.

    It is also possible to use :meth:`.SensitivityAnalysis.sort_parameters` to get the
    parameters sorted according to :attr:`.SensitivityAnalysis.main_method`. The
    :attr:`.SensitivityAnalysis.main_indices` are indices computed with the latter.

    Lastly, the :meth:`.SensitivityAnalysis.plot_comparison` method allows to compare
    the current :class:`.SensitivityAnalysis` with another one.
    """

    # TODO: API: rename to default_outputs or default_output_names
    default_output: Iterable[str]
    """The default outputs of interest."""

    dataset: IODataset
    """The dataset containing the discipline evaluations."""

    Method: ClassVar[type[StrEnum]]
    """The names of the sensitivity methods considering simple effects.

    A simple effect is the effect of an isolated input variable on an output variable
    while an interaction effect is the effect of the interaction between several input
    variables on an output variable.
    """

    _INTERACTION_METHODS: ClassVar[tuple[str]] = ()
    """The names of the sensitivity methods considering interaction effects."""

    DEFAULT_DRIVER = None

    _input_names: list[str]
    """The names of the inputs in parameter space order."""

    _output_names: Iterable[str]
    """The disciplines' outputs to be considered for the analysis."""

    _algo_name: str
    """The name of the DOE algorithm to sample the discipline."""

    _file_path_manager: FilePathManager
    """The file path manager for the figures."""

    _main_method: Method  # noqa: F821
    """The name of the main sensitivity analysis method."""

    _indices: dict[str, FirstOrderIndicesType]
    """The sensitivity indices computed by the method compute_indices.

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

    def __init__(
        self,
        disciplines: Collection[MDODiscipline],
        parameter_space: ParameterSpace,
        n_samples: int | None = None,
        output_names: Iterable[str] = (),
        algo: str = "",
        algo_options: Mapping[str, DOELibraryOptionType] = MappingProxyType({}),
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
                If empty, use all the outputs.
            algo: The name of the DOE algorithm.
                If empty, use the :attr:`.SensitivityAnalysis.DEFAULT_DRIVER`.
            algo_options: The options of the DOE algorithm.
            formulation: The name of the :class:`.MDOFormulation`
                to sample the disciplines.
            **formulation_options: The options of the :class:`.MDOFormulation`.
        """  # noqa: D205, D212, D415
        disciplines = list(disciplines)
        self._algo_name = algo or self.DEFAULT_DRIVER
        self._output_names = output_names or get_all_outputs(disciplines)
        self.default_output = self._output_names
        self._input_names = parameter_space.variable_names
        self.dataset = self.__sample_disciplines(
            disciplines,
            parameter_space,
            n_samples,
            algo_options,
            formulation,
            **(formulation_options or {}),
        ).to_dataset(opt_naming=False)
        self._main_method = None
        self._file_path_manager = FilePathManager(
            FilePathManager.FileType.FIGURE,
            default_name=FilePathManager.to_snake_case(self.__class__.__name__),
        )

    def to_pickle(self, file_path: str | Path) -> None:
        """Save the current sensitivity analysis on the disk.

        Args:
            file_path: The path to the file.
        """
        with Path(file_path).open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(file_path: str | Path) -> SensitivityAnalysis:
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
            formulation: The name of the :class:`.MDOFormulation`
                to sample the disciplines.
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
        algo_options = algo_options or {}
        algo_options["log_problem"] = False
        algo_options["use_one_line_progress_bar"] = True
        scenario.execute({
            scenario.ALGO: self._algo_name,
            scenario.N_SAMPLES: n_samples,
            scenario.ALGO_OPTIONS: algo_options,
        })
        return scenario

    def _create_scenario(
        self,
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
            formulation: The name of the :class:`.MDOFormulation`
                to sample the disciplines.
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
            name=f"{self.__class__.__name__}SamplingPhase",
            **formulation_options,
        )
        for discipline in disciplines:
            for output_name in discipline.get_output_data_names():
                if output_name in observable_names[1:]:
                    scenario.add_observable(output_name)
        return scenario

    @property
    def input_names(self) -> list[str]:
        """The names of the inputs."""
        return self._input_names

    @abstractmethod
    def compute_indices(
        self, outputs: str | Sequence[str] = ()
    ) -> dict[str, FirstOrderIndicesType | SecondOrderIndicesType]:
        """Compute the sensitivity indices.

        Args:
            outputs: The output(s)
                for which to display the sensitivity indices.
                If ``None``,
                use the default outputs set at instantiation.

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

    @property
    def indices(self) -> dict[str, FirstOrderIndicesType]:
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
        return self._indices

    @property
    def main_method(self) -> Method:  # noqa: F821
        """The name of the main method.

        One of the enum :class:`.Sensitivity.Method`.
        """
        return self._main_method

    @main_method.setter
    def main_method(self, method: Method) -> None:  # noqa: D102, F821
        self._main_method = method

    @property
    def main_indices(self) -> FirstOrderIndicesType:
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
        return self.indices[self._main_method]

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
                If ``None``, use the default outputs.

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
            [output] if isinstance(output, tuple) else get_all(output)
            for output in outputs
        ]
        return [item for sublist in result for item in sublist]

    def sort_parameters(self, output: VariableType) -> list[str]:
        """Return the parameters sorted in descending order.

        Args:
            output: Either a tuple as ``(output_name, output_component)``
                or an output name; in the second case, use the first output component.

        Returns:
            The input parameters sorted by decreasing order of sensitivity;
            in case of a multivariate input,
            aggregate the sensitivity indices
            associated to the different input components by adding them up typically.
        """
        if isinstance(output, str):
            output_name, output_index = output, 0
        else:
            output_name, output_index = output

        return [
            input_name
            for input_name, _ in sorted(
                self.main_indices[output_name][output_index].items(),
                key=lambda item: self._aggregate_sensitivity_indices(item[1]),
                reverse=True,
            )
        ]

    @staticmethod
    def _aggregate_sensitivity_indices(indices: NDArray[float]) -> float:
        """Aggregate sensitivity indices.

        Args:
            indices: The sensitivity indices to be aggregated.

        Returns:
            The aggregated index.
        """
        return indices.sum()

    def plot(
        self,
        output: VariableType,
        inputs: Iterable[str] = (),
        title: str = "",
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        file_format: str = "",
    ) -> None:
        """Plot the sensitivity indices.

        Args:
            output: The output
                for which to display sensitivity indices,
                either a name or a tuple of the form (name, component).
                If name, its first component is considered.
            inputs: The uncertain input variables
                for which to display the sensitivity indices.
                If empty, display all the uncertain input variables.
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
        """
        raise NotImplementedError

    def plot_field(
        self,
        output: VariableType,
        mesh: ndarray | None = None,
        inputs: Iterable[str] = (),
        standardize: bool = False,
        title: str = "",
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_format: str = "",
        properties: Mapping[str, DatasetPlotPropertyType] = MappingProxyType({}),
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
            inputs: The uncertain input variables
                for which to display the sensitivity indices.
                If empty, display all the uncertain input variables.
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
        if isinstance(output, str):
            output_name = output
            output_component = 0
        else:
            output_name, output_component = output

        input_names = self._sort_and_filter_input_parameters(
            (output_name, output_component), inputs
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
        inputs: Iterable[str] = (),
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
            inputs: The uncertain input variables
                for which to display the sensitivity indices.
                If empty, display all the uncertain input variables.
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
            **options: The options to instantiate the :class:`.BarPlot`.
                If empty, use a default file extension.
            sort: Whether to sort the uncertain variables
                by decreasing order of the sensitivity indices
                associated with the sorting output variable.
            sorting_output: The sorting output variable
                If empty, use the first one.
            **options: The options to instantiate the :class:`.BarPlot`.

        Returns:
            A bar chart representing the sensitivity indices.
        """
        outputs = outputs or self._output_names
        _options = {"n_digits": 2}
        _options.update(options)
        plot = BarPlot(
            self.__create_dataset_to_plot(
                inputs, outputs, standardize, sort, sorting_output
            ),
            **_options,
        )
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

    def __create_dataset_to_plot(
        self,
        inputs: Iterable[str],
        outputs: OutputsType,
        standardize: bool,
        sort: True,
        sorting_output: VariableType,
    ) -> Dataset:
        r"""Create the dataset to plot.

        Args:
            inputs: The uncertain input variables
                for which to display the sensitivity indices.
                If empty, display all the uncertain input variables.
            outputs: The outputs
                for which to display sensitivity indices,
                either a name,
                a list of names,
                a (name, component) tuple,
                a list of such tuples or
                a list mixing such tuples and names.
                When a name is specified, all its components are considered.
            standardize: Whether to scale the indices to :math:`[0,1]`.
            sort: Whether to sort the uncertain variables
                by decreasing order of the sensitivity indices
                associated with the sorting output variable.
            sorting_output: The sorting output variable
                If empty, use the first one.

        Returns:
            The dataset to plot.
        """
        outputs = self._outputs_to_tuples(outputs)
        if standardize:
            main_indices = self.standardize_indices(self.main_indices)
        else:
            main_indices = self.main_indices

        input_names = self._sort_and_filter_input_parameters(outputs[0], inputs)
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
                *output, size=self.dataset.variable_names_to_n_components[output[0]]
            )
            for output in outputs
        ]
        if sort:
            if sorting_output:
                if isinstance(sorting_output, str):
                    sorting_output = (sorting_output, 0)

                sorting_output = self._outputs_to_tuples([sorting_output])[0]
                by = repr_variable(
                    *sorting_output,
                    size=self.dataset.variable_names_to_n_components[sorting_output[0]],
                )
            else:
                by = dataset.index[0]
            dataset = dataset.sort_values(by=by, ascending=False, axis=1)

        return dataset

    def plot_radar(
        self,
        outputs: OutputsType = (),
        inputs: Iterable[str] = (),
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
            inputs: The uncertain input variables
                for which to display the sensitivity indices.
                If empty, display all the uncertain input variables.
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
            sort: Whether to sort the uncertain variables
                by decreasing order of the sensitivity indices
                associated with the sorting output variable.
            sorting_output: The sorting output variable
                If empty, use the first one.
            **options: The options to instantiate the :class:`.RadarChart`.

        Returns:
            A radar chart representing the sensitivity indices.
        """
        outputs = outputs or self._output_names
        plot = RadarChart(
            self.__create_dataset_to_plot(
                inputs, outputs, standardize, sort, sorting_output
            ),
            **options,
        )
        plot.title = title
        plot.rmin = min_radius or plot.rmin
        plot.rmax = max_radius or plot.rmax
        plot.execute(
            save=save,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
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
            names_to_keep: The names to keep. If ``None``, keep all.

        Returns:
            The filtered names.
        """
        if names_to_keep:
            names = [item for item in names if item in set(names_to_keep)]
        return names

    def _sort_and_filter_input_parameters(
        self, output: tuple[str, int], inputs_to_keep: Iterable[str]
    ) -> list[str]:
        """Sort and filter the input parameters.

        Args:
            output: An output for which to display sensitivity indices.
            inputs_to_keep: The inputs to keep. If ``None``, keep all.

        Returns:
            The filtered names.
        """
        return self._filter_names(self.sort_parameters(output), inputs_to_keep)

    def plot_comparison(
        self,
        indices: list[SensitivityAnalysis],
        output: VariableType,
        inputs: Iterable[str] = (),
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
            inputs: The uncertain input variables
                for which to display the sensitivity indices.
                If empty, display all the uncertain input variables.
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
        if not isinstance(output, tuple):
            output = (output, 0)
        if isinstance(indices, SensitivityAnalysis):
            indices = [indices]
        methods = [self, *indices]
        dataset = Dataset()
        input_names = self._sort_and_filter_input_parameters(output, inputs)
        for name in input_names:
            data = abs(
                array([
                    method.main_indices[output[0]][output[1]][name]
                    for method in methods
                ])
            )
            dataset.add_variable(name, data)
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

    def _save_show_plot(
        self,
        fig: Figure,
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_format: str = "",
    ) -> Figure:
        """Save or show the plot.

        Args:
            fig: The figure to be processed.
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
            file_path = ""

        save_show_figure(fig, show, file_path)
        return fig

    def to_dataset(self) -> Dataset:
        """Convert :attr:`.SensitivityAnalysis.indices` into a :class:`.Dataset`.

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
        for method, indices in self.indices.items():
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
