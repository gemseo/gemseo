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
"""Base class for sensitivity analysis."""

from __future__ import annotations

import pickle
from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Generic
from typing import TypeVar

from numpy import array
from numpy import hstack
from numpy import isnan
from numpy import linspace
from numpy import nanmax
from numpy import nansum
from numpy import newaxis
from numpy import vstack
from pandas import MultiIndex
from strenum import StrEnum

from gemseo.dataset.dataset import Dataset
from gemseo.dataset.io_dataset import IODataset
from gemseo.doe.factory import DOE_LIBRARY_FACTORY
from gemseo.formulation.mdf_settings import MDF_Settings
from gemseo.post.dataset.bar_plot import BarPlot
from gemseo.post.dataset.bar_plot_settings import BarPlot_Settings
from gemseo.post.dataset.curves import Curves
from gemseo.post.dataset.curves_settings import Curves_Settings
from gemseo.post.dataset.radar_chart import RadarChart
from gemseo.post.dataset.radar_chart_settings import RadarChart_Settings
from gemseo.post.dataset.surfaces import Surfaces
from gemseo.post.dataset.surfaces_settings import Surfaces_Settings
from gemseo.scenario.evaluation import EvaluationScenario
from gemseo.util.constant import READ_ONLY_EMPTY_DICT
from gemseo.util.data_conversion import split_array_to_dict_of_arrays
from gemseo.util.discipline import get_all_outputs
from gemseo.util.file_path_manager import FilePathManager
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta
from gemseo.util.pydantic import BaseSettings
from gemseo.util.pydantic import create_model
from gemseo.util.string import convert_strings_to_iterable
from gemseo.util.string import filter_names
from gemseo.util.string import get_name_and_component
from gemseo.util.string import get_variables_with_components
from gemseo.util.string import pretty_str
from gemseo.util.string import repr_variable
from gemseo.util.typing import RealArray

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Mapping

    from matplotlib.figure import Figure

    from gemseo.core.discipline import Discipline
    from gemseo.doe.core.base_doe_settings import BaseDOESettings
    from gemseo.formulation.core.base_settings import BaseFormulationSettings
    from gemseo.post.dataset.base import BaseDatasetPlot
    from gemseo.post.dataset.base import DatasetPlotPropertyType
    from gemseo.scenario.backup_settings import BackupSettings
    from gemseo.space.parameter import ParameterSpace
    from gemseo.util.string import VariableType
    from gemseo.util.typing import StrPath

OutputsType = str | tuple[str, int] | Sequence[str | tuple[str, int]]
FirstOrderIndicesType = dict[str, list[dict[str, RealArray] | None]]
SecondOrderIndicesType = dict[str, list[dict[str, dict[str, RealArray]] | None]]

T = TypeVar("T", bound=StrEnum)


class BaseGenericSensitivityAnalysis(
    Generic[T], metaclass=ABCGoogleDocstringInheritanceMeta
):
    """Base class for sensitivity analysis.

    The aim of a sensitivity analysis is to qualify or quantify
    how the uncertain inputs of a model impact its outputs,
    from samples of this model, also called samples.

    A
    [BaseSensitivityAnalysis][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis]
    can be created from such samples
    (passed as an [IODataset][gemseo.dataset.io_dataset.IODataset])
    or use its
    [compute_samples()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.compute_samples]
    method to generate them,
    using a [Discipline][gemseo.core.discipline.discipline.Discipline]
    representing the model,
    a [ParameterSpace][gemseo.space.parameter.ParameterSpace]
    describing the uncertain input variables
    and a set of options.
    In the second case,
    the samples returned by
    [compute_samples()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.compute_samples]
    can be saved on the disk for future use.
    """

    dataset: IODataset | None
    """The dataset containing the discipline evaluations.

    The samples must be
    either passed at instantiation
    or generated with
    [compute_samples()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.compute_samples].
    """

    _INTERACTION_METHODS: ClassVar[tuple[str, ...]] = ()
    """The names of the sensitivity methods considering interaction effects."""

    _DEFAULT_MAIN_METHOD: ClassVar[StrEnum]
    """The name of the default main sensitivity analysis method."""

    _input_names: list[str]
    """The names of the inputs in uncertain space order."""

    _output_names: list[str]
    """The names of the outputs."""

    _file_path_manager: FilePathManager
    """The file path manager for the figures."""

    main_method: T
    """The name of the main sensitivity analysis method."""

    @dataclass(frozen=True)
    class SensitivityIndices:
        """The sensitivity indices.

        Given a sensitivity method, an input and an output,
        a sensitivity index is a NumPy array that can be accessed through
        `indices.method_name[output_name][output_component][input_name]`.

        For constant output components,
        `indices.method_name[output_name][output_component]` is `None`.
        """

    _indices: SensitivityIndices
    """The sensitivity indices computed by the `compute_indices()` method."""

    def __init__(self, samples: IODataset | StrPath = "") -> None:
        """
        Args:
            samples: The samples for the estimation of the sensitivity indices,
                either as an [IODataset][gemseo.dataset.io_dataset.IODataset]
                or as a pickle file path generated
                from the [to_pickle()][gemseo.util.pickle.to_pickle] function.
                If empty, use
                [compute_samples()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.compute_samples].
        """  # noqa: D202, D205, D212
        if isinstance(samples, IODataset):
            self.dataset = samples
        elif samples:
            with Path(samples).open("rb") as f:
                samples = self.dataset = pickle.load(f)
        else:
            self.dataset = None

        self._file_path_manager = FilePathManager(
            FilePathManager.FileType.FIGURE,
            default_name=FilePathManager.to_snake_case(self.__class__.__name__),
        )
        self.main_method = self._DEFAULT_MAIN_METHOD
        if self.dataset is None:
            self._input_names = []
            self._output_names = []
        else:
            self._input_names = samples.input_names
            self._output_names = samples.output_names
        self._indices = self.SensitivityIndices()

    @abstractmethod
    def compute_samples(
        self,
        disciplines: Collection[Discipline],
        parameter_space: ParameterSpace,
        algo_settings: BaseSettings | None = None,
        formulation_settings: BaseFormulationSettings | None = None,
    ) -> IODataset:
        """Sample the model over the uncertain space.

        This step is a prerequisite for calculating the sensitivity indices
        if the samples were not passed during instantiation.

        Args:
            disciplines: The discipline or disciplines to use for the analysis.
            parameter_space: A parameter space.
            algo_settings: The settings of the DOE algorithm.
                If `None`,
                use the default settings of the default DOE algorithm
                (see
                [DEFAULT_DRIVER][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.DEFAULT_DRIVER]).
            formulation_settings: The settings of the MDO formulation.
                If `None`,
                use the default settings of the MDF formulation.

        Returns:
            The samples for the estimation of the sensitivity indices.
        """

    @property
    def default_output_names(self) -> list[str]:
        """The names of all of the outputs defined in the samples."""
        return self._output_names

    @property
    def input_names(self) -> list[str]:
        """The names of all of the inputs defined in the samples."""
        return self._input_names

    @abstractmethod
    def compute_indices(
        self, output_names: str | Iterable[str] = ()
    ) -> SensitivityIndices:
        """Compute the sensitivity indices of certain outputs.

        Args:
            output_names: The name(s) of the output(s)
                for which to compute the sensitivity indices.
                If empty, consider all of the outputs defined in the samples.

        Returns:
            The sensitivity indices.

            Given a sensitivity method, an input and an output,
            a sensitivity index is a NumPy array that can be accessed through
            `indices.method_name[output_name][output_component][input_name]`.

            For constant output components,
            `indices.method_name[output_name][output_component]` is `None`.
        """

    @property
    def indices(self) -> BaseGenericSensitivityAnalysis.SensitivityIndices:
        """The sensitivity indices.

        Given a sensitivity method, an input and an output,
        a sensitivity index is a NumPy array that can be accessed through
        `analysis.indices.method_name[output_name][output_component][input_name]`.

        For constant output components,
        `analysis.indices.method_name[output_name][output_component]` is `None`.
        """
        return self._indices

    @property
    def main_indices(self) -> FirstOrderIndicesType:
        """The main sensitivity indices.

        Given an input and an output,
        a main sensitivity index is a NumPy array that can be accessed through
        `analysis.main_indices.method_name[output_name][output_component][input_name]`.

        For constant output components,
        `analysis.main_indices.method_name[output_name][output_component]` is `None`.
        """
        return getattr(self.indices, self._get_index_field_name(self.main_method))

    @staticmethod
    def _get_index_field_name(method: StrEnum) -> str:
        """Get the `SensitivityIndices` field name for a sensitivity analysis method.

        Args:
            method: A sensitivity analysis method.

        Returns:
            The name of the field of `SensitivityIndices`
            related to the sensitivity analysis method.
        """
        return str(method).lower().replace("-", "_")

    def _get_output_n_components(self, name: str) -> int:
        """Return the number of components of an output.

        Args:
            name: The name of an output.

        Returns:
            The number of components of the output.
        """
        return self.dataset.variable_name_to_n_components[name]

    def _get_input_sample_array(self) -> RealArray:
        """Return the input samples as a 2D NumPy array.

        Returns:
            The input samples shaped as `(n_samples, input_dimension)`.
        """
        return self.dataset.get_view(group_names=self.dataset.INPUT_GROUP).to_numpy()

    def _iter_output_components(
        self, output_names: Iterable[str]
    ) -> Iterable[tuple[str, int, RealArray | None]]:
        """Iterate over the components of the outputs.

        Args:
            output_names: The names of the outputs to iterate over.

        Yields:
            For each output component,
            a tuple `(output_name, component_index, data)`
            where `data` is the `(n_samples, 1)` array of the component samples,
            or `None` when the component is constant.
        """
        dataset = self.dataset
        for output_name in output_names:
            samples = dataset.get_view(
                group_names=dataset.OUTPUT_GROUP, variable_names=output_name
            ).to_numpy()
            for component_index, component_samples in enumerate(samples.T):
                data = component_samples[:, newaxis]
                yield output_name, component_index, None if data.var() == 0.0 else data

    def _split_index_array(self, raw_array: Sequence[float]) -> dict[str, RealArray]:
        """Split a flat sensitivity-index array into a `{input_name: array}` dict.

        Args:
            raw_array: The sensitivity indices for a single output component,
                ordered along the input components.

        Returns:
            The sensitivity indices indexed by input name.
        """
        return split_array_to_dict_of_arrays(
            array(raw_array),
            self.dataset.variable_name_to_n_components,
            self._input_names,
        )

    def sort_input_variables(self, output: VariableType) -> list[str]:
        """Return the inputs sorted in descending order.

        Args:
            output: Either a tuple as `(output_name, output_component)`
                or an output name; in the second case, use the first output component.

        Returns:
            The names of the inputs sorted by cumulative sensitivity index,
            which is the sum of the absolute values of the sensitivity indices
            associated to the different components of an input,
            skipping the components whose index is `nan`;
            an input whose every component has a `nan` index is ranked last.
        """
        output_name, output_component = get_name_and_component(output)
        return [
            input_name
            for input_name, _ in sorted(
                self.main_indices[output_name][output_component].items(),
                key=lambda indices: (
                    not isnan(indices[1]).all(),
                    nansum(abs(indices[1])),
                ),
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
        file_path: StrPath = "",
        directory_path: StrPath = "",
        file_name: str = "",
        file_format: str = "",
    ) -> BaseDatasetPlot | Figure:
        """Plot the sensitivity indices.

        Args:
            output: The output
                for which to display sensitivity indices,
                either a name or a tuple of the form `(name, component)`.
                If name, its first component is considered.
            input_names: The inputs
                for which to display the sensitivity indices.
                If empty, all the inputs are considered.
            title: The title of the plot, if any.
            save: Whether to save the figure.
            show: Whether to show the figure.
            file_path: A file path.
                Either a complete file path, a directory name or a file name.
                If empty, use a default file name and a default directory.
                The file extension is inferred from filepath extension, if any.
            directory_path: The path of the directory to save the figures.
                If empty, use the current working directory.
            file_name: The name of the file to save the figures.
                If empty, use a default one generated by the post-processing.
            file_format: A file format, e.g. 'png', 'pdf', 'svg', ...
                Used when `file_path` does not have any extension.
                If empty, use a default file extension.

        Returns:
            The plot figure.
        """
        return self.plot_bar(
            outputs=output,
            input_names=input_names,
            title=title,
            save=save,
            show=show,
            file_path=file_path,
            directory_path=directory_path,
            file_name=file_name,
            file_format=file_format,
        )

    def plot_field(
        self,
        output: VariableType,
        mesh: RealArray | None = None,
        input_names: Iterable[str] = (),
        standardize: bool = False,
        title: str = "",
        save: bool = True,
        show: bool = False,
        file_path: StrPath = "",
        directory_path: StrPath = "",
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
                either a name or a tuple of the form `(name, component)`
                where `(name, component)` is used to sort the inputs.
                If it is a name, its first component is considered.
            mesh: The mesh on which the p-length output
                is represented. Either a p-length array for a 1D functional output
                or a (p, 2) array for a 2D one.
                If `None`, a 1D functional output is assumed.
            input_names: The names of the inputs
                for which to display the sensitivity indices.
                If empty, all the inputs are considered.
            standardize: Whether to scale the indices to $[0,1]$.
            title: The title of the plot, if any.
            save: Whether to save the figure.
            show: Whether to show the figure.
            file_path: The path of the file to save the figures.
                If empty,
                create a file path
                from `directory_path`, `file_name` and `file_extension`.
            directory_path: The path of the directory to save the figures.
                If empty, use the current working directory.
            file_name: The name of the file to save the figures.
                If empty, use a default one generated by the post-processing.
            file_format: A file extension, e.g. 'png', 'pdf', 'svg', ...
                If empty, use a default file extension.
            properties: The general properties
                of a [DatasetPlot][gemseo.post.dataset.base.BaseDatasetPlot].

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
            settings = Curves_Settings(
                mesh="mesh", variable=output_name, title=title, **properties
            )
            plot = Curves(dataset, settings)
        elif mesh_dimension == 2:
            settings = Surfaces_Settings(
                mesh="mesh", variable=output_name, title=title, **properties
            )
            plot = Surfaces(dataset, settings)
        else:
            raise NotImplementedError

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
        file_path: StrPath = "",
        directory_path: StrPath = "",
        file_name: str = "",
        file_format: str = "",
        sort: bool = True,
        sorting_output: VariableType = "",
        bar_plot_settings: BarPlot_Settings | None = None,
    ) -> BarPlot:
        """Plot the sensitivity indices on a bar plot.

        This method may consider one or more outputs,
        as well as all inputs (default behavior) or a subset.

        Args:
            outputs: The outputs
                for which to display sensitivity indices,
                either a name,
                a list of names,
                a `(name, component)` tuple,
                a list of such tuples or
                a list mixing such tuples and names.
                When a name is specified, all its components are considered.
                If empty, all the outputs are considered.
            input_names: The names of the inputs
                for which to display the sensitivity indices.
                If empty, all the inputs are considered.
                The input components without sensitivity indices are left out.
            standardize: Whether to scale the indices to $[0,1]$.
            title: The title of the plot, if any.
            save: Whether to save the figure.
            show: Whether to show the figure.
            file_path: The path of the file to save the figures.
                If the extension is missing, use `file_extension`.
                If empty,
                create a file path
                from `directory_path`, `file_name` and `file_extension`.
            directory_path: The path of the directory to save the figures.
                If empty, use the current working directory.
            file_name: The name of the file to save the figures.
                If empty, use a default one generated by the post-processing.
            file_format: A file extension, e.g. 'png', 'pdf', 'svg', ...
                If None, use a default file extension.
            sort: Whether to sort the inputs
                by decreasing order of the sensitivity indices
                associated with the sorting output.
            sorting_output: The sorting output
                If empty, use the first one.
            bar_plot_settings: The settings of the bar plot.
                If `None`,
                use the default bar plot settings,
                except for the number of digits,
                which is set to 2.

        Returns:
            A bar plot representing the sensitivity indices.

        Raises:
            ValueError: When no input component has sensitivity indices.
        """
        if bar_plot_settings is None:
            bar_plot_settings = BarPlot_Settings()

        bar_plot_settings.title = title

        if "n_digits" not in bar_plot_settings.model_fields_set:
            bar_plot_settings.n_digits = 2

        bar_plot = BarPlot(
            self.__create_dataset_to_plot(
                input_names,
                outputs or self._output_names,
                standardize,
                sort,
                sorting_output,
            ),
            bar_plot_settings,
        )
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
        sort: bool,
        sorting_output: VariableType,
    ) -> Dataset:
        r"""Create the dataset to plot.

        Args:
            input_names: The names of the inputs
                for which to display the sensitivity indices.
                If empty, all the inputs are considered.
            outputs: The outputs
                for which to display sensitivity indices,
                either a name,
                a list of names,
                a (name, component) tuple,
                a list of such tuples or
                a list mixing such tuples and names.
                When a name is specified, all its components are considered.
            standardize: Whether to scale the indices to $[0,1]$.
            sort: Whether to sort the inputs
                by decreasing order of the sensitivity indices
                associated with the sorting output.
            sorting_output: The sorting output
                If empty, use the first one.

        Returns:
            The dataset to plot,
            without the input components that have no sensitivity indices.

        Raises:
            ValueError: When no input component has sensitivity indices.
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

        output_labels = [
            repr_variable(name, component, size=self._get_output_n_components(name))
            for name, component in outputs
        ]
        # An input component whose index is NaN has no index at all;
        # it cannot be plotted
        # and would make the plot fail, e.g. as an axis limit.
        columns = []
        values = []
        for input_name in input_names:
            for component, component_values in enumerate(vstack(data[input_name]).T):
                if isnan(component_values).any():
                    continue

                columns.append((Dataset.PARAMETER_GROUP, input_name, component))
                values.append(component_values)

        if not values:
            msg = (
                "The sensitivity indices of the inputs "
                f"{pretty_str(input_names, use_and=True)} are NaN "
                f"for the outputs {pretty_str(output_labels, use_and=True)}; "
                "there is nothing to plot."
            )
            raise ValueError(msg)

        dataset = Dataset(
            vstack(values).T,
            columns=MultiIndex.from_tuples(columns, names=Dataset.COLUMN_LEVEL_NAMES),
        )
        dataset.index = output_labels
        if sort:
            if sorting_output:
                name, component = get_name_and_component(sorting_output)
                by = repr_variable(
                    name,
                    component,
                    size=self._get_output_n_components(name),
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
        file_path: StrPath = "",
        directory_path: StrPath = "",
        file_name: str = "",
        file_format: str = "",
        sort: bool = True,
        sorting_output: VariableType = "",
        radar_chart_settings: RadarChart_Settings | None = None,
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
                If empty, all the outputs are considered.
            input_names: The names of the inputs
                for which to display the sensitivity indices.
                If empty, all the inputs are considered.
                The input components without sensitivity indices are left out.
            standardize: Whether to scale the indices to $[0,1]$.
            title: The title of the plot, if any.
            save: Whether to save the figure.
            show: Whether to show the figure.
            file_path: The path of the file to save the figures.
                If the extension is missing, use `file_extension`.
                If empty,
                create a file path
                from `directory_path`, `file_name` and `file_extension`.
            directory_path: The path of the directory to save the figures.
                If empty, use the current working directory.
            file_name: The name of the file to save the figures.
                If empty, use a default one generated by the post-processing.
            file_format: A file extension, e.g. 'png', 'pdf', 'svg', ...
                If empty, use a default file extension.
            sort: Whether to sort the inputs
                by decreasing order of the sensitivity indices
                associated with the sorting output.
            sorting_output: The sorting output
                If empty, use the first one.
            radar_chart_settings: The settings of the radar chart.
                If `None`, use the default settings of the radar chart.

        Returns:
            A radar chart representing the sensitivity indices.

        Raises:
            ValueError: When no input component has sensitivity indices.
        """
        if radar_chart_settings is None:
            radar_chart_settings = RadarChart_Settings()

        radar_chart_settings.title = title
        radar_chart = RadarChart(
            self.__create_dataset_to_plot(
                input_names,
                outputs or self._output_names,
                standardize,
                sort,
                sorting_output,
            ),
            radar_chart_settings,
        )
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
            inputs_to_keep: The inputs to keep. If `None`, keep all.

        Returns:
            The filtered input names sorted in descending order of influence.
        """
        return filter_names(self.sort_input_variables(output), inputs_to_keep)

    def plot_comparison(
        self,
        indices: BaseGenericSensitivityAnalysis
        | Iterable[BaseGenericSensitivityAnalysis],
        output: VariableType,
        input_names: Iterable[str] = (),
        title: str = "",
        use_bar_plot: bool = True,
        save: bool = True,
        show: bool = False,
        file_path: StrPath = "",
        directory_path: StrPath = "",
        file_name: str = "",
        file_format: str = "",
    ) -> BarPlot | RadarChart:
        """Plot a comparison between the current sensitivity indices and other ones.

        This method allows to use either a bar plot (default option) or a radar chart.

        The indices of an analysis are divided by its largest index,
        ignoring the input components that have no index.

        Args:
            indices: The sensitivity indices.
            output: The output
                for which to display sensitivity indices,
                either a name or a tuple of the form (name, component).
                If name, its first component is considered.
            input_names: The inputs
                for which to display the sensitivity indices.
                If empty, all the inputs are considered.
            title: The title of the plot, if any.
            use_bar_plot: Whether to use a bar plot. Otherwise, use a radar chart.
            save: Whether to save the figure.
            show: Whether to show the figure.
            file_path: The path of the file to save the figures.
                If empty,
                create a file path
                from `directory_path`, `file_name` and `file_format`.
            directory_path: The path of the directory to save the figures.
                If empty, use the current working directory.
            file_name: The name of the file to save the figures.
                If empty, use a default one generated by the post-processing.
            file_format: A file format, e.g. 'png', 'pdf', 'svg', ...
                If empty, use a default file extension.

        Returns:
            A graph comparing sensitivity indices.
        """
        output = get_name_and_component(output)
        if isinstance(indices, BaseGenericSensitivityAnalysis):
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
        # An input component whose index is NaN must not spoil
        # the largest index of the analysis it belongs to;
        # the divisor of an analysis without any index is left at one,
        # as nanmax would reduce an all-NaN slice.
        maxima = array([
            1.0 if isnan(indices).all() else nanmax(indices) for indices in data
        ])
        dataset.update_data(
            data / maxima[:, newaxis], group_names=dataset.PARAMETER_GROUP
        )
        dataset.index = [method.main_method for method in methods]
        if use_bar_plot:
            settings = BarPlot_Settings(n_digits=2, title=title)
            plot = BarPlot(dataset, settings)
        else:
            settings = RadarChart_Settings(rmin=0.0, rmax=1.0, title=title)
            plot = RadarChart(dataset, settings)
            plot.rmin = 0.0
            plot.rmax = 1.0
        plot.execute(save, show, file_path, directory_path, file_name, file_format)
        return plot

    def to_dataset(self) -> Dataset:
        """Convert the sensitivity indices into a dataset.

        Returns:
            The sensitivity indices as a dataset.
        """
        sizes = self.dataset.variable_name_to_n_components

        row_names = []
        for input_name in self.input_names:
            for input_component in range(sizes[input_name]):
                row_names.append(  # noqa: PERF401
                    repr_variable(
                        input_name,
                        input_component,
                        size=self.dataset.variable_name_to_n_components[input_name],
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
        Thus, the standardized indices belong to the interval $[0,1]$.
        The indices that are `nan` do not contribute to the largest index
        and are left as they are.

        Args:
            indices: The indices to be standardized.

        Returns:
            The standardized indices.
        """
        new_indices = deepcopy(indices)
        for output_name, output_indices in indices.items():
            for output_component, output_component_indices in enumerate(output_indices):
                values = abs(hstack(list(output_component_indices.values())))
                if isnan(values).all():
                    # No index is a number,
                    # and so there is no largest index to divide by;
                    # nanmax would also warn when reducing an all-NaN slice.
                    continue

                max_value = nanmax(values)
                for input_name, input_indices in output_component_indices.items():
                    new_indices[output_name][output_component][input_name] = (
                        abs(input_indices) / max_value
                    )

        return new_indices

    def _get_output_names(
        self,
        output_names: str | Iterable[str],
        default_output_names: Iterable[str] = (),
    ) -> Iterable[str]:
        """Return the output names.

        Args:
            output_names: The initial output name(s).
                If empty, return the default output names.
            default_output_names: The default output names.
                If empty, use the property `default_output_names`.

        Returns:
            The output names.
        """
        if not output_names:
            return tuple(default_output_names) or self.default_output_names

        return convert_strings_to_iterable(output_names)


class BaseSensitivityAnalysis(BaseGenericSensitivityAnalysis[T]):
    """Base class for sensitivity analysis where outputs are disciplinary outputs."""

    DEFAULT_DRIVER: ClassVar[str] = ""
    """The default DOE algorithm to sample the disciplines."""

    def compute_samples(
        self,
        disciplines: Collection[Discipline],
        parameter_space: ParameterSpace,
        n_samples: int,
        output_names: str | Iterable[str] = (),
        algo_settings: BaseDOESettings | None = None,
        backup_settings: BackupSettings | None = None,
        formulation_settings: BaseFormulationSettings | None = None,
    ) -> IODataset:
        """
        Args:
            n_samples: The maximum total number of samples.
                If `0`, the number of samples is computed by the algorithm.
            output_names: The disciplines' outputs to be considered for the analysis.
                If empty, use all the outputs.
            algo_settings: The settings of the DOE algorithm.
                If `None`,
                use the default settings of the default DOE algorithm
                (see
                [DEFAULT_DRIVER][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.DEFAULT_DRIVER]).
            backup_settings: The settings of the backup file to store the samples
                if any.
            formulation_settings: The settings of the MDO formulation.
                If `None`,
                use the default settings of the MDF formulation.
        """  # noqa: D205, D212
        disciplines = list(disciplines)
        if algo_settings is None:
            algo_settings = DOE_LIBRARY_FACTORY.create_settings(self.DEFAULT_DRIVER)
        if n_samples > 0:
            algo_settings.n_samples = n_samples
        self._output_names = list(output_names or get_all_outputs(disciplines))
        self._input_names = parameter_space.variable_names
        algo_settings.use_one_line_progress_bar = True
        formulation_settings = create_model(
            MDF_Settings, settings_model=formulation_settings
        )
        scenario = EvaluationScenario(
            disciplines,
            parameter_space,
            f"{self.__class__.__name__}SamplingPhase",
            formulation_settings=formulation_settings,
        )
        for output_name in self._output_names:
            scenario.add_observable(output_name)
        if backup_settings is not None and backup_settings.file_path:
            scenario.set_backup_settings(
                backup_settings.file_path,
                at_each_iteration=backup_settings.at_each_iteration,
                at_each_function_call=backup_settings.at_each_function_call,
                erase=backup_settings.erase,
                load=backup_settings.load,
            )
        scenario.execute(algo_settings)
        self.dataset = scenario.to_dataset()
        return self.dataset
