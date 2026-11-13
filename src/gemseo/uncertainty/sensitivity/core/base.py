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
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Generic
from typing import TypeVar

from numpy import array
from numpy import full
from numpy import hstack
from numpy import isnan
from numpy import linspace
from numpy import nan
from numpy import nanmax
from numpy import nansum
from numpy import newaxis
from numpy import vstack
from pandas import MultiIndex

from gemseo.dataset.dataset import Dataset
from gemseo.dataset.io_dataset import IODataset
from gemseo.doe.factory import doe_library_factory
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
from gemseo.util.constant import read_only_empty_dict
from gemseo.util.data_conversion import split_array_to_dict_of_arrays
from gemseo.util.discipline import get_all_outputs
from gemseo.util.file_path_manager import FilePathManager
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta
from gemseo.util.pydantic import BaseSettings
from gemseo.util.pydantic import create_model
from gemseo.util.pydantic import log_replaced_setting
from gemseo.util.string import convert_strings_to_iterable
from gemseo.util.string import filter_names
from gemseo.util.string import get_variables_with_components
from gemseo.util.string import pretty_repr
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
    from gemseo.space.random import RandomSpace
    from gemseo.util.string import VariableOrComponent
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
    a [RandomSpace][gemseo.space.random.RandomSpace]
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

    _interaction_methods: ClassVar[tuple[str, ...]] = ()
    """The names of the sensitivity methods considering interaction effects."""

    _default_main_method: ClassVar[StrEnum]
    """The name of the default main sensitivity analysis method."""

    _input_names: list[str]
    """The names of the inputs in random space order."""

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
        self.main_method = self._default_main_method
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
        random_space: RandomSpace,
        algo_settings: BaseSettings | None = None,
        formulation_settings: BaseFormulationSettings | None = None,
    ) -> IODataset:
        """Sample the model over the random space.

        This step is a prerequisite for calculating the sensitivity indices
        if the samples were not passed during instantiation.

        Args:
            disciplines: The discipline or disciplines to use for the analysis.
            random_space: A random space.
            algo_settings: The settings of the DOE algorithm.
                If `None`,
                use the default settings of the default DOE algorithm
                (see
                [default_driver][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.default_driver]).
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
        return self.dataset.get_view(group_names=self.dataset.input_group).to_numpy()

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
                group_names=dataset.output_group, variable_names=output_name
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

    def _check_output(
        self, name: str, component: int | None = None, analysis: str = ""
    ) -> None:
        """Check that an output, or one of its components, exists.

        Args:
            name: The name of the output.
            component: The component of the output.
                If `None`, only check the name.
            analysis: The description of the analysis used in the error message.
                If empty, use the class name of the analysis.

        Raises:
            ValueError: When `name` is not an output of the analysis
                or when `component` is not in `[0, size)`,
                where `size` is the size of the output;
                a negative component is not counted from the last one.
        """
        output_indices = self.main_indices.get(name)
        if output_indices is None:
            raise self._create_unknown_output_error(name, analysis)

        analysis = analysis or self.__get_analysis_description()
        if component is not None and not 0 <= component < len(output_indices):
            msg = f"The output {name!r} of {analysis} has no component {component}."
            raise ValueError(msg)

    def _create_unknown_output_error(self, name: str, analysis: str = "") -> ValueError:
        """Create the error of an output that is not an output of the analysis.

        Args:
            name: The name of the output.
            analysis: The description of the analysis used in the error message.
                If empty, use the class name of the analysis.

        Returns:
            The error.
        """
        analysis = analysis or self.__get_analysis_description()
        return ValueError(f"The output {name!r} is not an output of {analysis}.")

    def __get_analysis_description(self) -> str:
        """Return the default description of the analysis in the error messages.

        Returns:
            The default description of the analysis.
        """
        return f"the sensitivity analysis ({self.__class__.__name__})"

    def _get_output_component_indices(
        self,
        indices: FirstOrderIndicesType | SecondOrderIndicesType,
        output_name: str,
        output_component: int,
        use_nan: bool = False,
    ) -> dict[str, RealArray] | dict[str, dict[str, RealArray]]:
        """Return the sensitivity indices of an output component.

        Args:
            indices: The sensitivity indices of a method.
            output_name: The name of the output.
            output_component: The component of the output.
            use_nan: Whether to return `nan` indices
                for a constant output component
                (first-order indices only).
                Otherwise, raise an error.

        Returns:
            The sensitivity indices of the output component indexed by input name.

        Raises:
            ValueError: When the sensitivity indices are undefined,
                as the output component is constant,
                and `use_nan` is `False`.
        """
        component_indices = indices[output_name][output_component]
        if component_indices is not None:
            return component_indices

        if use_nan:
            sizes = self.dataset.variable_name_to_n_components
            return {name: full(sizes[name], nan) for name in self._input_names}

        raise self._create_undefined_indices_error([(output_name, output_component)])

    def _create_undefined_indices_error(
        self, outputs: Iterable[tuple[str, int]]
    ) -> ValueError:
        """Create the error of undefined sensitivity indices.

        Args:
            outputs: The (name, component) pairs of the constant output components.

        Returns:
            The error stating that the variance of these outputs is zero.
        """
        name_to_components: dict[str, list[int]] = {}
        for name, component in outputs:
            name_to_components.setdefault(name, []).append(component)

        labels = []
        for name, components in name_to_components.items():
            size = self._get_output_n_components(name)
            if sorted(components) == list(range(size)):
                # All the components are constant; the output is named once.
                labels.append(repr(name))
            else:
                labels.extend(
                    repr(repr_variable(name, component, size=size))
                    for component in components
                )

        msg = (
            f"The sensitivity indices of {pretty_str(labels, sort=False)} "
            "are undefined, as the variance of "
            f"{'these outputs' if len(labels) > 1 else 'the output'} is zero."
        )
        return ValueError(msg)

    def _get_output_component_and_indices(
        self,
        output: VariableOrComponent | Sequence[str | int],
        indices: FirstOrderIndicesType | SecondOrderIndicesType,
        components: Iterable[int] = (),
    ) -> tuple[str, int, dict[str, RealArray] | dict[str, dict[str, RealArray]]]:
        """Return an output component and its sensitivity indices.

        Args:
            output: Either the name of an output,
                a `(name, component)` pair
                or a `[name, component]` sequence.
            indices: The sensitivity indices of a method.
            components: The components of the output to choose from
                when `output` is a name.
                If empty, all the components of the output are considered.

        Returns:
            The name of the output,
            either its component if `output` is a pair
            or its first non-constant component among `components`
            if `output` is a name,
            and the sensitivity indices of this output component.

        Raises:
            ValueError: When the output is unknown,
                when `output` is a pair whose output component does not exist
                (a negative component is not counted from the last one)
                or is constant
                or when `output` is a name
                and all the components of this output among `components`
                are constant.
        """
        if not isinstance(output, str):
            name, component = output
            self._check_output(name, component)
            return (
                name,
                component,
                self._get_output_component_indices(indices, name, component),
            )

        name = output
        self._check_output(name)
        output_indices = indices[name]
        components = list(components) or range(len(output_indices))
        for component in components:
            component_indices = output_indices[component]
            if component_indices is not None:
                return name, component, component_indices

        raise self._create_undefined_indices_error([
            (name, component) for component in components
        ])

    def sort_input_variables(self, output: VariableOrComponent) -> list[str]:
        """Return the inputs sorted in descending order.

        Args:
            output: Either a tuple as `(output_name, output_component)`
                or an output name;
                in the second case, use its first non-constant component.

        Returns:
            The names of the inputs sorted by cumulative sensitivity index,
            which is the sum of the absolute values of the sensitivity indices
            associated to the different components of an input,
            skipping the components whose index is `nan`;
            an input whose every component has a `nan` index is ranked last.

        Raises:
            ValueError: When the output is unknown,
                when the output component does not exist
                (a negative component is not counted from the last one),
                when the output component is constant,
                or when all the components of the output are constant
                if it is a name.
        """
        _, _, component_indices = self._get_output_component_and_indices(
            output, self.main_indices
        )
        return self._sort_input_names(component_indices)

    @staticmethod
    def _sort_input_names(
        component_indices: Mapping[str, RealArray], use_absolute_values: bool = True
    ) -> list[str]:
        """Return the inputs sorted in descending order.

        Args:
            component_indices: The sensitivity indices of an output component
                indexed by input name.
            use_absolute_values: Whether the cumulative sensitivity index
                sums the absolute values of the sensitivity indices.
                Otherwise, it sums the sensitivity indices.

        Returns:
            The names of the inputs sorted by cumulative sensitivity index,
            skipping the components whose index is `nan`;
            an input whose every component has a `nan` index is ranked last.
        """
        return [
            input_name
            for input_name, _ in sorted(
                component_indices.items(),
                key=lambda indices: (
                    not isnan(indices[1]).all(),
                    nansum(abs(indices[1]) if use_absolute_values else indices[1]),
                ),
                reverse=True,
            )
        ]

    def plot(
        self,
        output: VariableOrComponent,
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
                If it is a name, all its non-constant components are considered.
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

        Raises:
            ValueError: When the output is unknown,
                when the output component does not exist
                (a negative component is not counted from the last one),
                when the output component is constant,
                or when all the components of the output are constant
                if it is a name.
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
        output: VariableOrComponent | Sequence[str | int],
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
        properties: Mapping[str, DatasetPlotPropertyType] = read_only_empty_dict,
    ) -> Curves | Surfaces:
        """Plot the sensitivity indices related to a 1D or 2D functional output.

        The output is considered as a 1D or 2D functional variable,
        according to the shape of the mesh on which it is represented.
        Each input component is a curve or a surface.
        The mesh points where an output component is constant
        have no sensitivity index (`nan`), which shows as a gap.

        Args:
            output: The output
                for which to display sensitivity indices,
                either a name,
                a tuple of the form `(name, component)`
                or a sequence of the form `[name, component]`,
                where the output component is used to sort the inputs.
                If it is a name, its first non-constant component is considered.
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
            ValueError: When the output is unknown,
                when the output component does not exist
                (a negative component is not counted from the last one),
                when the sorting output component is constant,
                when all the components of the output are constant
                if it is a name,
                or when none of the input names are inputs.
        """
        main_indices = self.main_indices
        output_name, output_component, _ = self._get_output_component_and_indices(
            output, main_indices
        )
        requested_input_names = list(input_names)
        input_names = self._filter_sorted_input_names(
            (output_name, output_component), requested_input_names
        )
        if not input_names:
            msg = (
                f"None of the input names {pretty_repr(requested_input_names)} "
                "is an input of the sensitivity analysis, "
                f"whose inputs are {pretty_repr(self._input_names)}."
            )
            raise ValueError(msg)

        output_indices = main_indices[output_name]
        if standardize:
            output_indices = self.standardize_indices({output_name: output_indices})[
                output_name
            ]

        # A constant output component has no index,
        # which is a gap in the plot.
        all_component_indices = [
            self._get_output_component_indices(
                {output_name: output_indices}, output_name, component, True
            )
            for component in range(len(output_indices))
        ]
        labels = []
        data = []
        for input_name in input_names:
            labels.extend(self.dataset.get_columns([input_name]))
            data.extend(
                array([
                    component_indices[input_name]
                    for component_indices in all_component_indices
                ]).T.tolist()
            )

        data = array(data)
        dataset = Dataset.from_array(data, [output_name], {output_name: data.shape[1]})
        dataset.index = labels
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
        sorting_output: VariableOrComponent = "",
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
                The constant output components are left out,
                as their sensitivity indices are undefined.
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
            sorting_output: The sorting output,
                either a name or a tuple of the form `(name, component)`,
                among the plotted output components.
                If it is a name,
                its first non-constant component among the plotted ones is considered.
                If empty, use the first one.
            bar_plot_settings: The settings of the bar plot.
                If `None`,
                use the default bar plot settings,
                except for the number of digits,
                which is set to 2.

        Returns:
            A bar plot representing the sensitivity indices.

        Raises:
            ValueError: When an output or the sorting output is unknown
                or has no such component
                (a negative component is not counted from the last one),
                when no input component has sensitivity indices,
                when all the output components are constant,
                when the sorting output is constant
                or when the sorting output is not among the plotted outputs.
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
        sorting_output: VariableOrComponent,
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
            sorting_output: The sorting output,
                either a name or a tuple of the form `(name, component)`,
                among the plotted output components.
                If it is a name,
                its first non-constant component among the plotted ones is considered.
                If empty, use the first one.

        Returns:
            The dataset to plot,
            without the input components that have no sensitivity indices
            and without the constant output components.

        Raises:
            ValueError: When an output is unknown
                or has no such component
                (a negative component is not counted from the last one),
                when no input component has sensitivity indices,
                when all the output components are constant,
                when the sorting output is constant
                or when the sorting output is not among the plotted outputs.
        """
        raw_main_indices = self.main_indices
        sizes = {k: len(v) for k, v in raw_main_indices.items()}
        try:
            # An output component listed twice is plotted once.
            all_outputs = list(
                dict.fromkeys(get_variables_with_components(outputs, sizes))
            )
        except KeyError as error:
            # An output given by an unknown name has no size.
            raise self._create_unknown_output_error(error.args[0]) from None

        for name, component in all_outputs:
            self._check_output(name, component)

        # The sensitivity indices of a constant output component are undefined.
        outputs = [
            (name, component)
            for name, component in all_outputs
            if raw_main_indices[name][component] is not None
        ]
        if not outputs:
            raise self._create_undefined_indices_error(all_outputs)

        if standardize:
            main_indices = self.standardize_indices({
                name: raw_main_indices[name] for name, _ in outputs
            })
        else:
            main_indices = raw_main_indices

        if sort and sorting_output:
            sorting_output = self.__get_sorting_output(sorting_output, all_outputs)

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

                columns.append((Dataset.parameter_group, input_name, component))
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
            columns=MultiIndex.from_tuples(columns, names=Dataset.column_level_names),
        )
        dataset.index = output_labels
        if sort:
            if sorting_output:
                name, component = sorting_output
                by = repr_variable(
                    name,
                    component,
                    size=self._get_output_n_components(name),
                )
            else:
                by = dataset.index[0]
            dataset = dataset.sort_values(by=by, ascending=False, axis=1)

        return dataset

    def __get_sorting_output(
        self,
        sorting_output: VariableOrComponent,
        outputs: Sequence[tuple[str, int]],
    ) -> tuple[str, int]:
        """Return the output component to sort the inputs.

        Args:
            sorting_output: The sorting output,
                either a name or a tuple of the form `(name, component)`.
            outputs: The output components to plot, including the constant ones.

        Returns:
            Either the sorting output if it is a tuple
            or the first non-constant component to plot of this output if it is a name.

        Raises:
            ValueError: When the sorting output is unknown or has no such component
                (a negative component is not counted from the last one),
                is constant
                or is not among the output components to plot.
        """
        if isinstance(sorting_output, str):
            self._check_output(sorting_output)
            name = sorting_output
            is_plotted = any(output[0] == name for output in outputs)
            label = name
        else:
            name, component = sorting_output
            self._check_output(name, component)
            is_plotted = (name, component) in outputs
            label = repr_variable(
                name, component, size=self._get_output_n_components(name)
            )

        if not is_plotted:
            msg = f"The sorting output {label!r} is not among the plotted outputs."
            raise ValueError(msg)

        name, component, _ = self._get_output_component_and_indices(
            sorting_output,
            self.main_indices,
            [output[1] for output in outputs if output[0] == name],
        )
        return name, component

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
        sorting_output: VariableOrComponent = "",
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
                The constant output components are left out,
                as their sensitivity indices are undefined.
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
            sorting_output: The sorting output,
                either a name or a tuple of the form `(name, component)`,
                among the plotted output components.
                If it is a name,
                its first non-constant component among the plotted ones is considered.
                If empty, use the first one.
            radar_chart_settings: The settings of the radar chart.
                If `None`, use the default settings of the radar chart.

        Returns:
            A radar chart representing the sensitivity indices.

        Raises:
            ValueError: When an output or the sorting output is unknown
                or has no such component
                (a negative component is not counted from the last one),
                when no input component has sensitivity indices,
                when all the output components are constant,
                when the sorting output is constant
                or when the sorting output is not among the plotted outputs.
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
            output: A non-constant output component to sort the inputs.
            inputs_to_keep: The inputs to keep. If empty, keep all.

        Returns:
            The filtered input names sorted in descending order of influence.
        """
        return filter_names(self.sort_input_variables(output), inputs_to_keep)

    @staticmethod
    def __describe_analysis(
        analysis: BaseGenericSensitivityAnalysis, position: int
    ) -> str:
        """Describe an analysis of [plot_comparison()][.plot_comparison].

        Args:
            analysis: The analysis.
            position: The position of the analysis,
                0 for the current analysis
                and `i + 1` for the `i`-th compared analysis.

        Returns:
            The description of the analysis.
        """
        class_name = analysis.__class__.__name__
        if position == 0:
            return f"the current sensitivity analysis ({class_name})"

        return (
            f"the compared sensitivity analysis at position {position - 1} "
            f"({class_name})"
        )

    def __create_no_common_component_error(
        self,
        output_name: str,
        analyses: Sequence[BaseGenericSensitivityAnalysis],
    ) -> ValueError:
        """Create the error of an output without component defined for all analyses.

        Args:
            output_name: The name of the output.
            analyses: The compared analyses, the current one first.

        Returns:
            The error naming, for each component,
            the analyses for which it is constant.
        """
        all_output_indices = [
            analysis.main_indices[output_name] for analysis in analyses
        ]
        size = self._get_output_n_components(output_name)
        sentences = []
        for component in range(min(map(len, all_output_indices))):
            descriptions = [
                self.__describe_analysis(analysis, position)
                for position, (analysis, output_indices) in enumerate(
                    zip(analyses, all_output_indices, strict=True)
                )
                if output_indices[component] is None
            ]
            label = repr(repr_variable(output_name, component, size=size))
            verb = "is constant for" if not sentences else "for"
            sentences.append(f"{label} {verb} {pretty_str(descriptions, sort=False)}")

        msg = (
            f"No component of the output {output_name!r} has sensitivity indices "
            "defined for all the compared analyses: "
            f"{pretty_str(sentences, sort=False)}."
        )
        return ValueError(msg)

    def plot_comparison(
        self,
        indices: BaseGenericSensitivityAnalysis
        | Iterable[BaseGenericSensitivityAnalysis],
        output: VariableOrComponent,
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
                If it is a name,
                its first component that is not constant
                for the current analysis and the compared ones
                is considered.
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

        Raises:
            ValueError: When the output is not an output of one of the analyses,
                when the output component does not exist
                for one of the analyses if it is a tuple
                (a negative component is not counted from the last one),
                when the output component is constant
                for one of the analyses
                or when none of the components of the output is non-constant
                for all the analyses if it is a name.
        """
        if isinstance(indices, BaseGenericSensitivityAnalysis):
            indices = [indices]
        methods = [self, *indices]
        if isinstance(output, str):
            output_name, output_component = output, None
        else:
            output_name, output_component = output

        for position, method in enumerate(methods):
            method._check_output(
                output_name,
                output_component,
                self.__describe_analysis(method, position),
            )

        all_output_indices = [method.main_indices[output_name] for method in methods]
        if output_component is None:
            # A bare name means the first component defined for every analysis.
            output_component = next(
                (
                    component
                    for component in range(min(map(len, all_output_indices)))
                    if all(
                        output_indices[component] is not None
                        for output_indices in all_output_indices
                    )
                ),
                None,
            )
            if output_component is None:
                raise self.__create_no_common_component_error(output_name, methods)
        else:
            for position, (method, output_indices) in enumerate(
                zip(methods, all_output_indices, strict=True)
            ):
                if output_indices[output_component] is None:
                    error = method._create_undefined_indices_error([
                        (output_name, output_component)
                    ])
                    if position == 0:
                        raise error

                    description = self.__describe_analysis(method, position)
                    msg = f"{error} This is the case for {description}."
                    raise ValueError(msg)

        component_indices = [
            output_indices[output_component] for output_indices in all_output_indices
        ]

        dataset = Dataset()
        input_names = self._filter_sorted_input_names(
            (output_name, output_component), input_names
        )
        for input_name in input_names:
            data = abs(
                array([
                    method_indices[input_name] for method_indices in component_indices
                ])
            )
            dataset.add_variable(input_name, data)
        data = dataset.get_view(group_names=dataset.parameter_group).to_numpy()
        # An input component whose index is NaN must not spoil
        # the largest index of the analysis it belongs to;
        # the divisor of an analysis without any index is left at one,
        # as nanmax would reduce an all-NaN slice.
        maxima = array([
            1.0 if isnan(indices).all() else nanmax(indices) for indices in data
        ])
        dataset.update_data(
            data / maxima[:, newaxis], group_names=dataset.parameter_group
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
            The sensitivity indices of a constant output component are `nan`.
        """
        sizes = self.dataset.variable_name_to_n_components

        row_names = []
        for input_name in self.input_names:
            for input_component in range(sizes[input_name]):
                row_names.append(  # noqa: PERF401
                    repr_variable(
                        input_name,
                        input_component,
                        size=sizes[input_name],
                    )
                )

        dataset = Dataset()
        for method, indices in asdict(self.indices).items():
            if method in self._interaction_methods:
                dataset.misc[method] = indices
                continue

            variables = []
            sizes = {}
            data = []
            for output, components in indices.items():
                variables.append(output)
                sizes[output] = len(components)
                for component in range(len(components)):
                    component_indices = self._get_output_component_indices(
                        indices, output, component, True
                    )
                    data.append([
                        item
                        for name in self.input_names
                        for item in component_indices[name].tolist()
                    ])
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
        and are left as they are,
        and so are the undefined indices (`None`) of a constant output component.

        Args:
            indices: The indices to be standardized.

        Returns:
            The standardized indices.
        """
        new_indices = deepcopy(indices)
        for output_name, output_indices in indices.items():
            for output_component, output_component_indices in enumerate(output_indices):
                if output_component_indices is None:
                    continue

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

    default_driver: ClassVar[str] = ""
    """The default DOE algorithm to sample the disciplines."""

    def compute_samples(
        self,
        disciplines: Collection[Discipline],
        random_space: RandomSpace,
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
                [default_driver][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.default_driver]).
            backup_settings: The settings of the backup file to store the samples
                if any.
            formulation_settings: The settings of the MDO formulation.
                If `None`,
                use the default settings of the MDF formulation.
        """  # noqa: D205, D212
        disciplines = list(disciplines)
        if algo_settings is None:
            algo_settings = doe_library_factory.create_settings(self.default_driver)
        update = {"use_one_line_progress_bar": True}
        if n_samples > 0:
            log_replaced_setting(
                algo_settings, "n_samples", n_samples, "the argument 'n_samples'"
            )
            update["n_samples"] = n_samples

        # Copy the settings to leave those of the user unchanged;
        # the assignments are validated.
        algo_settings = algo_settings.model_copy()
        for name, value in update.items():
            setattr(algo_settings, name, value)

        self._output_names = list(output_names or get_all_outputs(disciplines))
        self._input_names = list(random_space.variables)
        formulation_settings = create_model(
            MDF_Settings, settings_model=formulation_settings
        )
        scenario = EvaluationScenario(
            disciplines,
            random_space,
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
