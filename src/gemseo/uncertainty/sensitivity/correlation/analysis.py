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
#
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Class for the estimation of various correlation coefficients."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Final

from numpy import array
from numpy import newaxis
from numpy import vstack
from openturns import Sample
from strenum import StrEnum

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.radar_chart import RadarChart
from gemseo.uncertainty.sensitivity.analysis import FirstOrderIndicesType
from gemseo.uncertainty.sensitivity.analysis import OutputsType
from gemseo.uncertainty.sensitivity.analysis import SensitivityAnalysis
from gemseo.utils.compatibility.openturns import IS_OT_LOWER_THAN_1_20
from gemseo.utils.compatibility.openturns import compute_kendall_tau
from gemseo.utils.compatibility.openturns import compute_pcc
from gemseo.utils.compatibility.openturns import compute_pearson_correlation
from gemseo.utils.compatibility.openturns import compute_prcc
from gemseo.utils.compatibility.openturns import compute_spearman_correlation
from gemseo.utils.compatibility.openturns import compute_squared_src
from gemseo.utils.compatibility.openturns import compute_src
from gemseo.utils.compatibility.openturns import compute_srrc
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from gemseo.algos.doe.doe_library import DOELibraryOptionType
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline import MDODiscipline
    from gemseo.post.dataset.dataset_plot import VariableType


class CorrelationAnalysis(SensitivityAnalysis):
    """Sensitivity analysis based on indices using correlation measures.

    Examples:
        >>> from numpy import pi
        >>> from gemseo import create_discipline, create_parameter_space
        >>> from gemseo.uncertainty.sensitivity.correlation.analysis import (
        ...     CorrelationAnalysis,
        ... )
        >>>
        >>> expressions = {"y": "sin(x1)+7*sin(x2)**2+0.1*x3**4*sin(x1)"}
        >>> discipline = create_discipline(
        ...     "AnalyticDiscipline", expressions=expressions
        ... )
        >>>
        >>> parameter_space = create_parameter_space()
        >>> parameter_space.add_random_variable(
        ...     "x1", "OTUniformDistribution", minimum=-pi, maximum=pi
        ... )
        >>> parameter_space.add_random_variable(
        ...     "x2", "OTUniformDistribution", minimum=-pi, maximum=pi
        ... )
        >>> parameter_space.add_random_variable(
        ...     "x3", "OTUniformDistribution", minimum=-pi, maximum=pi
        ... )
        >>>
        >>> analysis = CorrelationAnalysis(
        ...     [discipline], parameter_space, n_samples=1000
        ... )
        >>> indices = analysis.compute_indices()
    """  # noqa: E501

    class Method(StrEnum):
        """The names of the sensitivity methods."""

        KENDALL = "Kendall"
        """The Kendall rank correlation coefficient."""
        PCC = "PCC"
        """The partial correlation coefficient."""
        PEARSON = "Pearson"
        """The Pearson coefficient."""
        PRCC = "PRCC"
        """The partial rank correlation coefficient."""
        SPEARMAN = "Spearman"
        """The Spearman coefficient."""
        SRC = "SRC"
        """The standard regression coefficient."""
        SRRC = "SRRC"
        """The standard rank regression coefficient."""
        SSRC = "SSRC"
        """The squared standard regression coefficient."""

    __METHODS_TO_FUNCTIONS: Final[dict[Method, Callable]] = {
        Method.KENDALL: compute_kendall_tau,
        Method.PCC: compute_pcc,
        Method.PEARSON: compute_pearson_correlation,
        Method.PRCC: compute_prcc,
        Method.SPEARMAN: compute_spearman_correlation,
        Method.SRC: compute_src,
        Method.SRRC: compute_srrc,
        Method.SSRC: compute_squared_src,
    }

    DEFAULT_DRIVER = "OT_MONTE_CARLO"

    def __init__(  # noqa: D107
        self,
        disciplines: Collection[MDODiscipline],
        parameter_space: ParameterSpace,
        n_samples: int,
        output_names: Iterable[str] = (),
        algo: str = "",
        algo_options: Mapping[str, DOELibraryOptionType] = MappingProxyType({}),
        formulation: str = "MDF",
        **formulation_options: Any,
    ) -> None:
        super().__init__(
            disciplines,
            parameter_space,
            n_samples=n_samples,
            output_names=output_names,
            algo=algo,
            algo_options=algo_options,
            formulation=formulation,
            **formulation_options,
        )
        self.main_method = self.Method.SPEARMAN

    def compute_indices(  # noqa: D102
        self, outputs: str | Sequence[str] = ()
    ) -> dict[str, FirstOrderIndicesType]:
        output_names = outputs or self.default_output
        if isinstance(output_names, str):
            output_names = [output_names]

        input_samples = Sample(
            self.dataset.get_view(group_names=self.dataset.INPUT_GROUP).to_numpy()
        )
        self._indices = {}
        # For each correlation method
        new_methods = [self.Method.KENDALL, self.Method.SSRC]
        for method in self.Method:
            if IS_OT_LOWER_THAN_1_20 and method in new_methods:
                continue

            # The version of OpenTURNS offers this correlation method.
            get_indices = self.__METHODS_TO_FUNCTIONS[method]
            sizes = self.dataset.variable_names_to_n_components
            self._indices[method] = {
                output_name: [
                    split_array_to_dict_of_arrays(
                        array(
                            get_indices(
                                input_samples,
                                Sample(output_component_samples[:, newaxis]),
                            )
                        ),
                        sizes,
                        self._input_names,
                    )
                    # For each component of the output variable
                    for output_component_samples in self.dataset.get_view(
                        group_names=self.dataset.OUTPUT_GROUP,
                        variable_names=output_name,
                    )
                    .to_numpy()
                    .T
                ]
                # For each output variable
                for output_name in output_names
            }

        return self._indices

    @property
    def pcc(self) -> FirstOrderIndicesType:
        """The Partial Correlation Coefficients.

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
        return self._indices[self.Method.PCC]

    @property
    def prcc(self) -> FirstOrderIndicesType:
        """The Partial Rank Correlation Coefficients.

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
        return self._indices[self.Method.PRCC]

    @property
    def src(self) -> FirstOrderIndicesType:
        """The Standard Regression Coefficients.

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
        return self._indices[self.Method.SRC]

    @property
    def ssrc(self) -> FirstOrderIndicesType:
        """The Squared Standard Regression Coefficients.

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
        return self._indices.get(self.Method.SSRC, {})

    @property
    def kendall(self) -> FirstOrderIndicesType:
        """The Kendall rank correlation coefficients.

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
        return self._indices.get(self.Method.KENDALL, {})

    @property
    def srrc(self) -> FirstOrderIndicesType:
        """The Standard Rank Regression Coefficients.

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
        return self._indices[self.Method.SRRC]

    @property
    def pearson(self) -> FirstOrderIndicesType:
        """The Pearson coefficients.

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
        return self._indices[self.Method.PEARSON]

    @property
    def spearman(self) -> FirstOrderIndicesType:
        """The Spearman coefficients.

         ith the following structure:

        .. code-block:: python

            {
                "output_name": [
                    {
                        "input_name": data_array,
                    }
                ]
            }
        """
        return self._indices[self.Method.SPEARMAN]

    def plot(  # noqa: D102
        self,
        output: VariableType,
        inputs: Iterable[str] = (),
        title: str = "",
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_format: str = "",
    ) -> None:
        """
        Args:
            directory_path: The path to the directory where to save the plots.
            file_name: The name of the file.
        """  # noqa: D212, D205
        if isinstance(output, str):
            output_name, output_index = output, 0
        else:
            output_name, output_index = output

        all_indices = tuple(self.Method)
        dataset = Dataset()
        for input_name in self._filter_names(self._input_names, inputs):
            # Store all the sensitivity indices
            # related to the tuple (output_name, output_index, input_name)
            # in a 2D NumPy array shaped as (n_indices, input_dimension).
            dataset.add_variable(
                input_name,
                vstack([
                    getattr(self, method.lower())[output_name][output_index][input_name]
                    for method in all_indices
                ]),
            )

        dataset.index = all_indices
        plot = RadarChart(dataset)
        output_name = repr_variable(
            output_name,
            output_index,
            size=self.dataset.variable_names_to_n_components[output_name],
        )
        plot.title = title or f"Correlation indices for the output {output_name}"
        plot.rmin = -1.0
        plot.rmax = 1.0
        file_path = self._file_path_manager.create_file_path(
            file_path=file_path,
            directory_path=directory_path,
            file_name=file_name,
            file_extension=file_format,
        )
        plot.execute(
            save=save,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
        )

    def plot_radar(  # noqa: D102
        self,
        outputs: OutputsType = (),
        inputs: Iterable[str] = (),
        title: str = "",
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_format: str = "",
        min_radius: float = -1.0,
        max_radius: float = 1.0,
        **options: bool,
    ) -> RadarChart:
        return super().plot_radar(
            outputs,
            inputs=inputs,
            title=title,
            save=save,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
            min_radius=min_radius,
            max_radius=max_radius,
            **options,
        )

    @staticmethod
    def _aggregate_sensitivity_indices(indices: NDArray[float]) -> float:  # noqa: D102
        return abs(indices).sum()
