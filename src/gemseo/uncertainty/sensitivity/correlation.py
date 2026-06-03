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
"""Sensitivity analysis based on correlation measures."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from numpy import array
from numpy import newaxis
from numpy import vstack
from openturns import Sample
from strenum import StrEnum

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.radar_chart import RadarChart
from gemseo.post.dataset.radar_chart_settings import RadarChart_Settings
from gemseo.uncertainty.sensitivity.base import BaseSensitivityAnalysis
from gemseo.utils.compatibility.openturns import PEARSON_METHOD_NAME
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.string_tools import filter_names
from gemseo.utils.string_tools import get_name_and_component
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from pathlib import Path

    from gemseo.uncertainty.sensitivity.base import FirstOrderIndicesType
    from gemseo.uncertainty.sensitivity.base import OutputsType
    from gemseo.utils.string_tools import VariableType

from openturns import CorrelationAnalysis as OTCorrelationAnalysis


class CorrelationAnalysisMethod(StrEnum):
    """A correlation analysis method."""

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


class CorrelationAnalysis(BaseSensitivityAnalysis[CorrelationAnalysisMethod]):
    """Sensitivity analysis based on correlation measures."""

    @dataclass(frozen=True)
    class SensitivityIndices:  # noqa: D106
        kendall: FirstOrderIndicesType = field(default_factory=dict)
        """The Kendall rank correlation coefficients."""

        pcc: FirstOrderIndicesType = field(default_factory=dict)
        """The Partial Correlation Coefficients."""

        pearson: FirstOrderIndicesType = field(default_factory=dict)
        """The Pearson coefficients."""

        prcc: FirstOrderIndicesType = field(default_factory=dict)
        """The Partial Rank Correlation Coefficients."""

        spearman: FirstOrderIndicesType = field(default_factory=dict)
        """The Spearman coefficients."""

        src: FirstOrderIndicesType = field(default_factory=dict)
        """The Standard Regression Coefficients."""

        srrc: FirstOrderIndicesType = field(default_factory=dict)
        """The Standard Rank Regression Coefficients."""

        ssrc: FirstOrderIndicesType = field(default_factory=dict)
        """The Squared Standard Regression Coefficients."""

    _indices: SensitivityIndices

    __METHODS_TO_OT_METHOD_NAMES: Final[dict[CorrelationAnalysisMethod, str]] = {
        CorrelationAnalysisMethod.KENDALL: "computeKendallTau",
        CorrelationAnalysisMethod.PCC: "computePCC",
        CorrelationAnalysisMethod.PEARSON: PEARSON_METHOD_NAME,
        CorrelationAnalysisMethod.PRCC: "computePRCC",
        CorrelationAnalysisMethod.SPEARMAN: "computeSpearmanCorrelation",
        CorrelationAnalysisMethod.SRC: "computeSRC",
        CorrelationAnalysisMethod.SRRC: "computeSRRC",
        CorrelationAnalysisMethod.SSRC: "computeSquaredSRC",
    }
    """The mapping from the sensitivity method names to the OpenTURNS method names."""

    _DEFAULT_MAIN_METHOD: ClassVar[CorrelationAnalysisMethod] = (
        CorrelationAnalysisMethod.SPEARMAN
    )

    DEFAULT_DRIVER: ClassVar[str] = "OT_MONTE_CARLO"

    def compute_indices(  # noqa: D102
        self, output_names: str | Sequence[str] = ()
    ) -> SensitivityIndices:
        output_names = self._get_output_names(output_names)

        input_samples = Sample(
            self.dataset.get_view(group_names=self.dataset.INPUT_GROUP).to_numpy()
        )
        correlation_analyses = {
            output_name: [
                None
                if (data := output_component_samples[:, newaxis]).var() == 0.0
                else OTCorrelationAnalysis(input_samples, Sample(data))
                # For each component of the output variable
                for output_component_samples in self.dataset
                .get_view(
                    group_names=self.dataset.OUTPUT_GROUP,
                    variable_names=output_name,
                )
                .to_numpy()
                .T
            ]
            # For each output variable
            for output_name in output_names
        }
        indices = {}
        # For each correlation method
        sizes = self.dataset.variable_name_to_n_components
        for method in CorrelationAnalysisMethod:
            # The version of OpenTURNS offers this correlation method.
            method_name = self.__METHODS_TO_OT_METHOD_NAMES[method]
            indices[str(method).lower()] = {
                output_name: [
                    None
                    if correlation_analysis is None
                    else split_array_to_dict_of_arrays(
                        array(getattr(correlation_analysis, method_name)()),
                        sizes,
                        self._input_names,
                    )
                    # For each component of the output variable
                    for correlation_analysis in correlation_analyses[output_name]
                ]
                # For each output variable
                for output_name in output_names
            }

        self._indices = self.SensitivityIndices(**indices)
        return self._indices

    def plot(  # noqa: D102
        self,
        output: VariableType,
        input_names: Iterable[str] = (),
        title: str = "",
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_format: str = "",
    ) -> RadarChart:
        """
        Args:
            directory_path: The path to the directory where to save the plots.
            file_name: The name of the file.
        """  # noqa: D212, D205
        output_name, output_index = get_name_and_component(output)

        all_indices = tuple(CorrelationAnalysisMethod)
        dataset = Dataset()
        for input_name in filter_names(self._input_names, input_names):
            # Store all the sensitivity indices
            # related to the tuple (output_name, output_index, input_name)
            # in a 2D NumPy array shaped as (n_indices, input_dimension).
            dataset.add_variable(
                input_name,
                vstack([
                    getattr(self.indices, method.lower())[output_name][output_index][
                        input_name
                    ]
                    for method in all_indices
                ]),
            )

        dataset.index = all_indices
        output_name = repr_variable(
            output_name,
            output_index,
            size=self.dataset.variable_name_to_n_components[output_name],
        )
        settings = RadarChart_Settings(
            title=title or f"Correlation indices for the output {output_name}",
            rmin=-1.0,
            rmax=1.0,
        )
        radar_chart = RadarChart(dataset, settings)
        file_path = self._file_path_manager.create_file_path(
            file_path=file_path,
            directory_path=directory_path,
            file_name=file_name,
            file_extension=file_format,
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

    def plot_radar(  # noqa: D102
        self,
        outputs: OutputsType = (),
        input_names: Iterable[str] = (),
        title: str = "",
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_format: str = "",
        radar_chart_settings: RadarChart_Settings | None = None,
    ) -> RadarChart:
        """
        Args:
            radar_chart_settings: The settings of the radar chart.
                If `None`,
                use the default settings of the radar chart,
                except for the minimum and maximum radius values,
                which are set to -1.0 and 1.0, respectively.
        """  # noqa: D205, D212
        if radar_chart_settings is None:
            radar_chart_settings = RadarChart_Settings()
        if "rmin" not in radar_chart_settings.model_fields_set:
            radar_chart_settings.rmin = -1.0
        if "rmax" not in radar_chart_settings.model_fields_set:
            radar_chart_settings.rmax = 1.0
        return super().plot_radar(
            outputs,
            input_names=input_names,
            title=title,
            save=save,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
            radar_chart_settings=radar_chart_settings,
        )
