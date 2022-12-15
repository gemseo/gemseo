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

import logging
from pathlib import Path
from typing import Any
from typing import Collection
from typing import Iterable
from typing import Mapping
from typing import Sequence

from numpy import array
from numpy import vstack
from openturns import Sample

from gemseo.algos.doe.doe_lib import DOELibraryOptionType
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.post.dataset.radar_chart import RadarChart
from gemseo.uncertainty.sensitivity.analysis import IndicesType
from gemseo.uncertainty.sensitivity.analysis import OutputsType
from gemseo.uncertainty.sensitivity.analysis import SensitivityAnalysis
from gemseo.utils.compatibility.openturns import compute_pcc
from gemseo.utils.compatibility.openturns import compute_pearson_correlation
from gemseo.utils.compatibility.openturns import compute_prcc
from gemseo.utils.compatibility.openturns import compute_signed_src
from gemseo.utils.compatibility.openturns import compute_spearman_correlation
from gemseo.utils.compatibility.openturns import compute_src
from gemseo.utils.compatibility.openturns import compute_srrc
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

LOGGER = logging.getLogger(__name__)


class CorrelationAnalysis(SensitivityAnalysis):
    """Sensitivity analysis based on indices using correlation measures.

    Examples:
        >>> from numpy import pi
        >>> from gemseo.api import create_discipline, create_parameter_space
        >>> from gemseo.uncertainty.sensitivity.correlation.analysis import (
        ...     CorrelationAnalysis
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
        >>> analysis = CorrelationAnalysis([discipline], parameter_space, n_samples=1000)
        >>> indices = analysis.compute_indices()
    """

    _PEARSON = "pearson"
    _SPEARMAN = "spearman"
    _PCC = "pcc"
    _PRCC = "prcc"
    _SRC = "src"
    _SRRC = "srrc"
    _SSRRC = "ssrrc"
    _ALGORITHMS = {
        _PEARSON: compute_pearson_correlation,
        _SPEARMAN: compute_spearman_correlation,
        _PCC: compute_pcc,
        _PRCC: compute_prcc,
        _SRC: compute_src,
        _SRRC: compute_srrc,
        _SSRRC: compute_signed_src,
    }
    DEFAULT_DRIVER = "OT_MONTE_CARLO"

    def __init__(  # noqa: D107
        self,
        disciplines: Collection[MDODiscipline],
        parameter_space: ParameterSpace,
        n_samples: int,
        output_names: Iterable[str] | None = None,
        algo: str | None = None,
        algo_options: Mapping[str, DOELibraryOptionType] | None = None,
        formulation: str = "MDF",
        **formulation_options: Any,
    ) -> None:
        self.__correlation = None
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
        self.main_method = self._SPEARMAN

    @SensitivityAnalysis.main_method.setter
    def main_method(  # noqa: D102
        self,
        name: str,
    ) -> None:
        if name not in self._ALGORITHMS:
            methods = self._ALGORITHMS.keys()
            raise NotImplementedError(
                "{} is a bad method name. "
                "Available ones are {}.".format(name, methods)
            )
        else:
            LOGGER.info("Use {} indices as main indices.")
            self._main_method = name

    def compute_indices(  # noqa: D102
        self, outputs: Sequence[str] | None = None
    ) -> dict[str, IndicesType]:
        output_names = outputs or self.default_output
        if not isinstance(output_names, list):
            output_names = [output_names]
        inputs = Sample(self.dataset.get_data_by_group(self.dataset.INPUT_GROUP))
        outputs = self.dataset.get_data_by_names(output_names)
        self.__correlation = {}
        for algo_name, algo_value in self._ALGORITHMS.items():
            inputs_names = self.dataset.get_names(self.dataset.INPUT_GROUP)
            sizes = self.dataset.sizes
            self.__correlation[algo_name] = {}
            for output_name, value in outputs.items():
                self.__correlation[algo_name][output_name] = []
                for index in range(value.shape[1]):
                    sub_outputs = Sample(value[:, index][:, None])
                    coefficient = array(algo_value(inputs, sub_outputs))
                    coefficient = split_array_to_dict_of_arrays(
                        coefficient, sizes, inputs_names
                    )
                    self.__correlation[algo_name][output_name].append(coefficient)
        return self.indices

    @property
    def pcc(self) -> IndicesType:
        """dict: The Partial Correlation Coefficients.

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
        return self.__correlation[self._PCC]

    @property
    def prcc(self) -> IndicesType:
        """dict: The Partial Rank Correlation Coefficients.

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
        return self.__correlation[self._PRCC]

    @property
    def src(self) -> IndicesType:
        """dict: The Standard Regression Coefficients.

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
        return self.__correlation[self._SRC]

    @property
    def srrc(self) -> IndicesType:
        """dict: The Standard Rank Regression Coefficients.

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
        return self.__correlation[self._SRRC]

    @property
    def ssrrc(self) -> IndicesType:
        """The Signed Standard Rank Regression Coefficients.

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
        return self.__correlation[self._SSRRC]

    @property
    def pearson(self) -> IndicesType:
        """dict: The Pearson coefficients.

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
        return self.__correlation[self._PEARSON]

    @property
    def spearman(self) -> IndicesType:
        """dict: The Spearman coefficients.

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
        return self.__correlation[self._SPEARMAN]

    @property
    def indices(self) -> dict[str, IndicesType]:
        """dict: The sensitivity indices.

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
        return self.__correlation

    @property
    def main_indices(self) -> IndicesType:  # noqa: D102
        return self.__correlation[self.main_method]

    def plot(  # noqa: D102
        self,
        output: str | tuple[str, int],
        inputs: Iterable[str] | None = None,
        title: str | None = None,
        save: bool = True,
        show: bool = False,
        file_path: str | Path | None = None,
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_format: str | None = None,
    ) -> None:
        if not isinstance(output, tuple):
            output = (output, 0)
        dataset = Dataset()
        inputs_names = self.dataset.get_names(self.dataset.INPUT_GROUP)
        inputs_names = self._filter_names(inputs_names, inputs)
        algorithms = sorted(self._ALGORITHMS)
        data = {name: [] for name in inputs_names}
        for method in algorithms:
            indices = getattr(self, method)
            for name in inputs_names:
                data[name].append(indices[output[0]][output[1]][name])
        for name in inputs_names:
            dataset.add_variable(name, vstack(data[name]))
        dataset.row_names = algorithms
        plot = RadarChart(dataset)
        output = f"{output[0]}({output[1]})"
        plot.title = title or f"Correlation indices for the output {output}"
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
        outputs: OutputsType,
        inputs: Iterable[str] | None = None,
        title: str | None = None,
        save: bool = True,
        show: bool = False,
        file_path: str | Path | None = None,
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_format: str | None = None,
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
