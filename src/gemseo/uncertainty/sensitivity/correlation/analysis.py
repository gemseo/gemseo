# -*- coding: utf-8 -*-
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

from __future__ import division, unicode_literals

import logging
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from numpy import array, vstack
from openturns import (
    CorrelationAnalysis_PCC,
    CorrelationAnalysis_PearsonCorrelation,
    CorrelationAnalysis_PRCC,
    CorrelationAnalysis_SignedSRC,
    CorrelationAnalysis_SpearmanCorrelation,
    CorrelationAnalysis_SRC,
    CorrelationAnalysis_SRRC,
    Sample,
)

from gemseo.algos.doe.doe_lib import DOELibraryOptionType
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.post.dataset.radar_chart import RadarChart
from gemseo.uncertainty.sensitivity.analysis import (
    IndicesType,
    OutputsType,
    SensitivityAnalysis,
)
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.py23_compat import Path

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
        ...     "AnalyticDiscipline", expressions_dict=expressions
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
        >>> analysis = CorrelationAnalysis(discipline, parameter_space, n_samples=1000)
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
        _PEARSON: CorrelationAnalysis_PearsonCorrelation,
        _SPEARMAN: CorrelationAnalysis_SpearmanCorrelation,
        _PCC: CorrelationAnalysis_PCC,
        _PRCC: CorrelationAnalysis_PRCC,
        _SRC: CorrelationAnalysis_SRC,
        _SRRC: CorrelationAnalysis_SRRC,
        _SSRRC: CorrelationAnalysis_SignedSRC,
    }
    DEFAULT_DRIVER = "OT_MONTE_CARLO"

    def __init__(
        self,
        discipline,  # type: MDODiscipline
        parameter_space,  # type: ParameterSpace
        n_samples,  # type: int
        algo=None,  # type:Optional[str]
        algo_options=None,  # type: Optional[Mapping[str,DOELibraryOptionType]]
    ):  # type: (...) -> None  # noqa: D107,D205,D212,D415
        self.__correlation = None
        super(CorrelationAnalysis, self).__init__(
            discipline, parameter_space, n_samples
        )
        self.main_method = self._SPEARMAN

    @SensitivityAnalysis.main_method.setter
    def main_method(
        self,
        name,  # type: str
    ):  # type: (...) -> None
        # noqa: D102
        if name not in self._ALGORITHMS:
            methods = self._ALGORITHMS.keys()
            raise NotImplementedError(
                "{} is a bad method name. "
                "Available ones are {}.".format(name, methods)
            )
        else:
            LOGGER.info("Use {} indices as main indices.")
            self._main_method = name

    def compute_indices(
        self, outputs=None  # type: Optional[Sequence[str]]
    ):  # type: (...) -> Dict[str,IndicesType]
        # noqa: D102
        output_names = outputs or self.default_output
        if not isinstance(output_names, list):
            output_names = [output_names]
        inputs = Sample(self.dataset.get_data_by_group(self.dataset.INPUT_GROUP))
        outputs = self.dataset.get_data_by_names(output_names, True)
        self.__correlation = {}
        array_to_dict = DataConversion.array_to_dict
        for algo_name, algo_value in self._ALGORITHMS.items():
            inputs_names = self.dataset.get_names(self.dataset.INPUT_GROUP)
            sizes = self.dataset.sizes
            self.__correlation[algo_name] = {}
            for output_name, value in outputs.items():
                self.__correlation[algo_name][output_name] = []
                for index in range(value.shape[1]):
                    sub_outputs = Sample(value[:, index][:, None])
                    coefficient = array(algo_value(inputs, sub_outputs))
                    coefficient = array_to_dict(coefficient, inputs_names, sizes)
                    self.__correlation[algo_name][output_name].append(coefficient)
        return self.indices

    @property
    def pcc(self):  # type: (...) -> IndicesType
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
    def prcc(self):  # type: (...) -> IndicesType
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
    def src(self):  # type: (...) -> IndicesType
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
    def srrc(self):  # type: (...) -> IndicesType
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
    def ssrrc(self):  # type: (...) -> IndicesType
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
    def pearson(self):  # type: (...) -> IndicesType
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
    def spearman(self):  # type: (...) -> IndicesType
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
    def indices(self):  # type: (...) -> Dict[str,IndicesType]
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
    def main_indices(self):  # type: (...) -> IndicesType # noqa: D102
        return self.__correlation[self.main_method]

    def plot(
        self,
        output,  # type: Union[str,Tuple[str,int]]
        inputs=None,  # type: Optional[Iterable[str]]
        title=None,  # type: Optional[str]
        save=True,  # type: bool
        show=False,  # type: bool
        file_path=None,  # type: Optional[Union[str,Path]]
        directory_path=None,  # type: Optional[Union[str,Path]]
        file_name=None,  # type: Optional[str]
        file_format=None,  # type: Optional[str]
    ):  # type: (...) -> None # noqa: D417,D102
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
        output = "{}({})".format(output[0], output[1])
        plot.title = title or "Correlation indices for the output {}".format(output)
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

    def plot_radar(
        self,
        outputs,  # type: OutputsType
        inputs=None,  # type: Optional[Iterable[str]]
        title=None,  # type: Optional[str]
        save=True,  # type: bool
        show=False,  # type: bool
        file_path=None,  # type: Optional[Union[str,Path]]
        directory_path=None,  # type: Optional[Union[str,Path]]
        file_name=None,  # type: Optional[str]
        file_format=None,  # type: Optional[str]
        min_radius=-1.0,  # type: float
        max_radius=1.0,  # type: float
        **options  # type:bool
    ):  # type: (...) -> RadarChart #noqa: D102
        return super(CorrelationAnalysis, self).plot_radar(
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
            **options
        )
