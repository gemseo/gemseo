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

# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

r"""Class for the estimation of Sobol' indices.

Let us consider the model :math:`Y=f(X_1,\ldots,X_d)`
where:

- :math:`X_1,\ldots,X_d` are independent random variables,
- :math:`E\left[f(X_1,\ldots,X_d)^2\right]<\infty`.

Then, the following decomposition is unique:

.. math::

   Y=f_0 + \sum_{i=1}^df_i(X_i) + \sum_{i,j=1\atop i\neq j}^d f_{i,j}(X_i,X_j)
   + \sum_{i,j,k=1\atop i\neq j\neq k}^d f_{i,j,k}(X_i,X_j,X_k) + \ldots +
   f_{1,\ldots,d}(X_1,\ldots,X_d)

where:

- :math:`f_0=E[Y]`,
- :math:`f_i(X_i)=E[Y|X_i]-f_0`,
- :math:`f_{i,j}(X_i,X_j)=E[Y|X_i,X_j]-f_i(X_i)-f_j(X_j)-f_0`
- and so on.

Then, the shift to variance leads to:

.. math::

   V[Y]=\sum_{i=1}^dV\left[f_i(X_i)\right] +
   \sum_{i,j=1\atop j\neq i}^d V\left[f_{i,j}(X_i,X_j)\right] + \ldots +
   V\left[f_{1,\ldots,d}(X_1,\ldots,X_d)\right]

and the Sobol' indices are obtained by dividing by the variance and sum up to 1:

.. math::

   1=\sum_{i=1}^dS_i + \sum_{i,j=1\atop j\neq i}^d S_{i,j} +
   \sum_{i,j,k=1\atop i\neq j\neq k}^d S_{i,j,k} + \ldots + S_{1,\ldots,d}

A Sobol' index represents the share of output variance explained
by a parameter or a group of parameters. For the parameter :math:`X_i`,

- :math:`S_i` is the first-order Sobol' index
  measuring the individual effect of :math:`X_i`,
- :math:`S_{i,j}` is the second-order Sobol' index
  measuring the joint effect between :math:`X_i` and :math:`X_j`,
- :math:`S_{i,j,k}` is the third-order Sobol' index
  measuring the joint effect between :math:`X_i`, :math:`X_j` and :math:`X_k`,
- and so on.

In practice, we only consider the first-order Sobol' index:

.. math::

   S_i=\frac{V[E[Y|X_i]]}{V[Y]}

and the total-order Sobol' index:

.. math::

   S_i^T=\sum_{u\subset\{1,\ldots,d\}\atop u \ni i}S_u

The latter represents the sum of the individual effect of :math:`X_i` and
the joint effects between :math:`X_i` and any parameter or group of parameters.

This methodology relies on the :class:`.SobolAnalysis` class. Precisely,
:attr:`.SobolAnalysis.indices` contains
both :attr:`.SobolAnalysis.first_order_indices` and
:attr:`.SobolAnalysis.total_order_indices`
while :attr:`.SobolAnalysis.main_indices` represents total-order Sobol'
indices.
Lastly, the :meth:`.SobolAnalysis.plot` method represents
the estimations of both first-order and total-order Sobol' indices along with
their 95% confidence interval.

The user can select the algorithm to estimate the Sobol' indices.
The computation relies on
`OpenTURNS capabilities <http://www.openturns.org/>`_.
"""

from __future__ import division, unicode_literals

import logging
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from numpy import array
from openturns import (
    JansenSensitivityAlgorithm,
    MartinezSensitivityAlgorithm,
    MauntzKucherenkoSensitivityAlgorithm,
    SaltelliSensitivityAlgorithm,
    Sample,
)
from past.utils import old_div

from gemseo.algos.doe.doe_lib import DOELibraryOptionType
from gemseo.algos.doe.lib_openturns import OpenTURNS
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.uncertainty.sensitivity.analysis import IndicesType, SensitivityAnalysis
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.py23_compat import Path

LOGGER = logging.getLogger(__name__)


class SobolAnalysis(SensitivityAnalysis):
    """Sensitivity analysis based on the Sobol' indices.

    Examples:
        >>> from numpy import pi
        >>> from gemseo.api import create_discipline, create_parameter_space
        >>> from gemseo.uncertainty.sensitivity.sobol.analysis import SobolAnalysis
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
        >>> analysis = SobolAnalysis(discipline, parameter_space, n_samples=10000)
        >>> indices = analysis.compute_indices()
    """

    _ALGOS = {
        "Saltelli": SaltelliSensitivityAlgorithm,
        "Jansen": JansenSensitivityAlgorithm,
        "MauntzKucherenko": MauntzKucherenkoSensitivityAlgorithm,
        "Martinez": MartinezSensitivityAlgorithm,
    }
    _FIRST = "first"
    _TOTAL = "total"
    _FIRST_METHOD = "Sobol({})".format(_FIRST)
    _TOTAL_METHOD = "Sobol({})".format(_TOTAL)

    AVAILABLE_ALGOS = sorted(_ALGOS.keys())
    DEFAULT_DRIVER = OpenTURNS.OT_SOBOL_INDICES

    def __init__(
        self,
        discipline,  # type: MDODiscipline
        parameter_space,  # type: ParameterSpace
        n_samples,  # type: int
        algo=None,  # type:Optional[str]
        algo_options=None,  # type: Optional[Mapping[str,DOELibraryOptionType]]
    ):  # type: (...) -> None  # noqa: D107,D205,D212,D415
        self.__sobol = None
        super(SobolAnalysis, self).__init__(discipline, parameter_space, n_samples)
        self.main_method = self._FIRST

    @SensitivityAnalysis.main_method.setter
    def main_method(
        self,
        name,  # type: str
    ):  # type:(...) -> None # noqa: D102
        if name == self._FIRST:
            self._main_method = self._FIRST_METHOD
            LOGGER.info("Use first order indices as main indices.")
        elif name == self._TOTAL:
            self._main_method = self._TOTAL_METHOD
            LOGGER.info("Use total order indices as main indices.")
        else:
            raise NotImplementedError(
                "{} is a bad method name. "
                "Available ones are {}.".format(name, [self._FIRST, self._TOTAL])
            )

    def compute_indices(
        self,
        outputs=None,  # type: Optional[Sequence[str]]
        algo="Saltelli",  # type: str
    ):  # type: (...) -> Dict[str,IndicesType]
        # noqa:D205,D212,D415
        """
        Args:
            algo: The name of the algorithm to estimate the Sobol' indices
        """
        try:
            algo = self._ALGOS[algo]
        except Exception:
            raise TypeError(
                "{} is not an available algorithm "
                "to compute Sobol indices.".format(algo)
            )
        output_names = outputs or self.default_output
        if not isinstance(output_names, list):
            output_names = [output_names]
        inputs = Sample(self.dataset.get_data_by_group(self.dataset.INPUT_GROUP))
        outputs = self.dataset.get_data_by_names(output_names, True)
        dim = self.dataset.dimension[self.dataset.INPUT_GROUP]
        n_samples = int(old_div(len(self.dataset), (dim + 2)))
        self.__sobol = {}
        for output_name, value in outputs.items():
            self.__sobol[output_name] = []
            for index in range(value.shape[1]):
                sub_outputs = Sample(value[:, index][:, None])
                self.__sobol[output_name].append(algo(inputs, sub_outputs, n_samples))
        return self.indices

    def __get_indices(
        self,
        first_order=True,  # type: bool
    ):  # type: (...) -> IndicesType
        """Get the indices, either first-order or total order.

        Args:
            first_order: The type of indices.
                If True, return first-order Sobol' indices.
                Otherwise, return total-order Sobol' indices.

        Returns:
            The Sobol' indices, either first-order or total order.
        """
        if first_order:
            method = "getFirstOrderIndices"
        else:
            method = "getTotalOrderIndices"
        array_to_dict = DataConversion.array_to_dict
        inputs_names = self.dataset.get_names(self.dataset.INPUT_GROUP)
        sizes = self.dataset.sizes
        indices = {}
        for name in self.__sobol:
            indices[name] = [
                array_to_dict(array(getattr(sobol, method)()), inputs_names, sizes)
                for sobol in self.__sobol[name]
            ]
        return indices

    @property
    def first_order_indices(self):  # type: (...) -> IndicesType
        """dict: The first-order Sobol' indices.

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
        return self.__get_indices()

    @property
    def total_order_indices(self):  # type: (...) -> IndicesType
        """dict: The total-order Sobol' indices.

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
        return self.__get_indices(False)

    def get_intervals(
        self,
        first_order=True,  # type: bool
    ):  # type: (...) -> IndicesType
        """Get the confidence interval for Sobol' indices.

        Args:
            first_order: The type of indices.
                If True, returns the intervals for the first-order indices.
                Otherwise, for the total-order indices.

        Returns:
            dict: The confidence intervals for the Sobol' indices.

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
        array_to_dict = DataConversion.array_to_dict
        inputs_names = self.dataset.get_names(self.dataset.INPUT_GROUP)
        sizes = self.dataset.sizes
        intervals = {}
        for output_name in self.__sobol:
            intervals[output_name] = []
            for sobol in self.__sobol[output_name]:
                if first_order:
                    interval = sobol.getFirstOrderIndicesInterval()
                else:
                    interval = sobol.getTotalOrderIndicesInterval()
                lower_bound = array(interval.getLowerBound())
                upper_bound = array(interval.getUpperBound())
                lower_bound = array_to_dict(lower_bound, inputs_names, sizes)
                upper_bound = array_to_dict(upper_bound, inputs_names, sizes)
                intervals[output_name].append(
                    {
                        name: array([lower_bound[name][0], upper_bound[name][0]])
                        for name in lower_bound
                    }
                )
        return intervals

    @property
    def indices(
        self,
    ):  # type: (...) -> Dict[str, IndicesType] # noqa: D102
        return {"first": self.first_order_indices, "total": self.total_order_indices}

    @property
    def main_indices(self):  # type: (...) -> IndicesType # noqa: D102
        if self.main_method == self._TOTAL_METHOD:
            return self.total_order_indices
        else:
            return self.first_order_indices

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
        sort=True,  # type:bool
        sort_by_total=True,  # type:bool
    ):  # noqa: D417
        r"""Plot the first- and total-order Sobol' indices.

        For :math:`i\in\{1,\ldots,d\}`, plot :math:`S_i^{1}` and :math:`S_T^{1}`
        with their confidence intervals,

        Args:
            sort: The sorting option.
                If True, sort variables before display.
            sort_by_total: The type of sorting.
                If True, sort variables according to total-order Sobol' indices.
                Otherwise, use first-order Sobol' indices.
        """
        if not isinstance(output, tuple):
            output = (output, 0)
        fig, ax = plt.subplots()
        if sort_by_total:
            indices = self.total_order_indices
        else:
            indices = self.first_order_indices
        intervals = self.get_intervals()
        indices = indices[output[0]][output[1]]
        intervals = intervals[output[0]][output[1]]
        first_order_indices = self.first_order_indices[output[0]][output[1]]
        total_order_indices = self.total_order_indices[output[0]][output[1]]
        if sort:
            names = [
                name
                for name, _ in sorted(
                    list(indices.items()), key=lambda item: item[1], reverse=True
                )
            ]
        else:
            names = list(indices.keys())
        names = self._filter_names(names, inputs)
        errorbar_options = {"marker": "o", "linestyle": "", "markersize": 7}
        trans1 = Affine2D().translate(-0.01, 0.0) + ax.transData
        trans2 = Affine2D().translate(+0.01, 0.0) + ax.transData
        values = [first_order_indices[name] for name in names]
        yerr = array(
            [
                [
                    first_order_indices[name][0] - intervals[name][0],
                    intervals[name][1] - first_order_indices[name][0],
                ]
                for name in names
            ]
        ).T
        ax.errorbar(
            names,
            values,
            yerr,
            label="First order",
            transform=trans2,
            **errorbar_options
        )
        intervals = self.get_intervals(False)
        intervals = intervals[output[0]][output[1]]
        values = [total_order_indices[name] for name in names]
        yerr = array(
            [
                [
                    total_order_indices[name][0] - intervals[name][0],
                    intervals[name][1] - total_order_indices[name][0],
                ]
                for name in names
            ]
        ).T
        ax.errorbar(
            names,
            values,
            yerr,
            label="Total order",
            transform=trans1,
            **errorbar_options
        )
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=2,
            borderaxespad=0,
            frameon=False,
        )
        output = "{}({})".format(output[0], output[1])
        ax.set_title(title or "Sobol indices for the output {}".format(output))
        self._save_show_plot(
            fig,
            save=save,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
        )
