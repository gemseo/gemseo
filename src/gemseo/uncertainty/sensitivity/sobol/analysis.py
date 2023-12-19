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
their confidence intervals whose default level is 95%.

The user can select the algorithm to estimate the Sobol' indices.
The computation relies on
`OpenTURNS capabilities <https://openturns.github.io/www/>`_.
"""

from __future__ import annotations

from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from numpy import array
from numpy import newaxis
from openturns import JansenSensitivityAlgorithm
from openturns import MartinezSensitivityAlgorithm
from openturns import MauntzKucherenkoSensitivityAlgorithm
from openturns import SaltelliSensitivityAlgorithm
from openturns import Sample
from strenum import PascalCaseStrEnum
from strenum import StrEnum

from gemseo.algos.doe.lib_openturns import OpenTURNS
from gemseo.uncertainty.sensitivity.analysis import FirstOrderIndicesType
from gemseo.uncertainty.sensitivity.analysis import SecondOrderIndicesType
from gemseo.uncertainty.sensitivity.analysis import SensitivityAnalysis
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from gemseo.algos.doe.doe_library import DOELibraryOptionType
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline import MDODiscipline
    from gemseo.post.dataset.dataset_plot import VariableType


class SobolAnalysis(SensitivityAnalysis):
    """Sensitivity analysis based on the Sobol' indices.

    Examples:
        >>> from numpy import pi
        >>> from gemseo import create_discipline, create_parameter_space
        >>> from gemseo.uncertainty.sensitivity.sobol.analysis import SobolAnalysis
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
        >>> analysis = SobolAnalysis([discipline], parameter_space, n_samples=10000)
        >>> indices = analysis.compute_indices()
    """

    class Algorithm(PascalCaseStrEnum):
        """The algorithms to estimate the Sobol' indices."""

        SALTELLI = "Saltelli"
        JANSEN = "Jansen"
        MAUNTZ_KUCHERENKO = "MauntzKucherenko"
        MARTINEZ = "Martinez"

    __ALGO_NAME_TO_CLASS: Final[dict[Algorithm, type]] = {
        Algorithm.SALTELLI: SaltelliSensitivityAlgorithm,
        Algorithm.JANSEN: JansenSensitivityAlgorithm,
        Algorithm.MAUNTZ_KUCHERENKO: MauntzKucherenkoSensitivityAlgorithm,
        Algorithm.MARTINEZ: MartinezSensitivityAlgorithm,
    }

    class Method(StrEnum):
        """The names of the sensitivity methods."""

        FIRST = "first"
        """The first-order Sobol' index."""

        TOTAL = "total"
        """The total-order Sobol' index."""

    __SECOND: Final[str] = "second"

    _INTERACTION_METHODS: ClassVar[tuple[str]] = (__SECOND,)

    __GET_FIRST_ORDER_INDICES: Final[str] = "getFirstOrderIndices"
    __GET_SECOND_ORDER_INDICES: Final[str] = "getSecondOrderIndices"
    __GET_TOTAL_ORDER_INDICES: Final[str] = "getTotalOrderIndices"

    DEFAULT_DRIVER: ClassVar[str] = OpenTURNS.OT_SOBOL_INDICES

    output_variances: dict[str, NDArray[float]]
    """The variances of the output variables."""

    output_standard_deviations: dict[str, NDArray[float]]
    """The standard deviations of the output variables."""

    def __init__(
        self,
        disciplines: Collection[MDODiscipline],
        parameter_space: ParameterSpace,
        n_samples: int,
        output_names: Iterable[str] = (),
        algo: str = "",
        algo_options: Mapping[str, DOELibraryOptionType] = MappingProxyType({}),
        formulation: str = "MDF",
        compute_second_order: bool = True,
        use_asymptotic_distributions: bool = True,
        **formulation_options: Any,
    ) -> None:
        r"""
        Args:
            compute_second_order: Whether to compute the second-order indices.
            use_asymptotic_distributions: Whether to estimate
                the confidence intervals
                of the first- and total-order Sobol' indices
                with the asymptotic distributions;
                otherwise, use bootstrap.

        Notes:
             The estimators of Sobol' indices rely on the same DOE algorithm.
             This algorithm starts with two independent input datasets
             composed of :math:`N` independent samples
             and this number :math:`N` is the usual sampling size for Sobol' analysis.
             When ``compute_second_order=False``
             or when the input dimension :math:`d` is equal to 2,
             :math:`N=\frac{n_\text{samples}}{2+d}`.
             Otherwise, :math:`N=\frac{n_\text{samples}}{2+2d}`.
             The larger :math:`N`,
             the more accurate the estimators of Sobol' indices are.
             Therefore,
             for a small budget ``n_samples``,
             the user can choose to set ``compute_second_order`` to ``False``
             to ensure a better estimation of the first- and second-order indices.
        """  # noqa: D205, D212, D415
        self.__output_names_to_sobol_algos = {}
        algo_options = algo_options or {}
        algo_options["eval_second_order"] = compute_second_order
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
        self.__eval_second_order = compute_second_order
        self.__use_asymptotic_distributions = use_asymptotic_distributions
        self._main_method = self.Method.FIRST
        dataset = self.dataset
        input_dimension = parameter_space.dimension
        sample_size = len(dataset) // (
            2 + input_dimension * (1 + (compute_second_order and input_dimension > 2))
        )

        # Variance computation.
        _output_variances = (
            dataset.get_view(group_names=dataset.OUTPUT_GROUP)
            .to_numpy()[:sample_size]
            .var(0)
        )
        self.output_variances = {
            column[1]: []
            for column in dataset.get_view(group_names=dataset.OUTPUT_GROUP).columns
        }
        for i, column in enumerate(
            dataset.get_view(group_names=dataset.OUTPUT_GROUP).columns
        ):
            self.output_variances[column[1]].extend([_output_variances[i]])

        self.output_variances = {k: array(v) for k, v in self.output_variances.items()}

        self.output_standard_deviations = {
            k: v**0.5 for k, v in self.output_variances.items()
        }

    def compute_indices(
        self,
        outputs: str | Sequence[str] = (),
        algo: Algorithm = Algorithm.SALTELLI,
        confidence_level: float = 0.95,
    ) -> dict[str, FirstOrderIndicesType]:
        """
        Args:
            algo: The name of the algorithm to estimate the Sobol' indices.
            confidence_level: The level of the confidence intervals.
        """  # noqa:D205,D212,D415
        output_names = outputs or self.default_output
        if isinstance(output_names, str):
            output_names = [output_names]

        inputs = Sample(
            self.dataset.get_view(
                group_names=self.dataset.INPUT_GROUP, variable_names=self._input_names
            ).to_numpy()
        )

        input_dimension = self.dataset.group_names_to_n_components[
            self.dataset.INPUT_GROUP
        ]

        # If eval_second_order is set to False, the input design is of size N(2+n_X).
        # If eval_second_order is set to False,
        #   if n_X = 2, the input design is of size N(2+n_X).
        #   if n_X != 2, the input design is of size N(2+2n_X).
        # Ref: https://openturns.github.io/openturns/latest/user_manual/_generated/
        # openturns.SobolIndicesExperiment.html#openturns.SobolIndicesExperiment
        n_samples = len(self.dataset)
        if self.__eval_second_order and input_dimension > 2:
            sub_sample_size = int(n_samples / (2 * input_dimension + 2))
        else:
            sub_sample_size = int(n_samples / (input_dimension + 2))

        self.__output_names_to_sobol_algos = {}
        for output_name in output_names:
            output_data = self.dataset.get_view(
                group_names=self.dataset.OUTPUT_GROUP, variable_names=output_name
            ).to_numpy()
            algos = self.__output_names_to_sobol_algos[output_name] = []
            for sub_output_data in output_data.T:
                algos.append(
                    self.__ALGO_NAME_TO_CLASS[algo](
                        inputs, Sample(sub_output_data[:, newaxis]), sub_sample_size
                    )
                )
                algos[-1].setUseAsymptoticDistribution(
                    self.__use_asymptotic_distributions
                )
                algos[-1].setConfidenceLevel(confidence_level)

        self._indices = {
            self.Method.FIRST: self.__get_indices(self.__GET_FIRST_ORDER_INDICES),
            self.__SECOND: self.__get_indices(self.__GET_SECOND_ORDER_INDICES),
            self.Method.TOTAL: self.__get_indices(self.__GET_TOTAL_ORDER_INDICES),
        }

        return self._indices

    def __get_indices(
        self, method_name: str
    ) -> FirstOrderIndicesType | SecondOrderIndicesType:
        """Get the first-, second- or total-order indices.

        Args:
            method_name: The name of the OpenTURNS method to compute the indices.

        Returns:
            The first-, second- or total-order indices.
        """
        if (
            method_name == self.__GET_SECOND_ORDER_INDICES
            and not self.__eval_second_order
        ):
            return {}

        names_to_sizes = self.dataset.variable_names_to_n_components
        indices = {
            output_name: [
                split_array_to_dict_of_arrays(
                    array(getattr(ot_algorithm, method_name)()),
                    names_to_sizes,
                    self._input_names,
                )
                for ot_algorithm in self.__output_names_to_sobol_algos[output_name]
            ]
            for output_name in self.__output_names_to_sobol_algos
        }
        if method_name == self.__GET_SECOND_ORDER_INDICES:
            return {
                output_name: [
                    {
                        k: split_array_to_dict_of_arrays(
                            v.T, names_to_sizes, self._input_names
                        )
                        for k, v in output_component_indices.items()
                    }
                    for output_component_indices in output_indices
                ]
                for output_name, output_indices in indices.items()
            }

        return indices

    @property
    def first_order_indices(self) -> FirstOrderIndicesType:
        """The first-order Sobol' indices.

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
        return self._indices[self.Method.FIRST]

    @property
    def second_order_indices(self) -> SecondOrderIndicesType:
        """The second-order Sobol' indices.

        With the following structure:

        .. code-block:: python

            {
                "output_name": [
                    {
                        {"input_name": {"other_input_name": data_array},
                    }
                ]
            }
        """
        return self._indices[self.__SECOND]

    @property
    def total_order_indices(self) -> FirstOrderIndicesType:
        """The total-order Sobol' indices.

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
        return self._indices[self.Method.TOTAL]

    def __unscale_index(
        self,
        sobol_index: NDArray[float] | Mapping[str, NDArray[float]],
        output_name: str,
        output_index: int,
        use_variance: bool,
    ) -> NDArray[float] | dict[str, NDArray[float]]:
        """Unscaled a Sobol' index.

        Args:
            sobol_index: The Sobol' index to unscale.
            output_name: The name of the related output.
            output_index: The index of the related output.
            use_variance: Whether to use the variance of the outputs;
                otherwise, use their standard deviation.

        Returns:
            The unscaled Sobol' index.
        """
        factor = self.output_variances[output_name][output_index]
        if isinstance(sobol_index, Mapping):
            unscaled_data = {k: v * factor for k, v in sobol_index.items()}
            if not use_variance:
                return {k: v**0.5 for k, v in unscaled_data.items()}
        else:
            unscaled_data = sobol_index * factor
            if not use_variance:
                return unscaled_data**0.5

        return unscaled_data

    def unscale_indices(
        self,
        indices: FirstOrderIndicesType | SecondOrderIndicesType,
        use_variance: bool = True,
    ) -> FirstOrderIndicesType | SecondOrderIndicesType:
        """Unscale the Sobol' indices.

        Args:
            indices: The Sobol' indices.
            use_variance: Whether to express an unscaled Sobol' index
                as a share of output variance;
                otherwise,
                express it as the square root of this part
                and therefore with the same unit as the output.

        Returns:
            The unscaled Sobol' indices.
        """
        return {
            output_name: [
                {
                    input_name: self.__unscale_index(
                        sensitivity_indices, output_name, i, use_variance
                    )
                    for input_name, sensitivity_indices in output_value.items()
                }
                for i, output_value in enumerate(output_sensitivity_indices)
            ]
            for output_name, output_sensitivity_indices in indices.items()
        }

    def get_intervals(
        self,
        first_order: bool = True,
    ) -> FirstOrderIndicesType:
        """Get the confidence intervals for the Sobol' indices.

        Warnings:
            You must first call :meth:`.compute_indices`.

        Args:
            first_order: If ``True``, compute the intervals for the first-order indices.
                Otherwise, for the total-order indices.

        Returns:
            The confidence intervals for the Sobol' indices.

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
        names_to_sizes = self.dataset.variable_names_to_n_components
        intervals = {}
        for output_name, sobol_algos in self.__output_names_to_sobol_algos.items():
            intervals[output_name] = []
            for sobol_algorithm in sobol_algos:
                if first_order:
                    interval = sobol_algorithm.getFirstOrderIndicesInterval()
                else:
                    interval = sobol_algorithm.getTotalOrderIndicesInterval()

                names_to_lower_bounds = split_array_to_dict_of_arrays(
                    array(interval.getLowerBound()), names_to_sizes, self._input_names
                )
                names_to_upper_bounds = split_array_to_dict_of_arrays(
                    array(interval.getUpperBound()), names_to_sizes, self._input_names
                )
                intervals[output_name].append({
                    input_name: (
                        names_to_lower_bounds[input_name],
                        names_to_upper_bounds[input_name],
                    )
                    for input_name in self._input_names
                })

        return intervals

    def plot(
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
        sort: bool = True,
        sort_by_total: bool = True,
    ) -> None:
        r"""Plot the first- and total-order Sobol' indices.

        For the :math:`i`-th uncertain input variable,
        plot its first-order Sobol' index :math:`S_i^{1}`
        and its total-order Sobol' index :math:`S_i^{T}` with dots
        and their confidence intervals with vertical lines.

        The subtitle displays the standard deviation (StD) and the variance (Var)
        of the output of interest.

        Args:
            directory_path: The path to the directory where to save the plots.
            file_name: The name of the file.
            title: The title of the plot.
                If empty, use a default one.
            sort: Whether to sort the uncertain variables by decreasing order.
            sort_by_total: Whether to sort according to the total-order Sobol' indices
                when ``sort`` is ``True``.
                Otherwise, use the first-order Sobol' indices.
        """  # noqa: D415 D417
        if not isinstance(output, tuple):
            output = (output, 0)

        fig, ax = plt.subplots()
        if sort_by_total:
            indices = self.total_order_indices
        else:
            indices = self.first_order_indices

        intervals = self.get_intervals()
        output_name, output_component = output
        indices = indices[output_name][output_component]
        intervals = intervals[output_name][output_component]
        first_order_indices = self.first_order_indices[output_name][output_component]
        total_order_indices = self.total_order_indices[output_name][output_component]
        if sort:
            names = [
                name
                for name, _ in sorted(
                    indices.items(), key=lambda item: item[1].sum(), reverse=True
                )
            ]
        else:
            names = indices.keys()

        names = self._filter_names(names, inputs)
        errorbar_options = {"marker": "o", "linestyle": "", "markersize": 7}
        trans1 = Affine2D().translate(-0.01, 0.0) + ax.transData
        trans2 = Affine2D().translate(+0.01, 0.0) + ax.transData
        names_to_sizes = {
            name: value.size for name, value in first_order_indices.items()
        }
        values = [
            first_order_indices[name][index]
            for name in names
            for index in range(names_to_sizes[name])
        ]
        yerr = array([
            [
                first_order_indices[name][index] - intervals[name][0][index],
                intervals[name][1][index] - first_order_indices[name][index],
            ]
            for name in names
            for index in range(names_to_sizes[name])
        ]).T
        x_labels = []
        for name in names:
            if names_to_sizes[name] == 1:
                x_labels.append(name)
            else:
                size = names_to_sizes[name]
                x_labels.extend([
                    repr_variable(name, index, size) for index in range(size)
                ])

        ax.errorbar(
            x_labels,
            values,
            yerr=yerr,
            label="First order",
            transform=trans2,
            **errorbar_options,
        )
        intervals = self.get_intervals(False)
        intervals = intervals[output_name][output_component]
        values = [
            total_order_indices[name][index]
            for name in names
            for index in range(names_to_sizes[name])
        ]
        yerr = array([
            [
                total_order_indices[name][index] - intervals[name][0][index],
                intervals[name][1][index] - total_order_indices[name][index],
            ]
            for name in names
            for index in range(names_to_sizes[name])
        ]).T
        ax.errorbar(
            x_labels,
            values,
            yerr,
            label="Total order",
            transform=trans1,
            **errorbar_options,
        )
        ax.legend(loc="lower left")
        pretty_output_name = repr_variable(
            output_name,
            output_component,
            len(self.total_order_indices[output_name]),
        )
        if not title:
            title = f"Sobol' indices for the output {pretty_output_name}"
        variance = self.output_variances[output_name][output_component]
        ax.set_title(f"{title}\nVar={variance:.1e}    StD={variance**0.5:.1e}")
        ax.set_axisbelow(True)
        ax.grid()
        self._save_show_plot(
            fig,
            save=save,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
        )
