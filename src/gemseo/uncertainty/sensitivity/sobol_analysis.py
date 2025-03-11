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
by an input variable or a group of input variables. For the input variable :math:`X_i`,

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
the joint effects between :math:`X_i` and any input variable or group of input variable.

This methodology relies on the :class:`.SobolAnalysis` class. Precisely,
:attr:`.SobolAnalysis.indices` contains
both :attr:`.SobolAnalysis.indices.first` and
:attr:`.SobolAnalysis.indices.total`
while :attr:`.SobolAnalysis.main_indices` represents first-order Sobol'
indices.
Lastly, the :meth:`.SobolAnalysis.plot` method represents
the estimations of both first-order and total-order Sobol' indices along with
their confidence intervals whose default level is 95%.

The user can select the algorithm to estimate the Sobol' indices.
The computation relies on
`OpenTURNS capabilities <https://openturns.github.io/www/>`_.

Control variates can be given to compute indices. In this case, the algorithm selection
is disregarded and the estimation is based on the Monte Carlo estimator proposed by
Saltelli in :cite:`saltelli2010`.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from numpy import array
from numpy import hstack
from numpy import newaxis
from numpy import sign
from numpy import vstack
from numpy.random import default_rng
from openturns import JansenSensitivityAlgorithm
from openturns import MartinezSensitivityAlgorithm
from openturns import MauntzKucherenkoSensitivityAlgorithm
from openturns import ResourceMap
from openturns import SaltelliSensitivityAlgorithm
from openturns import Sample
from pandas import Series
from strenum import PascalCaseStrEnum
from strenum import StrEnum

from gemseo.uncertainty.sensitivity._cv_sobol_algorithm import CVSobolAlgorithm
from gemseo.uncertainty.sensitivity.base_sensitivity_analysis import (
    BaseSensitivityAnalysis,
)
from gemseo.uncertainty.sensitivity.base_sensitivity_analysis import (
    FirstOrderIndicesType,
)
from gemseo.uncertainty.sensitivity.base_sensitivity_analysis import (
    SecondOrderIndicesType,
)
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.matplotlib_figure import save_show_figure_from_file_path_manager
from gemseo.utils.seeder import SEED
from gemseo.utils.string_tools import filter_names
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from gemseo.algos.base_driver_library import DriverSettingType
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline import Discipline
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.scenarios.backup_settings import BackupSettings
    from gemseo.typing import RealArray
    from gemseo.utils.string_tools import VariableType

LOGGER = logging.getLogger(__name__)


class SobolAnalysis(BaseSensitivityAnalysis):
    """Sensitivity analysis based on the Sobol' indices.

    Examples:
        >>> from numpy import pi
        >>> from gemseo import create_discipline, create_parameter_space
        >>> from gemseo.uncertainty.sensitivity.sobol_analysis import SobolAnalysis
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
        >>> analysis = SobolAnalysis()
        >>> analysis.compute_samples([discipline], parameter_space, n_samples=10000)
        >>> indices = analysis.compute_indices()

    .. note:: The second-order Sobol' indices cannot be estimated with control variates.
    """

    @dataclass(frozen=True)
    class SensitivityIndices:  # noqa: D106
        first: FirstOrderIndicesType = field(default_factory=dict)
        """The first-order Sobol' indices."""

        second: SecondOrderIndicesType = field(default_factory=dict)
        """The second-order Sobol' indices."""

        total: FirstOrderIndicesType = field(default_factory=dict)
        """The total order Sobol' indices."""

    _indices: SensitivityIndices

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
    """The mapping from the sensitivity algorithms to the OT classes."""

    class Method(StrEnum):
        """The names of the sensitivity methods."""

        FIRST = "first"
        """The first-order Sobol' index."""

        TOTAL = "total"
        """The total-order Sobol' index."""

    @dataclass
    class ControlVariate:
        """A control variate based on a cheap discipline.

        If either ``indices`` or ``variance`` is missing,
        both are estimated from ``n_samples`` evaluations of ``discipline``.
        """

        discipline: Discipline
        """A cheap discipline, e.g. a surrogate discipline.

        It must have as inputs the input variables and the output variables
        used by ``SobolAnalysis``.
        """

        indices: Mapping[SobolAnalysis.Method, FirstOrderIndicesType] = field(
            default_factory=dict
        )
        """The mapping between method names and first-order Sobol' indices.

        If empty, ``SobolAnalysis`` will compute it.
        """

        n_samples: int = 0
        """The number of samples to estimate the variance and the indices.

        If 0, use 100 times more samples than the number passed at instantiation.
        """

        variance: Mapping[str, RealArray] = field(default_factory=dict)
        """The mapping between output names and output variances.

        If empty, ``SobolAnalysis`` will compute it.
        """

    __SECOND: Final[str] = "second"

    _INTERACTION_METHODS: ClassVar[tuple[str]] = (__SECOND,)

    __GET_FIRST_ORDER_INDICES: Final[str] = "getFirstOrderIndices"
    __GET_SECOND_ORDER_INDICES: Final[str] = "getSecondOrderIndices"
    __GET_TOTAL_ORDER_INDICES: Final[str] = "getTotalOrderIndices"

    __SETTINGS_TO_INTERVAL_GETTERS: Final[dict[tuple[bool, bool], str]] = {
        (True, True): "first_indices_interval",
        (True, False): "total_indices_interval",
        (False, True): "getFirstOrderIndicesInterval",
        (False, False): "getTotalOrderIndicesInterval",
    }
    """A mapping from settings to the name of the interval getter.

    The settings are the following boolean parameters:

    - Whether control variates are used or not.
    - Whether the order of the computed Sobol' index is
      the first order or the total order.
    """

    DEFAULT_DRIVER: ClassVar[str] = "OT_SOBOL_INDICES"

    _DEFAULT_MAIN_METHOD: ClassVar[Method] = Method.FIRST

    def compute_samples(
        self,
        disciplines: Collection[Discipline],
        parameter_space: ParameterSpace,
        n_samples: int,
        output_names: Iterable[str] = (),
        algo: str = "",
        algo_settings: Mapping[str, DriverSettingType] = READ_ONLY_EMPTY_DICT,
        backup_settings: BackupSettings | None = None,
        formulation_name: str = "MDF",
        compute_second_order: bool = True,
        **formulation_settings: Any,
    ) -> IODataset:
        r"""
        Args:
            compute_second_order: Whether to compute the second-order indices.

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
        algo_settings = algo_settings or {}
        algo_settings["eval_second_order"] = compute_second_order
        super().compute_samples(
            disciplines,
            parameter_space,
            n_samples=n_samples,
            output_names=output_names,
            algo=algo,
            algo_settings=algo_settings,
            backup_settings=backup_settings,
            formulation_name=formulation_name,
            **formulation_settings,
        )
        dataset = self.dataset
        dataset.misc["output_names_to_sobol_algos"] = {}
        n_inputs = parameter_space.dimension

        # If eval_second_order is set to False, the input design is of size N(2+n_X).
        # If eval_second_order is set to True,
        #   if n_X = 2, the input design is of size N(2+n_X).
        #   if n_X != 2, the input design is of size N(2+2n_X).
        # Ref: https://openturns.github.io/openturns/latest/user_manual/_generated/
        # openturns.SobolIndicesExperiment.html#openturns.SobolIndicesExperiment
        sample_size = len(dataset) // (
            2 + n_inputs * (1 + (compute_second_order and n_inputs > 2))
        )
        self.dataset.misc["eval_second_order"] = compute_second_order
        self.dataset.misc["parameter_space"] = parameter_space
        self.dataset.misc["sample_size"] = sample_size
        self.dataset.misc["n_inputs"] = n_inputs
        output_variances = split_array_to_dict_of_arrays(
            dataset.get_view(group_names=dataset.OUTPUT_GROUP)
            .to_numpy()[: 2 * sample_size]
            .var(0),
            dataset.variable_names_to_n_components,
            dataset.output_names,
        )
        self.dataset.misc["output_variances"] = output_variances
        self.dataset.misc["output_standard_deviations"] = {
            k: v**0.5 for k, v in output_variances.items()
        }
        self.dataset.misc["use_control_variates"] = False
        self.dataset.misc["output_names_to_sobol_algos"] = {}
        return self.dataset

    @property
    def output_variances(self) -> dict[str, RealArray]:
        """The variances of the output variables."""
        return self.dataset.misc["output_variances"]

    @property
    def output_standard_deviations(self) -> dict[str, RealArray]:
        """The standard deviations of the output variables."""
        return self.dataset.misc["output_standard_deviations"]

    def __execute_cv(
        self,
        sample: Series,
        cv_d: Discipline,
    ) -> Series:
        """Execute a control variate on a sample.

        Args:
            sample: The sample on which the control variate is applied.
            cv_d: The discipline of the control variate .

        Returns:
            The outputs in a pandas series.
        """
        input_sample = sample[self.dataset.INPUT_GROUP]
        io_data = cv_d.execute({
            input_name: input_sample[input_name] for input_name in self._input_names
        })
        return Series(
            [io_data[output_name] for output_name in self._output_names],
            index=self._output_names,
        )

    def __compute_cv_stats(self, cv: ControlVariate) -> ControlVariate:
        """Compute the output variances or output indices of the control variate.

        They are computed only if they are not provided.

        Args:
            cv: A control variate.

        Returns:
            The control variate
            with the output variances and output indices computed if needed.
        """
        if cv.variance and cv.indices:
            return cv

        dataset = self.dataset
        n_samples = (
            100 * dataset.misc["sample_size"] * (2 + dataset.misc["n_inputs"])
            if cv.n_samples == 0
            else cv.n_samples
        )
        cv_analysis = self.__class__()
        cv_analysis.compute_samples(
            [cv.discipline],
            parameter_space=dataset.misc["parameter_space"],
            n_samples=n_samples,
            output_names=self._output_names,
            compute_second_order=False,
        )
        cv.variance = cv_analysis.output_variances
        cv.indices = cv_analysis.compute_indices()
        return cv

    def __compute_indices_classically(
        self,
        output_names: Iterable[str],
        algo: Algorithm = Algorithm.SALTELLI,
        confidence_level: float = 0.95,
        use_asymptotic_distributions: bool = True,
        n_replicates: int = 100,
    ) -> SensitivityIndices:
        """Compute the sensitivity indices with OpenTURNS capabilities.

        Args:
            output_names: The disciplines' outputs to be considered for the analysis.
            algo: The name of the algorithm to estimate the Sobol' indices.
            confidence_level: The level of the confidence intervals.
            use_asymptotic_distributions: Whether to estimate
                the confidence intervals
                of the first- and total-order Sobol' indices
                with the asymptotic distributions;
                otherwise, use bootstrap.
                If control variates are used, the confidence intervals can only be
                estimated via bootstrap.
            n_replicates: The number of bootstrap samples used for the computation of
                the confidence intervals.

        Returns:
            The sensitivity indices.
        """
        algo_class = self.__ALGO_NAME_TO_CLASS[algo]
        ResourceMap.SetAsUnsignedInteger(
            "SobolIndicesAlgorithm-DefaultBootstrapSize", n_replicates
        )
        dataset = self.dataset
        input_data = Sample(
            dataset.get_view(
                group_names=dataset.INPUT_GROUP, variable_names=self._input_names
            ).to_numpy()
        )
        output_names_to_sobol_algos = self.dataset.misc["output_names_to_sobol_algos"]
        for output_name in output_names:
            output_data = dataset.get_view(
                group_names=dataset.OUTPUT_GROUP, variable_names=output_name
            ).to_numpy()
            algos = output_names_to_sobol_algos[output_name] = []
            for sub_output_data in output_data.T:
                ot_algo = algo_class(
                    input_data,
                    Sample(sub_output_data[:, newaxis]),
                    dataset.misc["sample_size"],
                )
                ot_algo.setUseAsymptoticDistribution(use_asymptotic_distributions)
                ot_algo.setConfidenceLevel(confidence_level)
                algos.append(ot_algo)

        self._indices = self.SensitivityIndices(
            first=self.__get_indices(self.__GET_FIRST_ORDER_INDICES),
            second=self.__get_indices(self.__GET_SECOND_ORDER_INDICES),
            total=self.__get_indices(self.__GET_TOTAL_ORDER_INDICES),
        )
        return self._indices

    def __compute_indices_using_cv(
        self,
        output_names: Iterable[str],
        control_variates: Iterable[ControlVariate],
        confidence_level: float = 0.95,
        n_replicates: int = 100,
        seed: int | None = SEED,
    ) -> SensitivityIndices:
        """Compute the sensitivity indices using control variates.

        Args:
            output_names: The disciplines' outputs to be considered for the analysis.
            control_variates: The control variates.
            confidence_level: The level of the confidence intervals.
            n_replicates: The number of bootstrap samples used for the computation of
                the confidence intervals.
            seed: The seed to initialize the random generator used for the bootstrapping
                method.
                If ``None``,
                then fresh, unpredictable entropy will be pulled from the OS.

        Returns:
            The sensitivity indices.
        """
        dataset = self.dataset
        n = dataset.misc["sample_size"]

        generator = default_rng(seed)
        bootstrap_samples = []
        for _ in range(n_replicates):
            bootstrap_sample_a = generator.choice(n, n)
            bootstrap_sample_ab = hstack([bootstrap_sample_a, bootstrap_sample_a + n])
            bootstrap_samples.append((bootstrap_sample_a, bootstrap_sample_ab))

        n_inputs = dataset.misc["n_inputs"]
        n_samples_wo_second_order = n * (2 + n_inputs)

        control_variates = [self.__compute_cv_stats(cv) for cv in control_variates]

        cvs_dataset_list = [
            dataset.get_view(indices=range(n_samples_wo_second_order)).apply(
                lambda sample, cv_d=cv.discipline: self.__execute_cv(sample, cv_d),
                axis=1,
            )
            for cv in control_variates
        ]

        output_names_to_sobol_algos = dataset.misc["output_names_to_sobol_algos"]
        for output_name in output_names:
            output_data = dataset.get_view(
                group_names=dataset.OUTPUT_GROUP,
                variable_names=output_name,
                indices=range(n_samples_wo_second_order),
            ).to_numpy()
            cvs_output_data = [
                vstack(list(cv_dataset_list[output_name]))
                for cv_dataset_list in cvs_dataset_list
            ]
            algos = output_names_to_sobol_algos[output_name] = []
            for i, sub_output_data in enumerate(output_data.T):
                sub_cvs_output_data = [
                    cv_output_data.T[i] for cv_output_data in cvs_output_data
                ]
                sub_cvs_statistics = [
                    (
                        cv.variance[output_name][i],
                        {
                            method: getattr(cv.indices, str(method).lower())[
                                output_name
                            ][i]
                            for method in list(self.Method)
                        },
                    )
                    for cv in control_variates
                ]
                algo = CVSobolAlgorithm(
                    n_inputs,
                    sub_output_data,
                    array(sub_cvs_output_data),
                    sub_cvs_statistics,
                    bootstrap_samples,
                    confidence_level,
                )
                algos.append(algo)
                self.output_variances[output_name][i] = algo.variance

        self._indices = self.SensitivityIndices(
            first=self.__get_indices("compute_first_indices"),
            total=self.__get_indices("compute_total_indices"),
        )
        return self._indices

    def compute_indices(
        self,
        output_names: str | Iterable[str] = (),
        algo: Algorithm = Algorithm.SALTELLI,
        confidence_level: float = 0.95,
        control_variates: ControlVariate | Iterable[ControlVariate] = (),
        use_asymptotic_distributions: bool = True,
        n_replicates: int = 100,
        seed: int | None = SEED,
    ) -> SensitivityIndices:
        """
        Args:
            algo: The name of the algorithm to estimate the Sobol' indices.
            confidence_level: The level of the confidence intervals.
            control_variates: The control variates based on cheap disciplines.
            use_asymptotic_distributions: Whether to estimate
                the confidence intervals
                of the first- and total-order Sobol' indices
                with the asymptotic distributions;
                otherwise, use bootstrap.
                If control variates are used, the confidence intervals can only be
                estimated via bootstrap.
            n_replicates: The number of bootstrap samples used for the computation of
                the confidence intervals.
            seed: The seed to initialize the random generator used for the bootstrapping
                method when the indices are estimated using control variates.
                If ``None``,
                then fresh, unpredictable entropy will be pulled from the OS.
        """  # noqa:D205,D212,D415
        output_names = self._get_output_names(output_names)
        self.dataset.misc["output_names_to_sobol_algos"] = {}
        if control_variates:
            if isinstance(control_variates, self.ControlVariate):
                control_variates = [control_variates]
            self.dataset.misc["use_control_variates"] = True
            return self.__compute_indices_using_cv(
                output_names,
                control_variates,
                confidence_level,
                n_replicates,
                seed,
            )

        return self.__compute_indices_classically(
            output_names,
            algo,
            confidence_level,
            use_asymptotic_distributions,
            n_replicates,
        )

    def __get_indices(
        self, method_name: str
    ) -> FirstOrderIndicesType | SecondOrderIndicesType:
        """Get the first-, second- or total-order indices.

        Args:
            method_name: The name of the sensitivity method to compute the indices.

        Returns:
            The first-, second- or total-order indices.
        """
        dataset = self.dataset
        if (
            method_name == self.__GET_SECOND_ORDER_INDICES
            and not dataset.misc["eval_second_order"]
        ):
            return {}

        names_to_sizes = dataset.variable_names_to_n_components
        output_names_to_sobol_algos = dataset.misc["output_names_to_sobol_algos"]
        indices = {
            output_name: [
                split_array_to_dict_of_arrays(
                    array(getattr(algorithm, method_name)()),
                    names_to_sizes,
                    self._input_names,
                )
                for algorithm in algorithms
            ]
            for output_name, algorithms in output_names_to_sobol_algos.items()
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

    def __unscale_index(
        self,
        sobol_index: RealArray | Mapping[str, RealArray],
        output_name: str,
        output_index: int,
        use_variance: bool,
    ) -> RealArray | dict[str, RealArray]:
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
                return {
                    k: sign(v) * (sign(v) * v) ** 0.5 for k, v in unscaled_data.items()
                }
        else:
            unscaled_data = sobol_index * factor
            if not use_variance:
                return (
                    sign(unscaled_data) * (sign(unscaled_data) * unscaled_data) ** 0.5
                )

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
        use_cv = self.dataset.misc["use_control_variates"]
        names_to_sizes = self.dataset.variable_names_to_n_components
        intervals = {}
        for output_name, sobol_algos in self.dataset.misc[
            "output_names_to_sobol_algos"
        ].items():
            intervals[output_name] = []
            for sobol_algorithm in sobol_algos:
                interval = getattr(
                    sobol_algorithm,
                    self.__SETTINGS_TO_INTERVAL_GETTERS[use_cv, first_order],
                )
                if not use_cv:
                    interval = array([
                        interval().getLowerBound(),
                        interval().getUpperBound(),
                    ])
                names_to_lower_bounds = split_array_to_dict_of_arrays(
                    interval[0], names_to_sizes, self._input_names
                )
                names_to_upper_bounds = split_array_to_dict_of_arrays(
                    interval[1], names_to_sizes, self._input_names
                )
                intervals[output_name].append({
                    input_name: array([
                        names_to_lower_bounds[input_name],
                        names_to_upper_bounds[input_name],
                    ])
                    for input_name in self._input_names
                })

        return intervals

    def plot(
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
        sort: bool = True,
        sort_by_total: bool = True,
    ) -> Figure:
        r"""Plot the first- and total-order Sobol' indices.

        For the :math:`i`-th input variable,
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
            sort: Whether to sort the input variables by decreasing order.
            sort_by_total: Whether to sort according to the total-order Sobol' indices
                when ``sort`` is ``True``.
                Otherwise, use the first-order Sobol' indices.
        """  # noqa: D415 D417
        if not isinstance(output, tuple):
            output = (output, 0)

        fig, ax = plt.subplots()

        indices = self.indices.total if sort_by_total else self.indices.first
        output_name, output_component = output
        indices = indices[output_name][output_component]
        if sort:
            names = [
                name
                for name, _ in sorted(
                    indices.items(), key=lambda item: item[1].sum(), reverse=True
                )
            ]
        else:
            names = indices.keys()

        names = filter_names(names, input_names)

        first_order_indices = self.indices.first[output_name][output_component]
        total_order_indices = self.indices.total[output_name][output_component]
        names_to_sizes = {
            name: value.size for name, value in first_order_indices.items()
        }
        values_first_order = [
            first_order_indices[name][index]
            for name in names
            for index in range(names_to_sizes[name])
        ]
        values_total_order = [
            total_order_indices[name][index]
            for name in names
            for index in range(names_to_sizes[name])
        ]
        x_labels = []
        for name in names:
            if names_to_sizes[name] == 1:
                x_labels.append(name)
            else:
                size = names_to_sizes[name]
                x_labels.extend([
                    repr_variable(name, index, size) for index in range(size)
                ])
        pretty_output_name = repr_variable(
            output_name,
            output_component,
            len(self.indices.total[output_name]),
        )
        if not title:
            title = f"Sobol' indices for the output {pretty_output_name}"
        variance = self.output_variances[output_name][output_component]
        ax.set_title(f"{title}\nVar={variance:.1e}    StD={variance**0.5:.1e}")
        ax.set_axisbelow(True)
        ax.grid()

        intervals = self.get_intervals()
        intervals = intervals[output_name][output_component]
        errorbar_options = {"marker": "o", "linestyle": "", "markersize": 7}
        trans1 = Affine2D().translate(-0.01, 0.0) + ax.transData
        trans2 = Affine2D().translate(+0.01, 0.0) + ax.transData
        yerr = array([
            [
                first_order_indices[name][index] - intervals[name][0, index],
                intervals[name][1, index] - first_order_indices[name][index],
            ]
            for name in names
            for index in range(names_to_sizes[name])
        ]).T

        ax.errorbar(
            x_labels,
            values_first_order,
            yerr=yerr,
            label="First order",
            transform=trans2,
            **errorbar_options,
        )
        intervals = self.get_intervals(False)
        intervals = intervals[output_name][output_component]
        yerr = array([
            [
                total_order_indices[name][index] - intervals[name][0, index],
                intervals[name][1, index] - total_order_indices[name][index],
            ]
            for name in names
            for index in range(names_to_sizes[name])
        ]).T
        ax.errorbar(
            x_labels,
            values_total_order,
            yerr,
            label="Total order",
            transform=trans1,
            **errorbar_options,
        )
        ax.legend(loc="lower left")
        save_show_figure_from_file_path_manager(
            fig,
            self._file_path_manager if save else None,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
        )
        return fig
