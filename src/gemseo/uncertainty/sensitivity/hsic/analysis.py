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
#        :author: Olivier Sapin
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Sensitivity analysis based on the Hilbert-Schmidt independence criterion (HSIC)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from numpy import array
from numpy import newaxis
from openturns import CovarianceModelImplementation
from openturns import HSICEstimatorConditionalSensitivity
from openturns import HSICEstimatorGlobalSensitivity
from openturns import HSICEstimatorImplementation
from openturns import HSICEstimatorTargetSensitivity
from openturns import HSICStatImplementation
from openturns import HSICUStat
from openturns import HSICVStat
from openturns import IndicatorFunction
from openturns import Interval
from openturns import RandomGenerator
from openturns import Sample
from openturns import SquaredExponential
from strenum import StrEnum

from gemseo import SEED
from gemseo.uncertainty.sensitivity.analysis import FirstOrderIndicesType
from gemseo.uncertainty.sensitivity.analysis import SensitivityAnalysis
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.algos.doe.doe_library import DOELibraryOptionType
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline import MDODiscipline


class HSICAnalysis(SensitivityAnalysis):
    """Sensitivity analysis based on the Hilbert-Schmidt independence criterion (HSIC).

    The :meth:`.compute_indices` method proposes three types of sensitivity analysis:

    - global sensitivity analysis (GSA, default)
      to identify the input variables
      most likely to cause the output to deviate from a nominal value,
    - target sensitivity analysis (TSA)
      to identify the input variables
      most likely to cause the output to reach a certain domain,
    - conditional sensitivity analysis (CSA)
      most likely to cause the output to deviate from a nominal value
      under the condition that the considered output is in a certain domain.

    For GSA and CSA,
    the sensitivity indices can be estimated with both U- and V- statistics.
    For TSA,
    only U-statistics is possible.

    Given a sensitivity analysis type and a statistics estimation technique,
    the :meth:`.compute_indices` method returns the standard HSIC indices
    and the normalized ones, also called R2-HSIC indices.

    Examples:
        >>> from numpy import pi
        >>> from gemseo import create_discipline, create_parameter_space
        >>> from gemseo.uncertainty.sensitivity.hsic.analysis import HSICAnalysis
        >>> from gemseo.uncertainty.use_cases.ishigami.ishigami_discipline import (
        ...     IshigamiDiscipline,
        ... )
        >>> from gemseo.uncertainty.use_cases.ishigami.ishigami_space import (
        ...     IshigamiSpace,
        ... )
        >>>
        >>> discipline = IshigamiDiscipline()
        >>> uncertain_space = IshigamiSpace()
        >>>
        >>> analysis = HSICAnalysis([discipline], uncertain_space, n_samples=1000)
        >>> indices = analysis.compute_indices()
    """

    @staticmethod
    def __compute_covariance_models(
        sample: Sample,
        covariance_model_class: type[
            CovarianceModelImplementation
        ] = SquaredExponential,
    ) -> Sequence[CovarianceModelImplementation]:
        r"""Create :math:`d` covariance models.

        Args:
            sample: A sample of a :math:`d`-length vector.
            covariance_model_class: A name of covariance model class.

        Returns:
            One covariance model per vector dimension.
        """
        covariance_models = []
        for i in range(sample.getDimension()):
            scale = sample.getMarginal(i).computeStandardDeviation()
            covariance_model = covariance_model_class(1)
            covariance_model.setScale(scale)
            covariance_models.append(covariance_model)

        return covariance_models

    class Method(StrEnum):
        """The name of the sensitivity method."""

        HSIC = "HSIC"
        """The HSIC indices."""

        P_VALUE_ASYMPTOTIC = "p-value-asymptotic"
        """The p-value obtained with an asymptotic formula."""

        P_VALUE_PERMUTATION = "p-value-permutation"
        """The p-value estimated through permutations."""

        R2_HSIC = "R2-HSIC"
        """The normalized HSIC (R2-HSIC) indices."""

    __METHODS_TO_OT_METHODS: Final[dict[Method, str]] = {
        Method.HSIC: "getHSICIndices",
        Method.P_VALUE_ASYMPTOTIC: "getPValuesAsymptotic",
        Method.P_VALUE_PERMUTATION: "getPValuesPermutation",
        Method.R2_HSIC: "getR2HSICIndices",
    }

    class AnalysisType(StrEnum):
        """The sensitivity analysis type."""

        GLOBAL = "global"
        """Global sensitivity analysis."""

        TARGET = "target"
        """Target sensitivity analysis."""

        CONDITIONAL = "conditional"
        """Conditional sensitivity analysis.

        This sensitivity analysis is incompatible
        with :attr:`.StatisticEstimator.USTAT`.
        """

    __ANALYSIS_TO_OT_CLASSES: Final[
        dict[AnalysisType, type[HSICEstimatorImplementation]]
    ] = {
        AnalysisType.CONDITIONAL: HSICEstimatorConditionalSensitivity,
        AnalysisType.GLOBAL: HSICEstimatorGlobalSensitivity,
        AnalysisType.TARGET: HSICEstimatorTargetSensitivity,
    }

    class StatisticEstimator(StrEnum):
        """The statistic estimator type."""

        USTAT = "U-statistic"
        """U-statistic."""

        VSTAT = "V-statistic"
        """V-statistic."""

    __STATISTIC_ESTIMATORS_TO_OT_CLASSES: Final[
        dict[StatisticEstimator, type[HSICStatImplementation]]
    ] = {
        StatisticEstimator.USTAT: HSICUStat,
        StatisticEstimator.VSTAT: HSICVStat,
    }

    class CovarianceModel(StrEnum):
        """The covariance model type."""

        GAUSSIAN = "Gaussian"
        """Squared exponential covariance model."""

    __COVARIANCE_MODELS_TO_OT_CLASSES: Final[
        dict[CovarianceModel, type[CovarianceModelImplementation]]
    ] = {
        CovarianceModel.GAUSSIAN: SquaredExponential,
    }

    DEFAULT_DRIVER: ClassVar[str] = "OT_MONTE_CARLO"

    def __init__(  # noqa: D107
        self,
        disciplines: Collection[MDODiscipline],
        parameter_space: ParameterSpace,
        n_samples: int,
        output_names: Iterable[str] = (),
        algo: str = "",
        algo_options: Mapping[str, DOELibraryOptionType] = READ_ONLY_EMPTY_DICT,
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
        self._main_method = self.Method.R2_HSIC

    def compute_indices(
        self,
        outputs: str
        | Sequence[str]
        | Mapping[str, tuple[Iterable[float], Iterable[float]]] = (),
        statistic_estimator: StatisticEstimator = StatisticEstimator.USTAT,
        input_covariance_model: CovarianceModel = CovarianceModel.GAUSSIAN,
        output_covariance_model: CovarianceModel = CovarianceModel.GAUSSIAN,
        analysis_type: AnalysisType = AnalysisType.GLOBAL,
        seed: int = SEED,
        n_permutations: int = 100,
    ) -> dict[str, FirstOrderIndicesType]:
        """
        Args:
            outputs: For global sensitivity analysis,
                the name(s) of the output(s)
                for which to compute the sensitivity indices;
                if empty, use the names of the outputs set at instantiation.
                For target and conditional sensitivity analyses,
                the names and the lower and upper bounds of these outputs
                specified as ``{name: (lower, upper), ...}``;
                this argument is mandatory.
            statistic_estimator: The name of the statistic estimator type.
                This argument is ignored
                when ``analysis_type`` is :attr:`~.AnalysisType.CONDITIONAL`;
                in this case,
                the U-statistics do not exist and V-statistics are considered.
            input_covariance_model: The name of the covariance model class of the
                estimator associated to the input variables.
            output_covariance_model: The name of the covariance model class of the
                estimator associated to the output variables.
            analysis_type: The sensitivity analysis type.
            seed: The seed for reproducible results.
            n_permutations: The number of permutations used to estimate the p-values.
        """  # noqa: D205 D212 D415
        RandomGenerator.SetSeed(seed)
        if analysis_type == analysis_type.CONDITIONAL:
            statistic_estimator = self.StatisticEstimator.VSTAT

        outputs = outputs or self.default_output
        if isinstance(outputs, str):
            outputs = [outputs]

        statistic_estimator_class = self.__STATISTIC_ESTIMATORS_TO_OT_CLASSES[
            statistic_estimator
        ]

        if analysis_type in [analysis_type.CONDITIONAL, analysis_type.TARGET]:
            outputs = {name: tuple(zip(*value)) for name, value in outputs.items()}

        input_covariance_model_class = self.__COVARIANCE_MODELS_TO_OT_CLASSES[
            input_covariance_model
        ]
        output_covariance_model_class = self.__COVARIANCE_MODELS_TO_OT_CLASSES[
            output_covariance_model
        ]
        input_samples = Sample(
            self.dataset.get_view(group_names=self.dataset.INPUT_GROUP).to_numpy()
        )
        hsic_class = self.__ANALYSIS_TO_OT_CLASSES[analysis_type]
        self._indices = {}
        for method in self.Method:
            indices = self._indices[method] = {}
            if (
                method == method.P_VALUE_ASYMPTOTIC
                and analysis_type == analysis_type.CONDITIONAL
            ):
                continue

            sizes = self.dataset.variable_names_to_n_components
            input_covariance_models = self.__compute_covariance_models(
                input_samples, input_covariance_model_class
            )

            for output_name in outputs:
                output_indices = []
                for i, output_component_samples in enumerate(
                    self.dataset.get_view(
                        group_names=self.dataset.OUTPUT_GROUP,
                        variable_names=output_name,
                    )
                    .to_numpy()
                    .T
                ):
                    output_samples = Sample(output_component_samples[:, newaxis])
                    output_covariance_models = self.__compute_covariance_models(
                        output_samples, output_covariance_model_class
                    )
                    covariance_models = [
                        *input_covariance_models,
                        *output_covariance_models,
                    ]
                    if analysis_type == analysis_type.GLOBAL:
                        args = (statistic_estimator_class(),)
                    elif analysis_type == analysis_type.TARGET:
                        args = (
                            statistic_estimator_class(),
                            IndicatorFunction(Interval(*outputs[output_name][i])),
                        )
                    else:
                        args = (IndicatorFunction(Interval(*outputs[output_name][i])),)

                    hsic_estimator = hsic_class(
                        covariance_models, input_samples, output_samples, *args
                    )
                    hsic_estimator.setPermutationSize(n_permutations)

                    get_indices = getattr(
                        hsic_estimator, self.__METHODS_TO_OT_METHODS[method]
                    )
                    output_indices.append(
                        split_array_to_dict_of_arrays(
                            array(get_indices()),
                            sizes,
                            self._input_names,
                        )
                    )

                indices[output_name] = output_indices

        return self.indices

    @property
    def hsic(self) -> FirstOrderIndicesType:
        """The HSIC indices.

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
        return self._indices[self.Method.HSIC]

    @property
    def p_value_asymptotic(self) -> FirstOrderIndicesType:
        """The p-value obtained with an asymptotic formula.

        With the following structure:

        .. code-block:: python

            {
                "output_name": [
                    {
                        "input_name": data_array,
                    }
                ]
            }

        .. note:: Not yet implemented in OpenTURNS for conditional analysis.
        """
        return self._indices[self.Method.P_VALUE_ASYMPTOTIC]

    @property
    def p_value_permutation(self) -> FirstOrderIndicesType:
        """The p-value estimated through permutations.

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
        return self._indices[self.Method.P_VALUE_PERMUTATION]

    @property
    def r2_hsic(self) -> FirstOrderIndicesType:
        """The normalized HSIC indices.

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
        return self._indices[self.Method.R2_HSIC]
