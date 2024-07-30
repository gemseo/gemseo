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

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
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

from gemseo.uncertainty.sensitivity.base_sensitivity_analysis import (
    BaseSensitivityAnalysis,
)
from gemseo.uncertainty.sensitivity.base_sensitivity_analysis import (
    FirstOrderIndicesType,
)
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.seeder import SEED

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence


class HSICAnalysis(BaseSensitivityAnalysis):
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
        >>> from gemseo.uncertainty.sensitivity.hsic_analysis import HSICAnalysis
        >>> from gemseo.problems.uncertainty.ishigami.ishigami_discipline import (
        ...     IshigamiDiscipline,
        ... )
        >>> from gemseo.problems.uncertainty.ishigami.ishigami_space import (
        ...     IshigamiSpace,
        ... )
        >>>
        >>> discipline = IshigamiDiscipline()
        >>> uncertain_space = IshigamiSpace()
        >>>
        >>> analysis = HSICAnalysis()
        >>> analysis.compute_samples([discipline], uncertain_space, n_samples=1000)
        >>> indices = analysis.compute_indices()
    """

    @dataclass(frozen=True)
    class SensitivityIndices:  # noqa: D106
        hsic: FirstOrderIndicesType = field(default_factory=dict)
        """The HSIC indices."""

        r2_hsic: FirstOrderIndicesType = field(default_factory=dict)
        """The normalized HSIC indices."""

        p_value_permutation: FirstOrderIndicesType = field(default_factory=dict)
        """The p-value estimated through permutations."""

        p_value_asymptotic: FirstOrderIndicesType = field(default_factory=dict)
        """The p-value obtained with an asymptotic formula."""

    _indices: SensitivityIndices

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
            covariance_model = covariance_model_class(1)
            covariance_model.setScale(sample.getMarginal(i).computeStandardDeviation())
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
    """The mapping from the sensitivity indices to the OT classes."""

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
    """The mapping from the analysis types to the OT classes."""

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
    """The mapping from the statistic estimators to the OT classes."""

    class CovarianceModel(StrEnum):
        """The covariance model type."""

        GAUSSIAN = "Gaussian"
        """Squared exponential covariance model."""

    __COVARIANCE_MODELS_TO_OT_CLASSES: Final[
        dict[CovarianceModel, type[CovarianceModelImplementation]]
    ] = {
        CovarianceModel.GAUSSIAN: SquaredExponential,
    }
    """The mapping from the covariance model names to the OT classes."""

    DEFAULT_DRIVER: ClassVar[str] = "OT_MONTE_CARLO"

    _DEFAULT_MAIN_METHOD: ClassVar[Method] = Method.R2_HSIC

    def compute_indices(
        self,
        output_names: str | Iterable[str] = (),
        output_bounds: Mapping[str, tuple[Iterable[float], Iterable[float]]] = (),
        statistic_estimator: StatisticEstimator = StatisticEstimator.USTAT,
        input_covariance_model: CovarianceModel = CovarianceModel.GAUSSIAN,
        output_covariance_model: CovarianceModel = CovarianceModel.GAUSSIAN,
        analysis_type: AnalysisType = AnalysisType.GLOBAL,
        seed: int = SEED,
        n_permutations: int = 100,
    ) -> SensitivityIndices:
        """
        Args:
            output_names: The name(s) of the output(s)
                for which to compute the sensitivity indices.
                If empty,
                use the names of the outputs set at instantiation.
                In the case of target and conditional sensitivity analyses,
                these output names are the keys of the dictionary ``output_bounds``
                and the argument ``output_names`` is ignored.
            output_bounds: The lower and upper bounds of the outputs
                specified as ``{name: (lower, upper), ...}``.
                This argument is ignored in the case of global sensitivity analysis.
                This argument is mandatory
                in the case of target and conditional sensitivity analyses.
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

        output_names = self._get_output_names(output_names)

        statistic_estimator_class = self.__STATISTIC_ESTIMATORS_TO_OT_CLASSES[
            statistic_estimator
        ]

        if analysis_type in {analysis_type.CONDITIONAL, analysis_type.TARGET}:
            output_bounds = {
                name: tuple(zip(*value)) for name, value in output_bounds.items()
            }
            output_names = output_bounds.keys()

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
        indices = {}
        for method in self.Method:
            _indices = indices[str(method).lower().replace("-", "_")] = {}
            if (
                method == method.P_VALUE_ASYMPTOTIC
                and analysis_type == analysis_type.CONDITIONAL
            ):
                continue

            sizes = self.dataset.variable_names_to_n_components
            input_covariance_models = self.__compute_covariance_models(
                input_samples, input_covariance_model_class
            )

            for output_name in output_names:
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
                            IndicatorFunction(Interval(*output_bounds[output_name][i])),
                        )
                    else:
                        args = (
                            IndicatorFunction(Interval(*output_bounds[output_name][i])),
                        )

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

                _indices[output_name] = output_indices

        self._indices = self.SensitivityIndices(**indices)
        return self.indices
