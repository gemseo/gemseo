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
"""Sensitivity analysis based on the Sobol' indices."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import array
from numpy import asarray
from numpy import hstack
from numpy import sign
from numpy import vstack
from numpy import zeros
from numpy.random import default_rng
from openturns import RankSobolSensitivityAlgorithm
from openturns import Sample
from pandas import Series

from gemseo.algos.doe.factory import DOE_LIBRARY_FACTORY
from gemseo.algos.doe.openturns.settings.ot_sobol_indices import (
    OT_SOBOL_INDICES_Settings,
)
from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.heatmap import Heatmap
from gemseo.post.dataset.heatmap_settings import Heatmap_Settings
from gemseo.uncertainty.sensitivity._cv_sobol_algorithm import CVSobolAlgorithm
from gemseo.uncertainty.sensitivity._seeding import seed_ot_random_generator
from gemseo.uncertainty.sensitivity._sobol_indices_estimator import SobolAnalysisMethod
from gemseo.uncertainty.sensitivity._sobol_indices_estimator import (
    SobolIndicesEstimatorMixin,
)
from gemseo.uncertainty.sensitivity.base import BaseSensitivityAnalysis
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.seeder import SEED
from gemseo.utils.string_tools import get_name_and_component
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable

    from openturns import SobolIndicesAlgorithmImplementation

    from gemseo.algos.doe.base_doe_settings import BaseDOESettings
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline import Discipline
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.formulations.base_settings import BaseFormulationSettings
    from gemseo.scenarios.backup_settings import BackupSettings
    from gemseo.typing import RealArray
    from gemseo.typing import StrPath
    from gemseo.uncertainty.sensitivity.base import FirstOrderIndicesType
    from gemseo.uncertainty.sensitivity.base import SecondOrderIndicesType
    from gemseo.utils.string_tools import VariableType

LOGGER = logging.getLogger(__name__)


class SobolAnalysis(
    SobolIndicesEstimatorMixin, BaseSensitivityAnalysis[SobolAnalysisMethod]
):
    r"""Sensitivity analysis based on the Sobol' indices.

    Sobol' indices are variance-based sensitivity measures
    that quantify how much each input variable,
    individually or in combination with others,
    contributes to the variance of a model's output.

    Given independent random variables $X_1,\ldots,X_d$,
    the indices are grouped in order:

    - $S_i$ is the first-order Sobol' index
      measuring the individual effect of $X_i$,
    - $S_{i,j}$ is the second-order Sobol' index
      measuring the joint effect between $X_i$ and $X_j$,
    - $S_{i,j,k}$ is the third-order Sobol' index
      measuring the joint effect between $X_i$, $X_j$ and $X_k$,
    - and so on.

    The total Sobol' index $S^T_i$ is defined
    as the sum of the Sobol' indices associated with $X_i$.
    It represents the sum of the individual effect of $X_i$ and
    the joint effects between $X_i$ and any input variable or group of input variables.

    !!! quote "References"

        Andrea Saltelli, Paola Annoni, Ivano Azzini, Francesca Campolongo, Marco Ratto,
        and Stefano Tarantola.
        Variance based sensitivity analysis of model output design and estimator
        for the total sensitivity index.
        Computer physics communications, 181(2):259--270, 2010.
    """

    @dataclass
    class ControlVariate:
        """A control variate based on a cheap discipline.

        If either `indices` or `variance` is missing,
        both are estimated from `n_samples` evaluations of `discipline`.
        """

        discipline: Discipline
        """A cheap discipline, e.g. a surrogate discipline.

        It must have as inputs the input variables and the output variables
        used by `SobolAnalysis`.
        """

        indices: Mapping[SobolAnalysisMethod, FirstOrderIndicesType] = field(
            default_factory=dict
        )
        """The mapping between method names and first-order Sobol' indices.

        If empty, `SobolAnalysis` will compute it.
        """

        n_samples: int = 0
        """The number of samples to estimate the variance and the indices.

        If 0, use 100 times more samples than the number passed at instantiation.
        """

        variance: Mapping[str, RealArray] = field(default_factory=dict)
        """The mapping between output names and output variances.

        If empty, `SobolAnalysis` will compute it.
        """

    __output_standard_deviations: dict[str, RealArray]
    """The map between output names and standard deviations."""

    __output_variances: dict[str, RealArray]
    """The map between output names and variances."""

    __use_control_variates: bool
    """Whether to use control variates to estimate the indices."""

    DEFAULT_DRIVER: ClassVar[str] = "OT_SOBOL_INDICES"

    _DEFAULT_MAIN_METHOD: ClassVar[SobolAnalysisMethod] = SobolAnalysisMethod.FIRST

    def __init__(self, samples: IODataset | StrPath = "") -> None:  # noqa: D107
        super().__init__(samples)
        self.__use_control_variates = False
        self._output_name_to_sobol_algos = {}
        dataset = self.dataset
        if dataset is None or dataset.empty:
            self.__output_standard_deviations = {}
            self.__output_variances = {}
        elif "output_variances" in (misc := dataset.misc):
            self.__output_standard_deviations = misc["output_standard_deviations"]
            self.__output_variances = misc["output_variances"]
        else:
            output_variances = split_array_to_dict_of_arrays(
                dataset.get_view(group_names=dataset.OUTPUT_GROUP).to_numpy().var(0),
                dataset.variable_name_to_n_components,
                dataset.output_names,
            )
            self.__output_variances = output_variances
            self.__output_standard_deviations = {
                k: v**0.5 for k, v in output_variances.items()
            }

    def compute_samples(
        self,
        disciplines: Collection[Discipline],
        parameter_space: ParameterSpace,
        n_samples: int,
        output_names: str | Iterable[str] = (),
        algo_settings: BaseDOESettings | None = None,
        backup_settings: BackupSettings | None = None,
        formulation_settings: BaseFormulationSettings | None = None,
        compute_second_order: bool = True,
    ) -> IODataset:
        r"""
        Args:
            compute_second_order: Whether to compute the second-order indices.

        Notes:
             All the estimation techniques,
             except the rank-based one,
             expect samples generated using a pick-and-freeze DOE algorithm.
             This is the algorithm used when `algo_settings` is `None` (default).
             This algorithm starts with two independent input datasets
             composed of $N$ independent samples
             and this number $N$ is the usual sampling size for Sobol' analysis.
             When `compute_second_order=False`
             or when the input dimension $d$ is equal to 2,
             $N=\frac{n_\text{samples}}{2+d}$.
             Otherwise, $N=\frac{n_\text{samples}}{2+2d}$.
             The larger $N$,
             the more accurate the estimators of Sobol' indices are.
             Therefore,
             for a small budget `n_samples`,
             the user can choose to set `compute_second_order` to `False`
             to ensure a better estimation of the first- and total-order indices.
        """  # noqa: D205, D212, D415
        if algo_settings is None:
            algo_settings = DOE_LIBRARY_FACTORY.create_settings(self.DEFAULT_DRIVER)

        use_pick_and_freeze = isinstance(algo_settings, OT_SOBOL_INDICES_Settings)
        if use_pick_and_freeze:
            algo_settings.eval_second_order = compute_second_order
        elif compute_second_order:
            msg = (
                "The second-order indices can only be computed "
                "with the OT_SOBOL_INDICES algorithm."
            )
            LOGGER.warning(msg)
            compute_second_order = False

        super().compute_samples(
            disciplines,
            parameter_space,
            n_samples,
            output_names=output_names,
            algo_settings=algo_settings,
            backup_settings=backup_settings,
            formulation_settings=formulation_settings,
        )

        dataset = self.dataset
        dataset: IODataset
        n_inputs = parameter_space.dimension
        dataset.misc["use_pick_and_freeze"] = use_pick_and_freeze
        if use_pick_and_freeze:
            # If eval_second_order is set to False,
            # the input design is of size N(2+n_X).
            # If eval_second_order is set to True,
            #   if n_X = 2, the input design is of size N(2+n_X).
            #   if n_X != 2, the input design is of size N(2+2n_X).
            # Ref: https://openturns.github.io/openturns/latest/user_manual/_generated/
            # openturns.SobolIndicesExperiment.html#openturns.SobolIndicesExperiment
            sample_size = len(dataset) // (
                2 + n_inputs * (1 + (compute_second_order and n_inputs > 2))
            )
            dataset.misc["sample_size"] = sample_size
            output_variances = split_array_to_dict_of_arrays(
                dataset
                .get_view(group_names=dataset.OUTPUT_GROUP)
                .to_numpy()[: 2 * sample_size]
                .var(0),
                dataset.variable_name_to_n_components,
                dataset.output_names,
            )
        else:
            output_variances = split_array_to_dict_of_arrays(
                dataset.get_view(group_names=dataset.OUTPUT_GROUP).to_numpy().var(0),
                dataset.variable_name_to_n_components,
                dataset.output_names,
            )
            dataset.misc["sample_size"] = len(dataset)

        self.dataset.misc["eval_second_order"] = compute_second_order
        self.__output_variances = output_variances
        self.__output_standard_deviations = {
            k: v**0.5 for k, v in output_variances.items()
        }
        dataset.misc["parameter_space"] = parameter_space
        dataset.misc["n_inputs"] = n_inputs
        dataset.misc["output_variances"] = output_variances
        dataset.misc["output_standard_deviations"] = self.__output_standard_deviations
        return dataset

    @property
    def output_variances(self) -> dict[str, RealArray]:
        """The variances of the output variables."""
        return self.__output_variances

    @property
    def output_standard_deviations(self) -> dict[str, RealArray]:
        """The standard deviations of the output variables."""
        return self.__output_standard_deviations

    def __execute_cv(
        self,
        sample: Series,
        cv_d: Discipline,
    ) -> Series:
        """Execute a control variate on a sample.

        Args:
            sample: The sample on which the control variate is applied.
            cv_d: The discipline of the control variate.

        Returns:
            The outputs in a pandas series.
        """
        input_sample = sample[self.dataset.INPUT_GROUP]
        io_data = cv_d.execute({
            input_name: input_sample[input_name].to_numpy()
            for input_name in self._input_names
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
        algo: SobolAnalysis.Algorithm,
        confidence_level: float,
        use_asymptotic_distributions: bool,
        n_replicates: int,
        seed: int | None = None,
    ) -> SobolAnalysis.SensitivityIndices:
        """Compute the sensitivity indices with OpenTURNS capabilities.

        Args:
            output_names: The output names.
            algo: The algorithm name.
            confidence_level: The confidence level.
            use_asymptotic_distributions: Whether to estimate the confidence intervals
                using the asymptotic distributions.
                Otherwise, use the bootstrap method.
            n_replicates: The number of bootstrap replicates
                used for the computation of the confidence intervals.
            seed: The seed of the OpenTURNS random generator for bootstrapping.
                If `None`,
                the current state of `openturns.RandomGenerator` is used (no reseeding).

        Returns:
            The sensitivity indices.
        """
        algo_class = self._ALGO_NAME_TO_CLASS[algo]
        use_rank_algorithm = issubclass(algo_class, RankSobolSensitivityAlgorithm)
        # Bootstrap-based estimation (rank algorithm or non-asymptotic intervals)
        # consumes the random generator, so reseed it for reproducible results.
        do_seed = use_rank_algorithm or not use_asymptotic_distributions

        dataset = self.dataset
        input_data = Sample(
            dataset.get_view(
                group_names=dataset.INPUT_GROUP, variable_names=self._input_names
            ).to_numpy()
        )
        sample_size = dataset.misc.get("sample_size", len(input_data))
        with seed_ot_random_generator(seed if do_seed else None) as seeded:
            for output_name, _, data in self._iter_output_components(output_names):
                algos = self._output_name_to_sobol_algos.setdefault(output_name, [])
                if data is None:
                    algos.append(None)
                    continue

                ot_algo = self._build_sobol_algo(
                    algo_class,
                    input_data,
                    Sample(data),
                    sample_size,
                    n_replicates,
                    use_asymptotic_distributions,
                    confidence_level,
                )
                algos.append(ot_algo)
                # Prime bootstrap-based intervals inside the seeded scope so that
                # later get_intervals() calls do not depend on global RNG state.
                if seeded:
                    ot_algo.getFirstOrderIndicesInterval()
                    if not use_rank_algorithm:
                        ot_algo.getTotalOrderIndicesInterval()

        if use_rank_algorithm:
            self._indices = self.SensitivityIndices(
                first=self._get_sobol_indices(self._GET_FIRST_ORDER_INDICES),
            )
        else:
            self._indices = self.SensitivityIndices(
                first=self._get_sobol_indices(self._GET_FIRST_ORDER_INDICES),
                second=self._get_sobol_indices(self._GET_SECOND_ORDER_INDICES),
                total=self._get_sobol_indices(self._GET_TOTAL_ORDER_INDICES),
            )

        return self._indices

    def __compute_indices_using_cv(
        self,
        output_names: Iterable[str],
        control_variates: Iterable[ControlVariate],
        confidence_level: float,
        n_replicates: int,
        seed: int | None,
    ) -> SobolAnalysis.SensitivityIndices:
        """Compute the sensitivity indices using control variates.

        Args:
            output_names: The output names.
            control_variates: The control variates.
            confidence_level: The confidence level.
            n_replicates: The number of bootstrap replicates
                used for the computation of the confidence intervals.
            seed: The seed for reproducible results.
                If `None`,
                the current state of `openturns.RandomGenerator` is used
                (no reseeding).

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
            algos = self._output_name_to_sobol_algos[output_name] = []
            for i, sub_output_data in enumerate(output_data.T):
                if sub_output_data.var() == 0.0:
                    algos.append(None)
                    self.output_variances[output_name][i] = 0.0
                    continue

                sub_cvs_output_data = [
                    cv_output_data.T[i] for cv_output_data in cvs_output_data
                ]
                sub_cvs_statistics = [
                    (
                        cv.variance[output_name][i],
                        {
                            method: getattr(
                                cv.indices, self._get_index_field_name(method)
                            )[output_name][i]
                            for method in list(SobolAnalysisMethod)
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
            first=self._get_sobol_indices("compute_first_indices"),
            total=self._get_sobol_indices("compute_total_indices"),
        )
        return self._indices

    def compute_indices(
        self,
        output_names: str | Iterable[str] = (),
        algo: SobolAnalysis.Algorithm | None = None,
        confidence_level: float = 0.95,
        control_variates: ControlVariate | Iterable[ControlVariate] = (),
        use_asymptotic_distributions: bool = True,
        n_replicates: int = 100,
        seed: int | None = SEED,
    ) -> SobolAnalysis.SensitivityIndices:
        """
        Args:
            algo: The name of the algorithm
                to estimate the Sobol' indices from samples.
                All the algorithms assume that the samples have been generated
                using the `OT_SOBOL_INDICES` DOE algorithm,
                except for `Algorithm.RANK`,
                which assumes that they have been generated
                using crude Monte Carlo or quasi-Monte Carlo.
                If `None`,
                use `Algorithm.SALTELLI` or `Algorithm.RANK` according to the samples.
            confidence_level: The confidence level
                of the confidence intervals associated with the estimates.
            control_variates: The control variates based on cheap disciplines, if any.
                The use of control variates is not compatible
                with rank-based estimation (i.e. `algo=Algorithm.RANK`).
            use_asymptotic_distributions: Whether to compute
                the confidence intervals
                of the first- and total-order Sobol' indices
                using the asymptotic distributions;
                otherwise, use the bootstrap method.
                When control variates are used
                or when the algorithm is `Algorithm.RANK` (or `"Rank"`),
                the confidence intervals can only be estimated via bootstrap,
                and so, this argument is ignored.
            n_replicates: The number of bootstrap replicates
                used for the computation of the confidence intervals.
            seed: The seed of the random generator for bootstrapping.
                If `None`,
                the current state of `openturns.RandomGenerator` will be used
                (no reseeding) in the standard case,
                and fresh, unpredictable entropy will be pulled from the OS
                when control variates are used.

        Raises:
            ValueError: If control variates are provided
                and the algorithm is not `Algorithm.SALTELLI`,
                if `Algorithm.RANK` is used with pick-and-freeze (PF) samples
                or if another algorithm is used with non-PF samples.
        """  # noqa:D205,D212,D415
        use_pick_and_freeze = self.dataset.misc.get("use_pick_and_freeze", False)
        algo = self._select_sobol_algorithm(algo, use_pick_and_freeze)
        output_names = self._get_output_names(output_names)
        self._output_name_to_sobol_algos = {}
        self.__use_control_variates = bool(control_variates)
        if control_variates:
            if algo != self.Algorithm.SALTELLI:
                msg = (
                    "The Saltelli algorithm is required "
                    "for the use of control variates."
                )
                raise ValueError(msg)

            if isinstance(control_variates, self.ControlVariate):
                control_variates = [control_variates]

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
            seed,
        )

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

    def _get_interval_bounds(
        self,
        sobol_algorithm: SobolIndicesAlgorithmImplementation | CVSobolAlgorithm | None,
        first_order: bool,
    ) -> tuple[RealArray, RealArray]:
        """Return the lower and upper bounds of a Sobol' index confidence interval.

        Args:
            sobol_algorithm: The OpenTURNS or control-variate Sobol' indices
                algorithm.
                If `None`, i.e. the output has zero variance and no algorithm was
                built for it, return degenerate (zero) bounds.
            first_order: Whether the confidence interval is for a first-order index;
                otherwise, for a total-order index.

        Returns:
            The lower and upper bounds of the confidence interval.
        """
        if not self.__use_control_variates:
            return super()._get_interval_bounds(sobol_algorithm, first_order)

        if sobol_algorithm is None:
            name_to_size = self.dataset.variable_name_to_n_components
            n_inputs = sum(name_to_size[name] for name in self._input_names)
            zero_bounds = zeros(n_inputs)
            return zero_bounds, zero_bounds

        interval = (
            sobol_algorithm.first_indices_interval
            if first_order
            else sobol_algorithm.total_indices_interval
        )
        return interval[0], interval[1]

    def _get_plot_title(self, output_name: str, output_component: int) -> str:
        """Return the default plot title for an output component.

        Args:
            output_name: The name of the output.
            output_component: The component of the output.

        Returns:
            The default plot title.
        """
        pretty_output_name = repr_variable(
            output_name,
            output_component,
            len(self.indices.first[output_name]),
        )
        return f"Sobol' indices for the output {pretty_output_name}"

    def _get_plot_subtitle(self, output_name: str, output_component: int) -> str:
        """Return the plot subtitle for an output component.

        The subtitle displays the standard deviation (StD) and the variance (Var)
        of the output of interest.

        Args:
            output_name: The name of the output.
            output_component: The component of the output.

        Returns:
            The plot subtitle.
        """
        variance = self.output_variances[output_name][output_component]
        return f"Var={variance:.1e}    StD={variance**0.5:.1e}"

    def plot_second_order(
        self,
        output: VariableType,
        settings: Heatmap_Settings | None = None,
    ) -> Heatmap:
        """Plot the second-order Sobol' indices using a symmetric heat map.

        Args:
            output: The output of interest.
                Either a name or a tuple of the form (name, component).
                If name, its first component is considered.
            settings: The settings of the heat map.
                The `"symmetric"` option will be set to `True`.

        Returns:
            The heat map of the second-order Sobol' indices.
        """
        output_name, output_component = get_name_and_component(output)
        indices = self.indices.second[output_name][output_component]
        name_to_size = self.dataset.variable_name_to_n_components
        components = [
            (name, index) for name in indices for index in range(name_to_size[name])
        ]
        variables = [
            repr_variable(name, index, name_to_size[name]) for name, index in components
        ]
        n = len(variables)
        data = zeros((n, n))
        for i, (name_i, component_i) in enumerate(components):
            indices_i = indices[name_i]
            for j, (name_j, component_j) in enumerate(components):
                index = asarray(indices_i[name_j])[component_i, component_j]
                data[i, j] = max(index, 0.0)

        dataset = Dataset.from_array(data, variable_names=variables)
        settings_kwargs = settings.model_dump() if settings is not None else {}
        settings_kwargs["symmetric"] = True
        return Heatmap(dataset, settings=Heatmap_Settings(**settings_kwargs))
