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
"""Reliability-oriented Sobol' analysis combining FORM and importance sampling."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from numpy import array
from numpy import exp
from numpy import isnan
from numpy import newaxis
from numpy import vstack
from numpy import zeros
from openturns import CorrelationMatrix
from openturns import Normal
from openturns import Point
from scipy.stats import norm

from gemseo.dataset.io_dataset import IODataset
from gemseo.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.doe.factory import doe_library_factory
from gemseo.doe.openturns._algorithm.ot_sobol_doe import OTSobolDOE
from gemseo.doe.openturns.settings.ot_sobol_indices import OT_SOBOL_INDICES_Settings
from gemseo.scenario.evaluation import EvaluationScenario
from gemseo.uncertainty.reliability.openturns.form_settings import OT_FORM_Settings
from gemseo.uncertainty.reliability.scenario import ReliabilityScenario
from gemseo.uncertainty.sensitivity._is_sobol_indices_estimator import (
    ISSobolIndicesEstimator,
)
from gemseo.uncertainty.sensitivity._seeding import seed_ot_random_generator
from gemseo.uncertainty.sensitivity._sobol_indices_estimator import SobolAnalysisMethod
from gemseo.uncertainty.sensitivity._sobol_indices_estimator import (
    SobolIndicesEstimatorMixin,
)
from gemseo.uncertainty.sensitivity.base_ro import BaseROSensitivityAnalysis
from gemseo.util.seeder import seed

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline
    from gemseo.doe.core.base_doe_settings import BaseDOESettings
    from gemseo.formulation.core.base_settings import BaseFormulationSettings
    from gemseo.space.random import RandomSpace
    from gemseo.uncertainty.reliability.event import Event
    from gemseo.util.typing import RealArray
    from gemseo.util.typing import StrPath


logger = logging.getLogger(__name__)


class ISFORMSobolAnalysis(
    SobolIndicesEstimatorMixin, BaseROSensitivityAnalysis[SobolAnalysisMethod]
):
    r"""Reliability-oriented Sobol' analysis based on FORM and importance sampling.

    This reliability-oriented sensitivity analysis estimates
    the Sobol' indices of a binary event,
    e.g. a disciplinary output exceeding a threshold.

    !!! warning "These Sobol' indices are not for raw disciplinary outputs."
        For the Sobol' indices of a disciplinary output,
        it is best to use
        [SobolAnalysis][gemseo.uncertainty.sensitivity.sobol.SobolAnalysis].

    A crude Monte Carlo estimation of these indices is intractable for rare events.
    Instead, this analysis combines three ingredients:

    - the first-order reliability method (FORM)
      to locate the most probable failure point (MPFP), a.k.a. design point,
    - an importance sampling (IS) auxiliary density,
      namely a unit-variance normal distribution
      centered on the design point in the standard space,
      so that the samples land around the limit state,
    - IS estimators of the Sobol' indices of the binary event
      using this IS auxiliary density.

    As the auxiliary density depends on the design point of an event,
    each event is processed independently
    and the Sobol' indices are estimated event by event.

    See the [rare events][concept-isformsobol-analysis] part
    of the sensitivity analysis concepts
    for the estimators, their confidence intervals and their cost.

    !!! note "The Sobol' indices are computed in the standard space"
        The pick-and-freeze design is drawn from the IS auxiliary density
        in the standard space,
        so the indices quantify the influence of the *standard* uncertain inputs.
        For independent marginals,
        these standard inputs map one-to-one to the physical inputs
        and share their names.
    """

    def __init__(self, samples: IODataset | StrPath = "") -> None:  # noqa: D107
        super().__init__(samples)
        self._output_name_to_estimators = {}
        dataset = self.dataset
        # The variance of the indicator of an event of probability p is p(1-p).
        event_to_probability = (
            {}
            if dataset is None or dataset.empty
            else dataset.misc.get("probability", {})
        )
        self._set_output_variances({
            event_name: array([probability * (1.0 - probability)])
            for event_name, probability in event_to_probability.items()
        })

    def compute_samples(  # noqa: D102
        self,
        disciplines: Sequence[Discipline],
        random_space: RandomSpace,
        events: Mapping[str, Event],
        n_samples: int,
        algo_settings: BaseDOESettings | None = None,
        form_settings: OT_FORM_Settings | None = None,
        formulation_settings: BaseFormulationSettings | None = None,
        compute_second_order: bool = True,
        seed: int = seed,
    ) -> IODataset:
        r"""
        Args:
            events: The events of interest,
                indexed by their names,
                e.g. `{"y_high": y > 3.0}` with `y = analysis.get_event_variables("y")`.
                Each event is processed independently,
                with its own FORM design point, auxiliary density and sampling design.
                Reminder: FORM and SORM do not support event combinations,
                e.g. `{"y_high_z_low": (y > 3.0) & (z < 5.6)}`.
            n_samples: The maximum total number of model evaluations
                spent over all the events,
                including the evaluations performed by FORM
                to locate the design point of every event.
                Once every design point has been located,
                the budget left is split equally between the events
                to draw their Sobol' samples.
            algo_settings: The settings of the DOE algorithm.
                If `None`, use the pick-and-freeze strategy.
            form_settings: The settings
                of the first-order or second-order reliability method (FORM/SORM)
                used to locate the design points.
                If `None`, use the default settings of the FORM algorithm.
            formulation_settings: The settings of the MDO formulation.
                If `None`, use the default settings of the MDF formulation.
            compute_second_order: Whether to compute the second-order indices.
                Only available with a pick-and-freeze design.
            seed: The seed of the OpenTURNS random generator.

        Notes:
             All the estimation techniques,
             except the rank-based one,
             expect samples generated using a pick-and-freeze DOE algorithm.
             This is the algorithm used when `algo_settings` is `None` (default).
             After the FORM stage,
             this algorithm starts with two independent input datasets
             composed of $N$ independent samples
             and this number $N$ is the usual sampling size for Sobol' analysis.
             When `compute_second_order=False`
             or when the input dimension $d$ is equal to 2,
             $N=\frac{b}{2+d}$
             with $b=(n_\text{samples}-n_\text{FORM})/n_\text{events}$
             where $n_\text{samples}$ is the total budget,
             $n_\text{FORM}$ is the total number of FORM evaluations,
             and $n_\text{events}$ is the number of events.
             Otherwise, $N=\frac{b}{2+2d}$.
             The larger $N$,
             the more accurate the estimators of Sobol' indices are.
             Therefore,
             for a small budget `n_samples`,
             the user can choose to set `compute_second_order` to `False`
             to ensure a better estimation of the first- and total-order indices.
        """  # noqa: D205, D212, D415
        if algo_settings is None:
            algo_settings = doe_library_factory.create_settings("OT_SOBOL_INDICES")

        use_pick_and_freeze = isinstance(algo_settings, OT_SOBOL_INDICES_Settings)

        # Only pick-and-freeze strategy can compute second-order indices.
        if not use_pick_and_freeze:
            compute_second_order = False

        dimension = random_space.dimension

        event_names = list(events)
        n_events = len(events)
        event_to_standard_samples = []
        event_to_indicator = []
        event_to_sample_size = {}
        event_to_probability = {}
        event_to_design_point = {}

        # First pass: locate the design point of every event with FORM.
        # The FORM evaluations are part of the shared n_samples budget,
        # so they must all be known before allocating the sampling budget.
        n_form_evaluations_total = 0
        for event_name, event in events.items():
            standard_design_point, n_form_evaluations = (
                self.__compute_standard_design_point(
                    disciplines,
                    random_space,
                    event_name,
                    event,
                    form_settings or OT_FORM_Settings(),
                    formulation_settings,
                )
            )
            event_to_design_point[event_name] = standard_design_point
            n_form_evaluations_total += n_form_evaluations

        # n_samples is the total budget of model evaluations over all the events,
        # FORM included; the rest is split equally between the events
        # and spent on their Sobol' sampling.
        sampling_budget = n_samples - n_form_evaluations_total
        sampling_factor = 2 + dimension * (1 + (compute_second_order and dimension > 2))
        # Only the pick-and-freeze design needs N(2+d) (or N(2+2d)) rows per event;
        # the Rank/i.i.d. design only needs at least one sample per event.
        min_sampling_budget = (
            n_events * sampling_factor if use_pick_and_freeze else n_events
        )
        if sampling_budget < min_sampling_budget:
            msg = (
                f"n_samples ({n_samples}) is too small to draw a sample "
                f"for each of the {n_events} event(s) "
                f"after their {n_form_evaluations_total} FORM evaluations; "
                "increase n_samples."
            )
            raise ValueError(msg)

        budget = sampling_budget // n_events

        # Second pass: draw the Sobol' samples of every event with its share
        # of the sampling budget.
        for event_name, event in events.items():
            standard_design_point = event_to_design_point[event_name]

            if use_pick_and_freeze:
                # The OTSobolDOE pick-and-freeze design is the largest one whose
                # number of rows N(2+d) (or N(2+2d)) does not exceed the budget.
                sample_size = budget // sampling_factor

                # OTSobolDOE draws a unit-hypercube design; map it to the auxiliary
                # density by shifting the standard normal quantiles of the unit
                # samples by the design point
                # (the auxiliary marginals are independent unit-variance normals).
                with seed_ot_random_generator(seed):
                    unit_samples = OTSobolDOE().generate_samples(
                        dimension,
                        OT_SOBOL_INDICES_Settings(
                            n_samples=budget,
                            eval_second_order=compute_second_order,
                            seed=seed,
                        ),
                    )
                standard_samples = norm.ppf(unit_samples) + standard_design_point
            else:
                # The auxiliary IS density is a unit-variance normal distribution
                # centered on the design point in the standard space.
                auxiliary_distribution = Normal(
                    Point(standard_design_point),
                    Point([1.0] * dimension),
                    CorrelationMatrix(dimension),
                )
                sample_size = budget
                with seed_ot_random_generator(seed):
                    standard_samples = array(auxiliary_distribution.getSample(budget))

            # Map the standard samples to the physical space.
            ot_distribution = random_space.variables.distribution.distribution
            inverse_transform = (
                ot_distribution.getInverseIsoProbabilisticTransformation()
            )
            physical_samples = array(inverse_transform(standard_samples))

            output_values = self.__evaluate_model(
                disciplines,
                random_space,
                event,
                physical_samples,
                formulation_settings,
            )

            # The event probability is the IS estimate mean(w * 1_F),
            # where w = phi(u) / phi(u - u*) is the likelihood ratio
            # between the true standard normal density and the auxiliary density
            # and 1_F is the event indicator; log w = -u* . u + |u*|^2 / 2.
            indicator = event.evaluate(output_values).astype(float)
            weights = exp(
                -standard_samples @ standard_design_point
                + standard_design_point @ standard_design_point / 2
            )

            event_to_standard_samples.append(standard_samples)
            event_to_indicator.append(indicator[:, newaxis])
            event_to_sample_size[event_name] = sample_size
            event_to_probability[event_name] = float((weights * indicator).mean())

        variables = random_space.variables
        variable_names = list(variables)
        dataset = self.__create_dataset(
            event_names,
            event_to_standard_samples,
            event_to_indicator,
            variable_names,
            {name: variable.size for name, variable in variables.items()},
        )
        dataset.misc["use_pick_and_freeze"] = use_pick_and_freeze
        dataset.misc["eval_second_order"] = compute_second_order
        dataset.misc["sample_size"] = event_to_sample_size
        dataset.misc["probability"] = event_to_probability
        dataset.misc["design_point"] = event_to_design_point
        self.dataset = dataset
        self._input_names = variable_names
        self._output_names = event_names
        return dataset

    @staticmethod
    def __compute_standard_design_point(
        disciplines: Sequence[Discipline],
        random_space: RandomSpace,
        event_name: str,
        event: Event,
        form_settings: OT_FORM_Settings,
        formulation_settings: BaseFormulationSettings | None,
    ) -> tuple[RealArray, int]:
        """Locate the design point of an event in the standard space using FORM.

        Args:
            disciplines: The disciplines that make up the model.
            random_space: The random space.
            event_name: The name of the event.
            event: The event of interest.
            form_settings: The settings of the FORM algorithm.
            formulation_settings: The settings of the MDO formulation.

        Returns:
            The design point in the standard space
            and the number of model evaluations performed by FORM.
        """
        scenario = ReliabilityScenario(
            disciplines, random_space, formulation_settings=formulation_settings
        )
        scenario.add_event(event, event_name)
        scenario.execute(form_settings)
        result = scenario.event_name_to_reliability_result[event_name]
        n_evaluations = result.raw_result.getOptimizationResult().getCallsNumber()
        return result.design_point.standard, n_evaluations

    @staticmethod
    def __evaluate_model(
        disciplines: Sequence[Discipline],
        random_space: RandomSpace,
        event: Event,
        physical_samples: RealArray,
        formulation_settings: BaseFormulationSettings | None,
    ) -> dict[str, RealArray]:
        """Evaluate the disciplinary outputs of an event at given physical samples.

        Args:
            disciplines: The disciplines that make up the model.
            random_space: The random space.
            event: The event of interest.
            physical_samples: The samples in the physical space,
                shaped as `(n_samples, input_dimension)`.
            formulation_settings: The settings of the MDO formulation.

        Returns:
            The disciplinary output samples indexed by output name.
        """
        output_names = {
            elementary_event.name
            for intersection_event in event
            for elementary_event in intersection_event
        }
        scenario = EvaluationScenario(
            disciplines,
            random_space,
            name="ISFORMSobolAnalysisSamplingPhase",
            formulation_settings=formulation_settings,
        )
        for output_name in output_names:
            scenario.add_observable(output_name)

        scenario.execute(CustomDOE_Settings(samples=physical_samples))
        dataset = scenario.to_dataset()
        output_name_to_values = {}
        for output_name in output_names:
            values = dataset.get_view(
                group_names=dataset.output_group, variable_names=output_name
            ).to_numpy()
            if values.shape[1] > 1:
                msg = (
                    f"The event output {output_name!r} has {values.shape[1]} "
                    "components; ISFORMSobolAnalysis only supports event outputs "
                    "with a single component."
                )
                raise ValueError(msg)

            output_name_to_values[output_name] = values[:, 0]

        return output_name_to_values

    @staticmethod
    def __create_dataset(
        event_names: list[str],
        standard_samples_per_event: list[RealArray],
        indicator_per_event: list[RealArray],
        variable_names: list[str],
        variable_sizes: Mapping[str, int],
    ) -> IODataset:
        """Create the dataset of the standard samples and indicators of all the events.

        As each event has its own design,
        the per-event input samples are stacked vertically into a single input group
        and each event has its own output column,
        non-zero only on the rows of its design.
        The row range of each event is stored in `dataset.misc["event_slices"]`.

        Args:
            event_names: The names of the events.
            standard_samples_per_event: The input samples in the standard space
                of each event, each shaped as `(n_samples_e, input_dimension)`.
            indicator_per_event: The indicator of each event,
                each shaped as `(n_samples_e, 1)`.
            variable_names: The names of the random variables.
            variable_sizes: The sizes of the random variables.

        Returns:
            The dataset of the standard samples and indicators.
        """
        input_data = vstack(standard_samples_per_event)
        n_rows = len(input_data)
        output_data = zeros((n_rows, len(event_names)))
        event_slices = {}
        start = 0
        for column, (event_name, indicator) in enumerate(
            zip(event_names, indicator_per_event, strict=True)
        ):
            stop = start + len(indicator)
            output_data[start:stop, column] = indicator[:, 0]
            event_slices[event_name] = (start, stop)
            start = stop

        dataset = IODataset()
        dataset.add_input_group(
            input_data,
            variable_names=variable_names,
            variable_name_to_n_components={
                name: variable_sizes[name] for name in variable_names
            },
        )
        dataset.add_output_group(
            output_data,
            variable_names=event_names,
            variable_name_to_n_components=dict.fromkeys(event_names, 1),
        )
        dataset.misc["event_slices"] = event_slices
        return dataset

    def compute_indices(
        self,
        output_names: str | Iterable[str] = (),
        algo: ISFORMSobolAnalysis.Algorithm | None = None,
        confidence_level: float = 0.95,
        use_asymptotic_distributions: bool = True,
        n_replicates: int = 100,
        seed: int | None = seed,
    ) -> SobolIndicesEstimatorMixin.SensitivityIndices:
        """
        Args:
            algo: The name of the algorithm
                to estimate the Sobol' indices from the importance-sampling samples.
                All the algorithms assume a pick-and-freeze design,
                except `Rank`, which assumes independent samples.
                If `None`,
                use `Saltelli` or `Rank` according to the design.
            confidence_level: The confidence level
                of the confidence intervals associated with the estimates.
            use_asymptotic_distributions: Whether to compute the confidence intervals
                using the asymptotic distributions; otherwise, use the bootstrap method.
                When the algorithm is `Algorithm.RANK` (or `"Rank"`),
                the confidence intervals can only be estimated via bootstrap,
                and so, this argument is ignored.
            n_replicates: The number of bootstrap replicates
                used for the computation of the confidence intervals.
            seed: The seed of the bootstrap.
                If `None`, the bootstrap intervals are not reproducible.

        Raises:
            ValueError: If `Rank` is used with a pick-and-freeze design,
                if another algorithm is used with non-pick-and-freeze samples
                or if the bootstrap is used with less than one replicate.
        """  # noqa: D205, D212, D415
        dataset = self.dataset
        use_pick_and_freeze = dataset.misc.get("use_pick_and_freeze", False)
        algo = self._select_sobol_algorithm(algo, use_pick_and_freeze)
        output_names = self._get_output_names(output_names)
        sample_size_per_event = dataset.misc["sample_size"]
        design_point_per_event = dataset.misc["design_point"]
        compute_second_order = dataset.misc.get("eval_second_order", False)
        event_slices = dataset.misc["event_slices"]
        # Each event has its own design stored in its own row range,
        # so the input and output samples are sliced event by event.
        all_input_data = dataset.get_view(
            group_names=dataset.input_group, variable_names=self._input_names
        ).to_numpy()
        self._output_name_to_estimators = {}
        for output_name in output_names:
            estimators = self._output_name_to_estimators.setdefault(output_name, [])
            start, stop = event_slices[output_name]
            indicator = dataset.get_view(
                group_names=dataset.output_group, variable_names=output_name
            ).to_numpy()[start:stop, 0]
            # The estimators divide by the variance of the indicator
            # over the rows of the A block only
            # (over all the rows for the rank-based estimator,
            # whose sample size is the number of rows),
            # so a design whose failing rows all sit in the other blocks
            # would make every index NaN: skip the event.
            # This guard does not make every index defined:
            # the Martinez estimator also divides by the variances
            # of the other blocks, see the warnings below.
            sample_size = sample_size_per_event[output_name]
            if indicator[:sample_size].var() == 0.0:
                estimators.append(None)
                continue

            estimator = ISSobolIndicesEstimator(
                algo,
                all_input_data[start:stop],
                indicator,
                design_point_per_event[output_name],
                sample_size,
                compute_second_order,
                confidence_level,
                use_asymptotic_distributions,
                n_replicates,
                seed,
            )
            self.__warn_about_undefined_indices(
                output_name, algo, estimator, n_replicates
            )
            estimators.append(estimator)

        # Use the variance of each estimator, i.e. the very quantity
        # its indices were divided by, rather than the probability
        # estimated over all the rows of the design.
        self._set_output_variances({
            event_name: array([
                0.0 if estimator is None else estimator.variance
                for estimator in event_estimators
            ])
            for event_name, event_estimators in self._output_name_to_estimators.items()
        })

        if algo == self.Algorithm.RANK:
            self._indices = self.SensitivityIndices(
                first=self._get_first_order_indices()
            )
        else:
            self._indices = self.SensitivityIndices(
                first=self._get_first_order_indices(),
                second=self._get_second_order_indices(),
                total=self._get_total_order_indices(),
            )

        return self._indices

    @staticmethod
    def __warn_about_undefined_indices(
        output_name: str,
        algo: ISFORMSobolAnalysis.Algorithm,
        estimator: ISSobolIndicesEstimator,
        n_replicates: int,
    ) -> None:
        """Log a warning when an estimator has undefined indices or replicates.

        Args:
            output_name: The name of the event.
            algo: The Sobol' estimation algorithm.
            estimator: The estimator of the event.
            n_replicates: The number of bootstrap replicates.
        """
        if (
            isnan(estimator.first_order_indices).any()
            or isnan(estimator.total_order_indices).any()
        ):
            logger.warning(
                "Some Sobol' indices of the event %r are undefined "
                "with the %s algorithm: "
                "the indicator is constant over a block of the design "
                "that the estimator divides by; "
                "they are set to NaN.",
                output_name,
                algo,
            )

        if estimator.n_degenerate_replicates > 0:
            logger.warning(
                "%s of the %s bootstrap replicates of the event %r "
                "were discarded with the %s algorithm "
                "because the indicator was constant over a resampled block; "
                "the confidence intervals use the other replicates.",
                estimator.n_degenerate_replicates,
                n_replicates,
                output_name,
                algo,
            )

    def _get_plot_title(self, output_name: str, output_component: int) -> str:
        """Return the default plot title for an event.

        Args:
            output_name: The name of the event.
            output_component: The component of the event (always 0, as an event is
                scalar).

        Returns:
            The default plot title.
        """
        return f"Sobol' indices for the event {output_name!r}"

    def _get_plot_subtitle(self, output_name: str, output_component: int) -> str:
        """Return the plot subtitle for an event.

        The subtitle displays the estimated probability of the event.

        Args:
            output_name: The name of the event.
            output_component: The component of the event (always 0, as an event is
                scalar).

        Returns:
            The plot subtitle.
        """
        probability = self.dataset.misc["probability"][output_name]
        return f"P={probability:.1e}"
