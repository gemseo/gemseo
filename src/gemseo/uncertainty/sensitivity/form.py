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
"""Sensitivity analysis based on the first-order reliability method (FORM)."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import auto
from typing import TYPE_CHECKING
from typing import ClassVar

from strenum import LowercaseStrEnum

from gemseo.uncertainty.reliability.openturns.form_settings import OT_FORM_Settings
from gemseo.uncertainty.reliability.scenario import ReliabilityScenario
from gemseo.uncertainty.sensitivity.base_ro import BaseROSensitivityAnalysis

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Mapping
    from pathlib import Path

    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline.discipline import Discipline
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.formulations.base_settings import BaseFormulationSettings
    from gemseo.uncertainty.reliability.event import Event
    from gemseo.uncertainty.sensitivity.base import FirstOrderIndicesType


class FORMAnalysisMethod(LowercaseStrEnum):
    """A FORM analysis method."""

    CLASSICAL = auto()
    """The squares of the co-factors of the design point in the physical space."""

    ELLIPTICAL = auto()
    """The squares of the co-factors of the design point in the standard space."""

    PHYSICAL = auto()
    """The squares of the physical sensitivities of the reliability index."""


class FORMAnalysis(BaseROSensitivityAnalysis[FORMAnalysisMethod]):
    r"""Sensitivity analysis based on the first-order reliability method (FORM).

    The first-order reliability method (FORM) estimates the probability of an event,
    e.g. an output of interest exceeding a threshold,
    by searching for the most probable failure point (MPFP)
    and approximating the limit-state surface at this point by a hyperplane.

    As a by-product,
    FORM provides importance factors
    quantifying the contribution of each input variable to the event.
    These importance factors can be interpreted as sensitivity indices
    with respect to the events rather than to the raw outputs of interest.

    Three types of importance factors are available
    (see
    [ImportanceFactors][gemseo.uncertainty.reliability.openturns.form_result.ImportanceFactors]):

    - `classical`: the squares of the co-factors of the design point
      in the physical space,
    - `elliptical`: the squares of the co-factors of the design point
      in the standard space,
    - `physical`: the squares of the physical sensitivities,
      i.e. the partial derivatives of the Hasofer-Lind reliability index
      with respect to the inputs in the physical space.
    """

    @dataclass(frozen=True)
    class SensitivityIndices:  # noqa: D106
        classical: FirstOrderIndicesType = field(default_factory=dict)
        """The classical importance factors."""

        elliptical: FirstOrderIndicesType = field(default_factory=dict)
        """The elliptical importance factors."""

        physical: FirstOrderIndicesType = field(default_factory=dict)
        """The physical importance factors."""

    _indices: SensitivityIndices

    _DEFAULT_MAIN_METHOD: ClassVar[FORMAnalysisMethod] = FORMAnalysisMethod.CLASSICAL

    def __init__(self, samples: IODataset | str | Path = "") -> None:  # noqa: D107
        super().__init__(samples)
        if self.dataset is not None:
            self._output_names = list(self.dataset.misc["execution_result"])

    def compute_samples(
        self,
        disciplines: Collection[Discipline],
        parameter_space: ParameterSpace,
        events: Mapping[str, Event],
        algo_settings: OT_FORM_Settings | None = None,
        formulation_settings: BaseFormulationSettings | None = None,
    ) -> IODataset:
        """Run the FORM/SORM analysis and store the model evaluations.

        Unlike a sampling-based sensitivity analysis,
        FORM/SORM is an optimization method
        searching the standard normal space
        for the most probable failure point (MPFP).
        The returned dataset therefore contains
        the model evaluations performed during this optimization
        (the optimizer iterates),
        not points drawn from a sampling of the uncertain space.
        The importance factors are computed by
        [compute_indices][gemseo.uncertainty.sensitivity.form.FORMAnalysis.compute_indices]
        from the reliability result stored in `dataset.misc["execution_result"]`.

        Args:
            algo_settings: The settings
                of the first-order or second-order reliability method (FORM/SORM).
                If `None`, use the default FORM settings.

        Returns:
            The model evaluations performed during the FORM/SORM optimization.
        """
        settings = algo_settings or OT_FORM_Settings()
        if not settings.use_database:
            settings = settings.model_copy(update={"use_database": True})

        scenario = ReliabilityScenario(
            list(disciplines),
            parameter_space,
            formulation_settings=formulation_settings,
        )
        for event_name, event in events.items():
            scenario.add_event(event, event_name)

        scenario.execute(settings)
        self.dataset = scenario.to_dataset()
        self._input_names = parameter_space.variable_names
        self._output_names = list(events)
        return self.dataset

    def compute_indices(  # noqa: D102
        self, output_names: str | Iterable[str] = ()
    ) -> SensitivityIndices:
        output_names = list(self._get_output_names(output_names))
        results = self.dataset.misc["execution_result"]
        self._indices = self.SensitivityIndices(**{
            factor_type: {
                output_name: [
                    {
                        input_name: getattr(
                            results[output_name].importance_factors,
                            f"{factor_type}_as_dict",
                        )[input_name]
                        for input_name in self._input_names
                    }
                ]
                for output_name in output_names
            }
            for factor_type in ("classical", "elliptical", "physical")
        })

        return self._indices
