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
"""Base class for reliability-oriented sensitivity analysis."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo.uncertainty.reliability.event_variable import EventVariable
from gemseo.uncertainty.sensitivity.base import BaseGenericSensitivityAnalysis
from gemseo.uncertainty.sensitivity.base import T

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Mapping

    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline import Discipline
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.formulations.base_settings import BaseFormulationSettings
    from gemseo.uncertainty.reliability.base_settings import (
        BaseReliabilityAlgorithmSettings,
    )
    from gemseo.uncertainty.reliability.event import Event


class BaseROSensitivityAnalysis(BaseGenericSensitivityAnalysis[T]):
    """Base class for reliability-oriented sensitivity analysis.

    The aim of a reliability-oriented sensitivity analysis (ROSA)
    is to qualify or quantify how the uncertain inputs of a model impact binary events
    associated with disciplinary outputs
    (e.g. a disciplinary output exceeding a threshold).

    !!! note "The outputs are events"
        In this class and its subclasses,
        the *outputs* handled by the inherited API
        (e.g. the keys of
        [indices][gemseo.uncertainty.sensitivity.base.BaseGenericSensitivityAnalysis.indices],
        the `output`/`outputs` arguments of the plotting methods
        and the `output_names` argument of `compute_indices()`)
        are *events*, not disciplinary outputs.
        The notion of *output* is therefore equivalent to the notion of *event*:
        an output name is an event name
        and each event is scalar (it has a single component).
    """

    @staticmethod
    def get_event_variables(
        *names: str,
    ) -> EventVariable | tuple[EventVariable, ...]:
        """Return event variables from variable names.

        E.g. `y = analysis.get_event_variables("y")`
        then `y > 3.0` is the event `"y > 3.0"`.

        Args:
            *names: The names of the variables of interest.

        Returns:
            The event variables.
        """
        return EventVariable.from_names(*names)

    @abstractmethod
    def compute_samples(
        self,
        disciplines: Collection[Discipline],
        parameter_space: ParameterSpace,
        events: Mapping[str, Event],
        algo_settings: BaseReliabilityAlgorithmSettings | None = None,
        formulation_settings: BaseFormulationSettings | None = None,
    ) -> IODataset:
        """
        Args:
            events: The events of interest,
                indexed by their names,
                e.g. `{"y_high": analysis.get_event_variables("y") > 3.0}`.
            algo_settings: The settings of the algorithm to generate the samples.
                If `None`, use default settings.
            formulation_settings: The settings of the MDO formulation.
                If `None`,
                use the default settings of the MDF formulation.
        """  # noqa: D205, D212

    def _get_output_n_components(self, name: str) -> int:
        # An event is scalar.
        return 1
