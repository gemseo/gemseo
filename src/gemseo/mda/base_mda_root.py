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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Charlie Vanaret, Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base module for Newton algorithm variants for solving MDAs."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.core.parallel_execution.disc_parallel_execution import DiscParallelExecution
from gemseo.core.parallel_execution.disc_parallel_linearization import (
    DiscParallelLinearization,
)
from gemseo.mda.base_mda_root_settings import BaseMDARootSettings
from gemseo.mda.base_mda_solver import BaseMDASolver
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline
    from gemseo.typing import StrKeyMapping


class BaseMDARoot(BaseMDASolver):
    """Abstract class implementing MDAs based on (Quasi-)Newton methods."""

    Settings: ClassVar[type[BaseMDARootSettings]] = BaseMDARootSettings
    """The pydantic model for the settings."""

    settings: BaseMDARootSettings
    """The settings of the MDA"""

    _parallel_execution: DiscParallelExecution | None
    """Either an executor of disciplines in parallel or ``None`` in serial mode."""

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        settings_model: BaseMDARootSettings | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(disciplines, settings_model=settings_model, **settings)
        if self.settings.n_processes > 1:
            self._execute_disciplines = self._execute_disciplines_in_parallel
            self._linearize_disciplines = self._linearize_disciplines_in_parallel
            self._parallel_execution = DiscParallelExecution(
                disciplines,
                self.settings.n_processes,
                use_threading=self.settings.use_threading,
            )
            self._parallel_linearization = DiscParallelLinearization(
                disciplines,
                self.settings.n_processes,
                use_threading=self.settings.use_threading,
                execute=self.settings.execute_before_linearizing,
            )
        else:
            self._execute_disciplines = self._execute_disciplines_sequentially
            self._linearize_disciplines = self._linearize_disciplines_sequentially
            self._parallel_execution = None
            self._parallel_linearization = False

        self._compute_input_coupling_names()
        self._set_resolved_variables(self.coupling_structure.strong_couplings)

    def _linearize_disciplines_sequentially(self, input_data: StrKeyMapping) -> None:
        """Linearize the disciplines sequentially.

        Args:
            input_data: The input data to execute the disciplines.
        """
        for discipline in self.disciplines:
            discipline.linearize(
                input_data, execute=self.settings.execute_before_linearizing
            )

    def _linearize_disciplines_in_parallel(
        self,
        input_data: StrKeyMapping,
    ) -> None:
        """Linearize the disciplines in parallel.

        Args:
            input_data: The input data to execute the disciplines.
        """
        self._parallel_linearization.execute([input_data] * len(self.disciplines))

    def _execute_disciplines_and_update_local_data(
        self, input_data: StrKeyMapping = READ_ONLY_EMPTY_DICT
    ) -> None:
        input_data = input_data or self.io.data
        self._execute_disciplines(input_data)
        for discipline in self.disciplines:
            self.io.data.update(discipline.get_output_data())

    def _execute_disciplines_in_parallel(self, input_data: StrKeyMapping) -> None:
        """Execute the discipline in parallel.

        Args:
            input_data: The input data of the disciplines.
        """
        self._parallel_execution.execute([input_data] * len(self.disciplines))

    def _execute_disciplines_sequentially(self, input_data: StrKeyMapping) -> None:
        """Execute the discipline sequentially.

        Args:
            input_data: The input data of the disciplines.
        """
        for discipline in self.disciplines:
            discipline.execute(input_data)
