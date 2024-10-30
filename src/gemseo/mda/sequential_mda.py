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
# Copyright 2024 Capgemini
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A chain of MDAs to build hybrids of MDA algorithms sequentially."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.core.execution_status import ExecutionStatus
from gemseo.mda.base_mda import BaseMDA
from gemseo.mda.sequential_mda_settings import MDASequentialSettings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline


class MDASequential(BaseMDA):
    """A sequence of elementary MDAs."""

    Settings: ClassVar[type[MDASequentialSettings]] = MDASequentialSettings
    """The pydantic model for the settings."""

    settings: MDASequentialSettings
    """The settings of the MDA"""

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        mda_sequence: Sequence[BaseMDA],
        settings_model: MDASequentialSettings | None = None,
        **settings: Any,
    ) -> None:
        """
        Args:
            mda_sequence: The sequence of MDAs.
        """  # noqa:D205 D212 D415
        super().__init__(disciplines, settings_model=settings_model, **settings)
        self._compute_input_coupling_names()
        self._init_mda_sequence(mda_sequence)

    def _init_mda_sequence(self, mda_sequence: Sequence[BaseMDA]) -> None:
        """Initialize the MDA sequence.

        Args:
           mda_sequence: The sequence of MDAs to chain.
        """
        self.mda_sequence = mda_sequence
        self.settings._sub_mdas = self.mda_sequence

        log_convergence = self.settings.log_convergence
        for mda in self.mda_sequence:
            mda.reset_history_each_run = True
            log_convergence = log_convergence or mda.settings.log_convergence

    @BaseMDA.scaling.setter
    def scaling(self, scaling: BaseMDA.ResidualScaling) -> None:  # noqa: D102
        self._scaling = scaling
        for mda in self.mda_sequence:
            mda.scaling = scaling

    def _run(self) -> None:
        super()._run()

        if self.reset_history_each_run:
            self.residual_history = []

        # Execute the MDAs in sequence
        for mda in self.mda_sequence:
            mda.execution_status.value = ExecutionStatus.Status.PENDING

            # Execute the i-th MDA
            self.io.data = mda.execute(self.io.data)

            # Extend the residual history
            self.residual_history += mda.residual_history

            if mda.normed_residual < self.settings.tolerance:
                break
