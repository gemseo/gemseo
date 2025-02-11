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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A Bi-level formulation using the block coordinate descent algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.formulations.bilevel import BiLevel
from gemseo.formulations.bilevel_bcd_settings import BiLevel_BCD_Settings
from gemseo.mda.factory import MDAFactory

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
    from typing import ClassVar

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.discipline import Discipline
    from gemseo.mda.gauss_seidel import MDAGaussSeidel


class BiLevelBCD(BiLevel):
    """Block Coordinate Descent bi-level formulation.

    This formulation draws an optimization architecture
    that involves multiple optimization problems to be solved
    in order to obtain the solution of the MDO problem.

    At each iteration on the global design variables,
    the bi-level-BCD MDO formulation implementation performs:

    1. a first MDA to compute the coupling variables,
    2. a loop on several disciplinary optimizations on the
       local design variables using an :class:`.MDAGaussSeidel`,
    3. an MDA to compute precisely the system optimization criteria.
    """

    Settings: ClassVar[type[BiLevel_BCD_Settings]] = BiLevel_BCD_Settings

    _bcd_mda: MDAGaussSeidel
    """The MDA of the BCD algorithm."""

    _settings: BiLevel_BCD_Settings

    __mda_factory: ClassVar[MDAFactory] = MDAFactory()
    """The MDA factory."""

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        settings_model: BiLevel_BCD_Settings | None = None,
        **settings: Any,
    ) -> None:
        self._bcd_mda = None
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            settings_model=settings_model,
            **settings,
        )

    @property
    def bcd_mda(self) -> MDAGaussSeidel:
        """The MDA of the BCD algorithm."""
        return self._bcd_mda

    def _create_sub_scenarios_chain(self) -> MDAGaussSeidel:
        self._bcd_mda = self.__mda_factory.create(
            "MDAGaussSeidel",
            self.scenario_adapters,
            settings_model=self._settings.bcd_mda_settings,
        )
        return self._bcd_mda
