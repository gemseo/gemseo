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
from gemseo.formulations.bilevel_bcd_settings import BiLevelBCD_Settings
from gemseo.mda.factory import MDA_FACTORY

if TYPE_CHECKING:
    from typing import ClassVar

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
       local design variables using an
       [MDAGaussSeidel][gemseo.mda.gauss_seidel.MDAGaussSeidel],
    3. an MDA to compute precisely the system optimization criteria.
    """

    settings_class: ClassVar[type[BiLevelBCD_Settings]] = BiLevelBCD_Settings

    _bcd_mda: MDAGaussSeidel
    """The MDA of the BCD algorithm."""

    _settings: BiLevelBCD_Settings

    @property
    def bcd_mda(self) -> MDAGaussSeidel:
        """The MDA of the BCD algorithm."""
        return self._bcd_mda

    def _create_sub_scenarios_chain(self) -> MDAGaussSeidel:
        self._bcd_mda = MDA_FACTORY.create(
            "MDAGaussSeidel",
            self.scenario_adapters,
            settings=self._settings.bcd_mda_settings,
        )
        return self._bcd_mda
