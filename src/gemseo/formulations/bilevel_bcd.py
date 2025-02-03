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
from typing import ClassVar

from gemseo.core.chains.warm_started_chain import MDOWarmStartedChain
from gemseo.formulations.bilevel import BiLevel
from gemseo.formulations.bilevel_bcd_settings import BiLevel_BCD_Settings
from gemseo.mda.factory import MDAFactory

if TYPE_CHECKING:
    from gemseo.core.chains.chain import MDOChain


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

    _settings: BiLevel_BCD_Settings

    __mda_factory: ClassVar[MDAFactory] = MDAFactory()
    """The MDA factory."""

    def _create_multidisciplinary_chain(self) -> MDOChain:
        """Build the chain on top of which all functions are built.

        This chain is: MDA -> MDAGaussSeidel(MDOScenarios) -> MDA.

        Returns:
            The multidisciplinary chain.
        """
        # Build the scenario adapters to be chained with MDAs
        self._build_scenario_adapters(
            reset_x0_before_opt=self._settings.reset_x0_before_opt,
            keep_opt_history=True,
        )
        chain_disc, sub_opts = self._build_chain_dis_sub_opts()

        bcd_mda = self.__mda_factory.create(
            "MDAGaussSeidel",
            sub_opts,
            settings_model=self._settings.bcd_mda_settings,
        )

        chain_disc += [bcd_mda]
        if self._mda2:
            chain_disc += [self._mda2]

        return MDOWarmStartedChain(
            chain_disc,
            name=self.CHAIN_NAME,
            variable_names_to_warm_start=self._get_variable_names_to_warm_start(),
        )
