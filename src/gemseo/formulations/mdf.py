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
"""The Multi-disciplinary Design Feasible (MDF) formulation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation
from gemseo.formulations.mdf_settings import MDF_Settings
from gemseo.mda.factory import MDAFactory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.discipline import Discipline
    from gemseo.core.grammars.json_grammar import JSONGrammar
    from gemseo.mda.base_mda import BaseMDA
    from gemseo.typing import StrKeyMapping


class MDF(BaseMDOFormulation):
    """The Multidisciplinary Design Feasible (MDF) formulation.

    This formulation draws an optimization architecture
    where:

    - the coupling of strongly coupled disciplines is made consistent
      by means of a Multidisciplinary Design Analysis (MDA),
    - the optimization problem
      with respect to the local and global design variables is made at the top level.

    Note that the multidisciplinary analysis is made at each optimization iteration.
    """

    mda: BaseMDA
    """The MDA used in the formulation."""

    Settings: ClassVar[type[MDF_Settings]] = MDF_Settings

    _settings: MDF_Settings

    __mda_factory: ClassVar[MDAFactory] = MDAFactory()
    """The MDA factory."""

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        settings_model: MDF_Settings | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            settings_model=settings_model,
            **settings,
        )
        self.mda = self.__mda_factory.create(
            self._settings.main_mda_name,
            self.disciplines,
            settings_model=self._settings.main_mda_settings,
        )
        self._update_design_space()
        self._build_objective_from_disc(objective_name, discipline=self.mda)

    def get_top_level_disciplines(self) -> tuple[Discipline, ...]:  # noqa:D102
        return (self.mda,)

    @classmethod
    def get_sub_options_grammar(cls, **options: str) -> JSONGrammar:  # noqa:D102
        return cls.__mda_factory.get_options_grammar(cls.__check_mda(**options))

    @classmethod
    def get_default_sub_option_values(cls, **options: str) -> StrKeyMapping:  # noqa:D102
        return cls.__mda_factory.get_default_option_values(cls.__check_mda(**options))

    @staticmethod
    def __check_mda(**options: str) -> str:
        """Check that main_mda_name is available.

        Args:
            options: The options.

        Returns:
            The main MDA.

        Raises:
            ValueError: When main_mda_name is not available.
        """
        main_mda_name = options.get("main_mda_name")
        if main_mda_name is None:
            msg = "main_mda_name option required to deduce the sub options of MDF."
            raise ValueError(msg)
        return main_mda_name

    def _update_design_space(self) -> None:
        """Update the design space by removing the coupling variables."""
        self._set_default_input_values_from_design_space()
        # No couplings in design space (managed by MDA)
        self._remove_couplings_from_ds()
        # Cleanup
        self._remove_unused_variables()

    def _remove_couplings_from_ds(self) -> None:
        """Remove the coupling variables from the design space."""
        design_space = self.optimization_problem.design_space
        for coupling in self.mda.coupling_structure.all_couplings:
            if coupling in design_space:
                design_space.remove_variable(coupling)
