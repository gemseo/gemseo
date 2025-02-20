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
"""A formulation for uncoupled or weakly coupled problems."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.core.chains.chain import MDOChain
from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation
from gemseo.formulations.disciplinary_opt_settings import DisciplinaryOpt_Settings
from gemseo.utils.discipline import get_all_inputs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.discipline import Discipline


class DisciplinaryOpt(BaseMDOFormulation):
    """The disciplinary optimization.

    This formulation draws the architecture of a mono-disciplinary optimization process
    from an ordered list of disciplines, an objective function and a design space.
    """

    Settings: ClassVar[type[DisciplinaryOpt_Settings]] = DisciplinaryOpt_Settings

    __top_level_disciplines: tuple[Discipline]
    """The top-level disciplines."""

    def __init__(  # noqa:D107
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        settings_model: DisciplinaryOpt_Settings | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            settings_model=settings_model,
            **settings,
        )
        self.__top_level_disciplines = (
            MDOChain(disciplines) if len(disciplines) > 1 else disciplines[0],
        )
        self._filter_design_space()
        self._set_default_input_values_from_design_space()
        self._build_objective_from_disc(objective_name)

    def get_top_level_disciplines(self) -> tuple[Discipline]:  # noqa:D102
        return self.__top_level_disciplines

    def _filter_design_space(self) -> None:
        """Filter the design space to keep only available variables."""
        all_input_names = get_all_inputs(self.get_top_level_disciplines())
        design_space = self.optimization_problem.design_space
        kept_variable_names = set(all_input_names).intersection(design_space)
        design_space.filter(kept_variable_names)
