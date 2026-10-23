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
from typing import ClassVar
from typing import TypeVar

from gemseo.discipline.chain.chain import DisciplineChain
from gemseo.formulation.core.base_mdo import BaseMDOFormulation
from gemseo.formulation.disciplinary_opt_settings import DisciplinaryOpt_Settings
from gemseo.util.discipline import get_all_inputs

if TYPE_CHECKING:
    from gemseo.core.discipline import Discipline
    from gemseo.space.base import BaseVariableSpace

_SpaceT = TypeVar("_SpaceT", bound="BaseVariableSpace")


class DisciplinaryOpt(BaseMDOFormulation[DisciplinaryOpt_Settings, _SpaceT]):
    """The disciplinary optimization.

    This formulation draws the architecture of a mono-disciplinary optimization process
    from an ordered list of disciplines, an objective function and a design space.
    """

    settings_class: ClassVar[type[DisciplinaryOpt_Settings]] = DisciplinaryOpt_Settings

    __top_level_disciplines: tuple[Discipline]
    """The top-level disciplines."""

    def _create_multidisciplinary_process(self) -> None:
        disciplines = self.disciplines
        self.__top_level_disciplines = (
            DisciplineChain(disciplines) if len(disciplines) > 1 else disciplines[0],
        )

    def get_top_level_disciplines(  # noqa:D102
        self, include_sub_formulations: bool = False
    ) -> tuple[Discipline]:
        return self.__top_level_disciplines

    def _update_input_space(self) -> None:
        all_input_names = get_all_inputs(self.get_top_level_disciplines())
        design_space = self.problem.input_space
        kept_variable_names = set(all_input_names).intersection(design_space)
        design_space.filter(kept_variable_names)
        self._set_default_input_values_from_space()
