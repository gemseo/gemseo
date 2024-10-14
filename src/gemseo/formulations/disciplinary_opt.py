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

from gemseo.core.chains.chain import MDOChain
from gemseo.disciplines.utils import get_all_inputs
from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.discipline import Discipline


class DisciplinaryOpt(BaseMDOFormulation):
    """The disciplinary optimization.

    This formulation draws the architecture of a mono-disciplinary optimization process
    from an ordered list of disciplines, an objective function and a design space.
    """

    def __init__(  # noqa:D107
        self,
        disciplines: list[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        maximize_objective: bool = False,
        differentiated_input_names_substitute: Iterable[str] = (),
    ) -> None:
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            maximize_objective=maximize_objective,
            differentiated_input_names_substitute=differentiated_input_names_substitute,
        )
        if len(disciplines) > 1:
            self.__chain = MDOChain(disciplines)
        else:
            self.__chain = disciplines[0]
        self._filter_design_space()
        self._set_default_input_values_from_design_space()
        self._build_objective_from_disc(objective_name)

    def get_top_level_disciplines(self) -> tuple[Discipline, ...]:  # noqa:D102
        return (self.__chain,)

    def _filter_design_space(self) -> None:
        """Filter the design space to keep only available variables."""
        all_inpts = get_all_inputs(self.get_top_level_disciplines())
        kept = set(all_inpts).intersection(self.design_space)
        self.design_space.filter(kept)
