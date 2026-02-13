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
from __future__ import annotations

from typing import TYPE_CHECKING

from tests.formulations.not_mdo_formulations.formulation import NotMDOFormulation

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo import Discipline
    from gemseo.core.discipline.base_discipline import BaseDiscipline
    from gemseo.core.mdo_functions.mdo_function import MDOFunction


class ANotMDOFormulation(NotMDOFormulation):
    def _update_design_space(self) -> None: ...

    def _create_multidisciplinary_process(self) -> None: ...

    def create_objective(
        self, output_names: Iterable[str], objective_name: str = ""
    ) -> MDOFunction: ...

    def add_observable(
        self,
        output_names: Iterable[str],
        observable_name: str = "",
        discipline: Discipline | None = None,
    ) -> None:
        pass

    def create_constraint(
        self,
        output_names: Iterable[str],
        constraint_type: MDOFunction.ConstraintType,
        constraint_name: str,
        value: float,
        positive: bool,
    ) -> None:
        pass

    def get_top_level_disciplines(
        self, include_sub_formulations: bool = False
    ) -> tuple[BaseDiscipline, ...]:
        pass
