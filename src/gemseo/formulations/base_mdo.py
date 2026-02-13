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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The base class for all MDO formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.mdo_functions.collections.constraints import Constraints
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.formulations.base import BaseFormulation
from gemseo.formulations.base import T

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.core.discipline.discipline import Discipline


class BaseMDOFormulation(BaseFormulation[T]):
    """Base class for formulating an MDO problem."""

    def create_objective(  # noqa: D102
        self, output_names: Iterable[str], objective_name: str = ""
    ) -> MDOFunction:
        return self._create_function(output_names, name=objective_name)

    def add_observable(  # noqa: D102
        self,
        output_names: Iterable[str],
        observable_name: str = "",
        discipline: Discipline | None = None,
    ) -> None:
        function = self._create_function(
            output_names, discipline=discipline, name=observable_name
        )
        self.problem.add_observable(function)

    def create_constraint(  # noqa: D102
        self,
        output_names: Iterable[str],
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.EQ,
        constraint_name: str = "",
        value: float = 0,
        positive: bool = False,
        **kwargs: Any,
    ) -> MDOFunction:
        function = self._create_function(output_names, name=constraint_name)
        return Constraints.format(
            function, value=value, constraint_type=constraint_type, positive=positive
        )
