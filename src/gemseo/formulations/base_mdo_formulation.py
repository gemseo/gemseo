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

from numpy import zeros

from gemseo.core.mdo_functions.function_from_discipline import FunctionFromDiscipline
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.core.mdo_functions.taylor_polynomials import compute_linear_approximation
from gemseo.formulations.base_formulation import BaseFormulation
from gemseo.utils.string_tools import convert_strings_to_iterable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.discipline.discipline import Discipline


class BaseMDOFormulation(BaseFormulation):
    """A base class for MDO formulations."""

    def add_observable(  # noqa: D102
        self,
        output_names: str | Sequence[str],
        observable_name: str = "",
        discipline: Discipline | None = None,
    ) -> None:
        if isinstance(output_names, str):
            output_names = [output_names]
        obs_fun = FunctionFromDiscipline(output_names, self, discipline=discipline)
        if observable_name:
            obs_fun.name = observable_name
        self.optimization_problem.add_observable(obs_fun)

    def add_constraint(  # noqa: D102
        self,
        output_name: str | Sequence[str],
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.EQ,
        constraint_name: str = "",
        value: float = 0,
        positive: bool = False,
    ) -> None:
        output_names = convert_strings_to_iterable(output_name)
        constraint = FunctionFromDiscipline(output_names, self)
        if constraint.discipline_adapter.is_linear:
            constraint = compute_linear_approximation(
                constraint, zeros(constraint.discipline_adapter.input_dimension)
            )
        constraint.f_type = constraint_type
        if constraint_name:
            constraint.name = constraint_name
            constraint.has_default_name = False
        else:
            constraint.has_default_name = True
        self.optimization_problem.add_constraint(
            constraint, value=value, positive=positive
        )
