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
"""A factory to instantiate linear solvers from their class names."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.algos.base_algo_factory import BaseAlgoFactory
from gemseo.algos.linear_solvers.linear_solver_library import LinearSolverLibrary

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.algos.linear_solvers.linear_problem import LinearProblem


class LinearSolversFactory(BaseAlgoFactory):
    """MDA factory to create the MDA from a name or a class."""

    _CLASS = LinearSolverLibrary
    _MODULE_NAMES = ("gemseo.algos.linear_solvers",)

    @property
    def linear_solvers(self) -> list[str]:
        """The names of the available classes."""
        return self._factory.class_names

    def execute(  # noqa:D102
        self,
        problem: LinearProblem,
        algo_name: str,
        **options: Any,
    ) -> ndarray:
        return super().execute(problem, algo_name, **options)
