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
"""A factory of linear solver libraries."""

from __future__ import annotations

from typing import Final

from gemseo.core.algorithm.base_algorithm_factory import BaseAlgorithmFactory
from gemseo.linear.core.base_linear_solver_library import BaseLinearSolverLibrary


class LinearSolverLibraryFactory(BaseAlgorithmFactory):
    """A factory of linear solver libraries."""

    _CLASS = BaseLinearSolverLibrary
    _PACKAGE_NAMES = ("gemseo.linear",)

    @property
    def linear_solvers(self) -> list[str]:
        """The names of the available classes."""
        return self._factory.class_names


LINEAR_SOLVER_LIBRARY_FACTORY: Final[LinearSolverLibraryFactory] = (
    LinearSolverLibraryFactory()
)
"""The factory of `BaseLinearSolverLibrary` objects."""
