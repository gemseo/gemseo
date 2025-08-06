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

"""Optimization metadata to be passed to an optimization dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.algos.constraint_tolerances import ConstraintTolerances


@dataclass(frozen=True)
class OptimizationMetadata:
    """The optimization metadata to be passed to the :class:`.OptimizationDataset`."""

    objective_name: str
    """The name of the objective."""

    standardized_objective_name: str
    """The name of the standardized objective."""

    minimize_objective: bool
    """Whether to minimize the objective."""

    # TODO: API: move to BasePostSettings
    use_standardized_objective: bool
    """Whether to use standardized objective for logging and post-processing.

    The standardized objective corresponds to the original one expressed as a cost
    function to minimize. A :class:`.BaseDriverLibrary` works with this standardized
    objective and the :class:`.Database` stores its values. However, for convenience, it
    may be more relevant to log the expression and the values of the original objective.
    """

    tolerances: ConstraintTolerances
    """The equality and inequality constraint tolerances."""

    output_names_to_constraint_names: Mapping[str, list[str]]
    """The mapping from the output names to the constraint names."""

    feasible_iterations: list[int]
    """The iterations of the feasible points."""

    optimum_iteration: int
    """The iteration of the optimum solution, if any."""
