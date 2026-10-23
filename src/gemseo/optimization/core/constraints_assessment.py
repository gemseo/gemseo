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
"""The assessment of the constraints at a point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gemseo.optimization.core.constraint_status import ConstraintStatus
from gemseo.util.read_only_mapping import ReadOnlyMapping

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.optimization.core.constraint_assessment import ConstraintAssessment


@dataclass(frozen=True)
class ConstraintsAssessment:
    """The assessment of the constraints at a point."""

    name_to_assessment: Mapping[str, ConstraintAssessment]
    """The assessments of the constraints, by constraint name.

    It is read-only.
    """

    def __post_init__(self) -> None:
        # A read-only copy keeps the frozen assessment from being modified,
        # and unlike a `MappingProxyType`, it can be pickled.
        object.__setattr__(
            self, "name_to_assessment", ReadOnlyMapping(dict(self.name_to_assessment))
        )

    @property
    def are_satisfied(self) -> bool:
        """Whether all the constraints are satisfied."""
        return all(
            assessment.status is ConstraintStatus.SATISFIED
            for assessment in self.name_to_assessment.values()
        )

    @property
    def squared_violation(self) -> float:
        """The sum of the squared violations of the constraints.

        See
        [ConstraintAssessment.squared_violation][gemseo.optimization.core.constraint_assessment.ConstraintAssessment.squared_violation]
        for the term of a constraint.
        It is `inf` when a value is NaN or missing.
        """
        return sum(
            (
                assessment.squared_violation
                for assessment in self.name_to_assessment.values()
            ),
            0.0,
        )

    @property
    def max_violation(self) -> float:
        """The largest maximum violation of the constraints.

        See
        [ConstraintAssessment.max_violation][gemseo.optimization.core.constraint_assessment.ConstraintAssessment.max_violation]
        for the maximum violation of a constraint.
        It is `inf` when a value is NaN or missing,
        and `0.0` without constraint.
        """
        return max(
            (
                assessment.max_violation
                for assessment in self.name_to_assessment.values()
            ),
            default=0.0,
        )

    @property
    def n_violated_components(self) -> int:
        """The total number of violated components of the constraints.

        See
        [ConstraintAssessment.n_violated_components][gemseo.optimization.core.constraint_assessment.ConstraintAssessment.n_violated_components]
        for the number of violated components of a constraint.
        A missing constraint is ignored; a NaN component is included.
        """
        return sum(
            assessment.n_violated_components
            for assessment in self.name_to_assessment.values()
        )

    @property
    def missing_names(self) -> tuple[str, ...]:
        """The names of the constraints whose values are missing."""
        return tuple(
            name
            for name, assessment in self.name_to_assessment.items()
            if assessment.status is ConstraintStatus.MISSING
        )
