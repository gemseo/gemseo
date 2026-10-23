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
"""Elementary event."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import Field

from gemseo.core.function.array_function import ArrayFunction
from gemseo.uncertainty.reliability.threshold_comparator import ThresholdComparator

if TYPE_CHECKING:
    from gemseo.util.typing import BooleanArray
    from gemseo.util.typing import RealArray


class ElementaryEvent(
    BaseModel,
    frozen=True,
    arbitrary_types_allowed=True,
    extra="forbid",
):
    """An immutable elementary event defined by a single threshold comparison."""

    name: str = Field(description="The name of the variable of interest.")

    threshold: float = Field(
        0.0, description="The threshold compared with the variable of interest."
    )

    comparator: ThresholdComparator = Field(
        ThresholdComparator.GREATER,
        description=(
            "The comparator between the variable of interest and the threshold."
        ),
    )

    function: ArrayFunction | None = Field(
        None, description="The function evaluating the variable of interest, if known."
    )

    def __invert__(self) -> ElementaryEvent:
        return self.model_validate({
            **dict(self),
            "comparator": self.comparator.complement,
        })

    def bind_function(self, function: ArrayFunction) -> ElementaryEvent:
        """Return a copy of this elementary event bound to a function.

        Args:
            function: The function evaluating the variable of interest.

        Returns:
            The copy of this elementary event, bound to the function.
        """
        return self.model_validate({**dict(self), "function": function})

    def evaluate(self, values: RealArray) -> BooleanArray:
        """Compute the boolean indicator of this elementary event.

        Args:
            values: The values of the variable of interest.

        Returns:
            The boolean indicator of this elementary event, element-wise.
        """
        return self.comparator.compare(values, self.threshold)

    def __str__(self) -> str:
        return f"{self.name} {self.comparator.value} {self.threshold}"
