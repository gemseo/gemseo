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
"""Categorical variables."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from pydantic import Field
from pydantic import model_validator

from gemseo.enum import VariableType
from gemseo.space.variable._formatting import format_elided
from gemseo.space.variable.base_categorical import BaseCategoricalVariable
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from typing import Self

_max_displayed_labels: Final[int] = 6
"""The number of labels displayed in an error message."""


class CategoricalVariable(BaseCategoricalVariable):
    """A scalar categorical variable defined by its labels.

    The labels are the only input of the variable;
    they are strings, without duplication, in the order of your choice.
    """

    type: ClassVar[VariableType] = VariableType.CATEGORICAL

    labels: tuple[str, ...] = Field(
        description="""The labels that the variable can take.

    The position of a label is its index in this tuple.
    """
    )

    @model_validator(mode="after")
    def __validate_categorical_variable(self) -> Self:
        """Validate the variable.

        Returns:
            The instance.

        Raises:
            ValueError: If there is no label
                or if a label appears several times.
        """
        labels = self.labels
        if not labels:
            msg = "A categorical variable must have at least one label."
            raise ValueError(msg)

        seen = set()
        duplicates = set()
        for label in labels:
            if label in seen:
                duplicates.add(label)

            seen.add(label)
        if duplicates:
            plural = len(duplicates) > 1
            msg = (
                f"The following label{'s appear' if plural else ' appears'} "
                f"several times: {pretty_repr(duplicates)}."
            )
            raise ValueError(msg)

        return self

    def _format_labels(self) -> str:
        return format_elided(self.labels, "labels", _max_displayed_labels, repr)
