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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The base discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.discipline import MDODiscipline

if TYPE_CHECKING:
    from gemseo.problems.scalable.parametric.core.disciplines.base_discipline import (
        BaseDiscipline as _BaseDiscipline,
    )


class BaseDiscipline(MDODiscipline):
    """Base class for the disciplines of the scalable problem.

    This :class:`.MDODiscipline` relies on a |g|-free core discipline.
    """

    _CORE_DISCIPLINE_CLASS: type[_BaseDiscipline]
    """The class of the core discipline."""

    _discipline: _CORE_DISCIPLINE_CLASS  # noqa: F821
    """The core discipline."""

    def __init__(
        self,
        *core_discipline_parameters: Any,
        **default_input_values: Any,
    ) -> None:
        """
        Args:
            *core_discipline_parameters: The parameters
                to instantiate the core discipline
                as ``CoreDiscipline(*core_discipline_parameters)``.
            **default_input_values: The default values of the input variables.
        """  # noqa: D205 D212
        self._discipline = self._CORE_DISCIPLINE_CLASS(
            *core_discipline_parameters, **default_input_values
        )
        super().__init__(self._discipline.name)
        self.input_grammar.update_from_names(self._discipline.input_names)
        self.output_grammar.update_from_names(self._discipline.output_names)
        self.default_inputs = default_input_values
