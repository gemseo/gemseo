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
"""A base class for the disciplines of the customizable Sellar MDO problem."""

from typing import ClassVar

from gemseo.core.discipline import MDODiscipline
from gemseo.problems.mdo.sellar.utils import get_initial_data


class BaseSellar(MDODiscipline):
    """A base class for the disciplines of the customizable Sellar MDO problem."""

    _INPUT_NAMES: ClassVar[tuple[str]]
    """The names of the inputs."""

    _OUTPUT_NAMES: ClassVar[tuple[str]]
    """The names of the outputs."""

    def __init__(self, n: int = 1) -> None:
        """
        Args:
            n: The size of the local design variables and coupling variables.
        """  # noqa: D107 D205 D205 D212 D415
        super().__init__()
        default_inputs = get_initial_data(self._INPUT_NAMES, n)
        self.input_grammar.update_from_data(default_inputs)
        self.output_grammar.update_from_data(get_initial_data(self._OUTPUT_NAMES, n))
        self.default_inputs = default_inputs
