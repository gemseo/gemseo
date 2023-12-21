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

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


class BaseDiscipline(ABC):
    """The base discipline of the scalable problem."""

    name: str
    """The name of the discipline."""

    input_names_to_default_values: Mapping[str, NDArray[float]]
    """The default values of the input variables."""

    input_names: list[str]
    """The names of the input variables."""

    output_names: list[str]
    """The names of the output variables."""

    names_to_sizes: dict[str, int]
    """The sizes of the input and output variables."""
