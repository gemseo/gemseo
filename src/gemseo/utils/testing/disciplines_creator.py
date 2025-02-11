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
"""Provide functions to build disciplines for tests."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import TYPE_CHECKING

from numpy import ones

from gemseo.utils.discipline import DummyDiscipline

if TYPE_CHECKING:
    from gemseo.core.discipline import Discipline


def create_disciplines_from_desc(
    disc_desc: Mapping[str, Iterable[Iterable[str]]] | Sequence[Discipline],
) -> list[Discipline]:
    """Return the disciplines from their descriptions.

    Args:
        disc_desc: The description of the disciplines.
    """
    if isinstance(disc_desc, Sequence):
        # These are disciplines classes.
        return [cls() for cls in disc_desc]

    disciplines = []
    data = ones(1)

    for name, io_names in disc_desc.items():
        disc = DummyDiscipline(name)
        input_d = dict.fromkeys(io_names[0], data)
        disc.io.input_grammar.update_from_data(input_d)
        output_d = dict.fromkeys(io_names[1], data)
        disc.io.output_grammar.update_from_data(output_d)
        disciplines += [disc]

    return disciplines
