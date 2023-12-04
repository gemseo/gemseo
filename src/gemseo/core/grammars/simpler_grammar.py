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
"""A simple grammar that does not check the data types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.grammars.simple_grammar import SimpleGrammar

if TYPE_CHECKING:
    from gemseo.core.discipline_data import Data
    from gemseo.utils.string_tools import MultiLineString


class SimplerGrammar(SimpleGrammar):
    """A simple grammar that does not check the data types."""

    def _validate(  # noqa: D102
        self,
        data: Data,
        error_message: MultiLineString,
    ) -> bool:
        return True
