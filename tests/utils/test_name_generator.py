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
#        :author: Gilberto RUIZ
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from re import match

import pytest

from gemseo.utils.base_name_generator import BaseNameGenerator
from gemseo.utils.name_generator import NameGenerator


@pytest.mark.parametrize(
    "naming",
    [
        BaseNameGenerator.Naming.UUID,
        BaseNameGenerator.Naming.NUMBERED,
        "WRONGMETHOD",
    ],
)
def test_generate_name(naming) -> None:
    """Test the name generation."""
    name_generator = NameGenerator(naming_method=naming)
    name_1 = name_generator.generate_name()
    name_2 = name_generator.generate_name()
    if naming == BaseNameGenerator.Naming.UUID:
        assert match(r"[0-9a-fA-F]{12}$", name_1) is not None
        assert name_1 != name_2
    elif naming == BaseNameGenerator.Naming.NUMBERED:
        assert name_1 == "1"
        assert name_2 == "2"
    else:
        assert name_1 is None
        assert name_2 is None
