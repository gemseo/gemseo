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
from __future__ import annotations

import pytest

from gemseo.core.data_converters.factory import DataConverterFactory
from gemseo.core.data_converters.json import JSONGrammarDataConverter
from gemseo.core.data_converters.pydantic import PydanticGrammarDataConverter
from gemseo.core.data_converters.simple import SimpleGrammarDataConverter
from gemseo.core.grammars.simple_grammar import SimpleGrammar


@pytest.mark.parametrize(
    "cls",
    [
        JSONGrammarDataConverter,
        SimpleGrammarDataConverter,
        PydanticGrammarDataConverter,
    ],
)
def test_data_converter_factory(cls):
    assert isinstance(
        DataConverterFactory().create(cls.__name__, SimpleGrammar("dummy")),
        cls,
    )
