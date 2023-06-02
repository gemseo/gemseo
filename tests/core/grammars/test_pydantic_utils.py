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

import re

import pytest
from gemseo.core.grammars._pydantic_utils import ComplexError
from gemseo.core.grammars._pydantic_utils import NDArrayError
from gemseo.core.grammars._pydantic_utils import strict_bool_validator
from gemseo.core.grammars._pydantic_utils import strict_complex_validator
from gemseo.core.grammars._pydantic_utils import strict_ndarray_validator
from numpy import empty
from numpy.testing import assert_array_equal
from pydantic import BoolError

from .pydantic_models import get_model4

model4 = pytest.fixture(get_model4)


@pytest.mark.parametrize("value", (True, False))
def test_strict_bool_validator(value):
    """Verify the strict bool validator."""
    assert strict_bool_validator(value) == value


@pytest.mark.parametrize("value", ("no", "yes", 0.1))
def test_strict_bool_validator_error(value):
    """Verify the strict bool validator errors."""
    msg = "value could not be parsed to a boolean"
    with pytest.raises(BoolError, match=msg):
        strict_bool_validator(value)


@pytest.mark.parametrize("value", (1.0j, 0.0))
def test_strict_complex_validator(value):
    """Verify the strict complex validator error."""
    assert strict_complex_validator(value) == value


@pytest.mark.parametrize("value", ("no", 0, True))
def test_strict_complex_validator_error(value):
    """Verify the strict complex validator errors."""
    msg = "value could not be parsed to a complex"
    with pytest.raises(ComplexError, match=msg):
        strict_complex_validator(value)


@pytest.mark.parametrize(
    "array,names",
    (
        (empty(2, int), ("name1", "name2", "name3")),
        (empty(2, float), ("name2", "name3")),
    ),
)
def test_strict_ndarray_validator(model4, array, names):
    """Verify the strict ndarray validator."""
    for name in names:
        assert_array_equal(
            strict_ndarray_validator(array, model4.__fields__[name]), array
        )


@pytest.mark.parametrize(
    "data,name,msg_part",
    (
        (empty(2, float), "name1", " with dtype <class 'int'>"),
        ([0], "name1", ""),
        ([0], "name2", ""),
        ([0], "name3", ""),
    ),
)
def test_strict_ndarray_validator_error(model4, data, name, msg_part):
    """Verify the strict ndarray validator errors."""
    msg = f"value could not be parsed to a NumPy ndarray{msg_part}."
    with pytest.raises(NDArrayError, match=re.escape(msg)):
        strict_ndarray_validator(data, model4.__fields__[name])
