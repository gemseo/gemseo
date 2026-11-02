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

import logging

import pytest
from pydantic import BaseModel
from pydantic import ValidationError

from gemseo.util.pydantic import create_model
from gemseo.util.pydantic import log_replaced_setting
from gemseo.util.testing.helper import assert_exception


class Model(BaseModel):
    a: int


class WrongModel(BaseModel):
    a: int


def test_create_model_from_field():
    """Test creation from a field."""
    model = create_model(Model, a=1)
    assert model.a == 1


def test_create_model_from_field_failure():
    """Test creation from an unknown field."""
    with pytest.raises(ValidationError):
        create_model(Model, b=1)


def test_create_model_from_model():
    """Test creation from a model."""
    model_1 = Model(a=1)
    model_2 = create_model(Model, settings_model=model_1)
    assert model_1 == model_2


def test_create_model_from_a_wrong_model(snapshot):
    """Test creation from a wrong model."""
    with assert_exception(ValueError, snapshot):
        create_model(Model, settings_model=WrongModel(a=1))


@pytest.mark.parametrize(
    ("model", "value", "is_logged"),
    [
        (Model(a=1), 2, True),
        (Model(a=1), 1, False),
        (Model.model_construct(_fields_set=set(), a=1), 2, False),
    ],
    ids=["explicit_replaced", "explicit_same", "default"],
)
def test_log_replaced_setting(caplog, model, value, is_logged):
    """Check that only an explicit setting replaced by another value is logged."""
    log_replaced_setting(model, "a", value, "the origin")
    record = (
        "gemseo.util.pydantic",
        logging.WARNING,
        "The setting 'a' of Model (1) is replaced by the origin (2).",
    )
    assert caplog.record_tuples == ([record] if is_logged else [])
