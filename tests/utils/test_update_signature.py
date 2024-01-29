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
from inspect import signature
from textwrap import dedent
from typing import Any
from typing import ClassVar

import pytest
from pydantic import BaseModel
from pydantic import Field

from gemseo.utils._signature_updater import update_signature


def get_class(model_name: str = "AModel", method_name: str = "a_method") -> type:
    """Create a class with updated signature.

    Args:
        model_name: The name of the model to pass to the signature updater.
        method_name: The name of the method to pass to the signature updater.

    Returns:
        The class.
    """

    class AClass(metaclass=update_signature(model_name, method_name)):
        EXPECTED_KWARGS_NAMES = ("a", "b")
        EXPECTED_KWARGS_WITH_DEFAULTS: ClassVar[dict[str, Any]] = {"a": "hello", "b": 0}
        EXPECTED_KWARGS_WITHOUT_DEFAULTS: ClassVar[dict[str, str]] = {
            "a": "hello",
            "b": "world",
        }
        EXPECTED_DOCSTRING_ARGS_SECTION = """
    a: The description is missing.
    b: Description of b.
""".strip()
        EXPECTED_SIGNATURE = "(a: str, b: int = 0)"

        class AModel(BaseModel):
            a: str
            b: int = Field(0, description="Description of b.")

        def a_method(self, **kwargs):
            """The docstring of a_method.

            Args:
                **kwargs: The kwargs.

            Returns:
                Nothing.
            """
            return kwargs

        def bad_method_1(self, a):
            """This method cannot be updated."""

        def bad_method_2(self, a, **kwargs):
            """This method cannot be updated."""

    return AClass


def get_derived_class(override_model: bool) -> type:
    """Return a derived class with a parent that has an updated signature."""

    class DerivedClass(get_class()):
        if override_model:
            EXPECTED_KWARGS_NAMES = ("x", "y")
            EXPECTED_KWARGS_WITH_DEFAULTS = {"x": "hello", "y": 0.0}
            EXPECTED_KWARGS_WITHOUT_DEFAULTS = {"x": "hello", "y": "world"}
            EXPECTED_DOCSTRING_ARGS_SECTION = """
    x: The description is missing.
    y: The description is missing.
""".strip()
            EXPECTED_SIGNATURE = "(x: bool, y: float = 0.0)"

            class AModel(BaseModel):
                x: bool
                y: float = 0.0

    return DerivedClass


@pytest.fixture(params=(get_class(), get_derived_class(False), get_derived_class(True)))
def obj(request):
    return request.param()


def test_missing_positional_argument_error(obj):
    """Verify the required model field is well updated."""
    msg = (
        r".*() missing 1 required positional argument: '"
        f"{obj.EXPECTED_KWARGS_NAMES[0]}'"
    )
    with pytest.raises(TypeError, match=msg):
        obj.a_method()


def test_missing_argument_with_default(obj):
    """Verify the optional model field is well updated."""
    # We do not care about passing the proper type,
    # we only want to check the outcome.
    assert (
        obj.a_method(**{obj.EXPECTED_KWARGS_NAMES[0]: "hello"})
        == obj.EXPECTED_KWARGS_WITH_DEFAULTS
    )


def test_all_arguments(obj):
    """Verify calling with all the arguments."""
    # We do not care about passing the proper type,
    # we only want to check the outcome.
    assert (
        obj.a_method(**{
            obj.EXPECTED_KWARGS_NAMES[0]: "hello",
            obj.EXPECTED_KWARGS_NAMES[1]: "world",
        })
        == obj.EXPECTED_KWARGS_WITHOUT_DEFAULTS
    )


def test_docstring(obj):
    """Verify the docstring."""
    assert (
        dedent(obj.a_method.__doc__)
        == f"""
The docstring of a_method.

Args:
    {obj.EXPECTED_DOCSTRING_ARGS_SECTION}

Returns:
    Nothing.
""".strip()
    )


def test_signature(obj):
    """Verify the signature."""
    assert str(signature(obj.a_method)) == obj.EXPECTED_SIGNATURE


def test_missing_method_error():
    """Verify the error when the method cannot be found."""
    msg = "The method named dummy_name cannot be found."
    with pytest.raises(RuntimeError, match=re.escape(msg)):
        get_class(method_name="dummy_name")


@pytest.mark.parametrize("method_name", ["bad_method_1", "bad_method_2"])
def test_bad_signature_error(method_name):
    """Verify the error when the method does not have only kwargs."""
    msg = f"The method {method_name} must only have the argument **kwargs beside self."
    with pytest.raises(RuntimeError, match=re.escape(msg)):
        get_class(method_name=method_name)
