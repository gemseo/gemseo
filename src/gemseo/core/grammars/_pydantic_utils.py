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
"""Pydantic model with strict fields validations."""
from __future__ import annotations

from typing import Any

from numpy import can_cast
from numpy import complex128
from numpy import complex64
from numpy import dtype
from numpy import ndarray
from pydantic import BoolError
from pydantic import PydanticTypeError
from pydantic import schema
from pydantic import validators
from pydantic.fields import ModelField
from pydantic.typing import get_origin


def strict_bool_validator(val: Any) -> bool:
    """Validate strictly a boolean.

    Args:
        val: The value to be validated.

    Returns:
        The validated value.

    Raises:
        BoolError: If the value is not valid.
    """
    if val not in (True, False):
        raise BoolError()
    return val


class NDArrayError(PydanticTypeError):
    """Error for validating a NumPy array."""

    msg_template = "value could not be parsed to a NumPy ndarray{dtype_info}."


class ComplexError(PydanticTypeError):
    """Error for validating a complex."""

    msg_template = "value could not be parsed to a complex"


def strict_ndarray_validator(val: Any, field: ModelField) -> ndarray:
    """Validate strictly a NumPy ndarray.

    Args:
        val: The value to be validated.
        field: The target model field.

    Returns:
        The validated value.

    Raises:
        NDArrayError: If the value is not valid.
    """
    if not isinstance(val, ndarray):
        raise NDArrayError(dtype_info="")

    if field.sub_fields and not can_cast(
        val.dtype, dtype(field.sub_fields[1].sub_fields[0].type_)
    ):
        type_ = field.sub_fields[1].sub_fields[0].type_
        if not can_cast(val.dtype, dtype(type_)):
            raise NDArrayError(dtype_info=f" with dtype {type_}")

    return val


def strict_complex_validator(val: Any) -> complex:
    """Validate complex data.

    Args:
        val: The value to be validated.

    Returns:
        The validated value.

    Raises:
        ComplexError: If the value is not valid.
    """
    if isinstance(val, (float, complex)):
        return val
    raise ComplexError()


def _type_analysis(self) -> None:
    """Monkey patch pydantic to handle NumPy ndarray.

    This will detect a ndarray as a pydantic generic type.
    """
    origin = get_origin(self.outer_type_)
    if origin == ndarray:
        # This is the temporary trick to force the wrapped method to detect a ndarray
        # as a generic type.
        self.model_config.arbitrary_types_allowed = True
    self._wrapped_type_analysis()
    if origin == ndarray:
        self.model_config.arbitrary_types_allowed = False


def field_type_schema(
    field: ModelField, **kwargs: Any
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    """Monkey patch pydantic to handle NumPy ndarray.

    This will detect a ndarray as a pydantic generic type.
    """
    f_schema, definitions, nested_models = schema.wrapped_field_type_schema(
        field, **kwargs
    )
    if field.type_ == ndarray:
        f_schema = {
            "type": "array",
            "items": {"type": f_schema["allOf"][0]["items"][1]["allOf"][0]["type"]},
        }
    return f_schema, definitions, nested_models


TYPE_TO_VALIDATOR_DEFS: dict[type, list[Any]] = {
    int: [validators.strict_int_validator],
    float: [validators.strict_float_validator],
    str: [validators.strict_str_validator],
    bool: [strict_bool_validator],
}
"""The binding from type to validator definitions as expected in pydantic internals."""

NEW_TYPE_TO_VALIDATOR_DEFS: dict[type, list[Any]] = {
    complex: [strict_complex_validator],
    complex64: [strict_complex_validator],
    complex128: [strict_complex_validator],
    ndarray: [strict_ndarray_validator],
}
"""The validators to be added to pydantic."""


def patch_pydantic() -> None:
    """Patch pydantic validators definitions."""
    for i, (type_, _) in enumerate(validators._VALIDATORS):
        validator_def = TYPE_TO_VALIDATOR_DEFS.get(type_)
        if validator_def is not None:
            validators._VALIDATORS[i] = (type_, validator_def)

    validators._VALIDATORS += NEW_TYPE_TO_VALIDATOR_DEFS.items()

    # Inject our monkey patch to handle NumPy ndarray.
    ModelField._wrapped_type_analysis = ModelField._type_analysis
    ModelField._type_analysis = _type_analysis
    schema.wrapped_field_type_schema = schema.field_type_schema
    schema.field_type_schema = field_type_schema
