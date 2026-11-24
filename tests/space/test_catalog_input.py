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
"""Tests for the helpers of the inputs fed by catalog variables."""

from __future__ import annotations

from typing import Final

import pytest
from numpy import array
from numpy.testing import assert_array_equal

from gemseo.space import DesignSpace
from gemseo.space._catalog_input import get_catalog_input_values
from gemseo.space._catalog_input import get_fed_input_names
from gemseo.space._catalog_input import get_feeding_variable_names
from gemseo.space._catalog_input import get_input_to_variable_names
from gemseo.space.variable import CatalogVariable
from gemseo.space.variable import RealVariable
from gemseo.util.testing.helper import assert_exception

_catalog: Final[CatalogVariable] = CatalogVariable(
    catalog={
        "young_modulus": [70.0, 200.0],
        "density": [2700, 7800],
        "supplier": ["A", "B"],
    },
)
"""A catalog variable with two numerical properties and a categorical one."""


@pytest.mark.parametrize(
    ("name", "variable", "input_names", "expected"),
    [
        pytest.param("x", RealVariable(), ["x", "density"], ("x",), id="real"),
        pytest.param("x", RealVariable(), ["y"], (), id="real-not-input"),
        pytest.param(
            "material",
            _catalog,
            ["density", "young_modulus"],
            ("young_modulus", "density"),
            id="properties-in-catalog-order",
        ),
        pytest.param(
            "material",
            _catalog,
            ["material", "density", "supplier"],
            ("material", "density"),
            id="name-and-property",
        ),
        pytest.param("material", _catalog, ["y"], (), id="catalog-not-input"),
        pytest.param(
            "density",
            _catalog,
            ["density", "young_modulus"],
            ("density", "young_modulus"),
            id="property-named-like-variable",
        ),
    ],
)
def test_get_fed_input_names(name, variable, input_names, expected) -> None:
    """Check the inputs fed by a variable."""
    assert get_fed_input_names(name, variable, input_names) == expected


def test_get_feeding_variable_names() -> None:
    """Check the variables feeding inputs, in the order of the mapping."""
    variables = {"x": RealVariable(), "material": _catalog, "z": RealVariable()}
    assert get_feeding_variable_names(variables, ["z", "density"]) == (
        "material",
        "z",
    )


def _create_space(
    *,
    catalog_name: str = "material",
    catalog_value: str | None = "1",
    with_catalog: bool = True,
) -> DesignSpace:
    """Create a design space with a real variable and maybe a catalog one.

    Args:
        catalog_name: The name of the catalog variable.
        catalog_value: The current value of the catalog variable.
        with_catalog: Whether to add the catalog variable.

    Returns:
        The design space.
    """
    space = DesignSpace()
    space.add_variable("x", value=1.0)
    if with_catalog:
        space.add_catalog_variable(catalog_name, _catalog.catalog, value=catalog_value)

    return space


def test_get_catalog_input_values() -> None:
    """Check the property values at the current alternative, with their dtype."""
    values = get_catalog_input_values(_create_space(), ["x", "density", "supplier"])
    assert values.keys() == {"density"}
    assert_array_equal(values["density"], array([7800]))
    assert values["density"].dtype == _catalog.catalog.properties["density"].dtype


@pytest.mark.parametrize(
    "space",
    [
        pytest.param(_create_space(with_catalog=False), id="no-catalog"),
        pytest.param(_create_space(catalog_value=None), id="no-current-value"),
        pytest.param(
            _create_space(catalog_name="density"), id="variable-named-like-property"
        ),
    ],
)
def test_get_catalog_input_values_empty(space) -> None:
    """Check that no property value is returned for the input density."""
    assert get_catalog_input_values(space, ["density"]) == {}


@pytest.mark.parametrize(
    "variables",
    [
        pytest.param({"skin": _catalog, "spar": _catalog}, id="two-catalogs"),
        pytest.param(
            {"density": RealVariable(), "material": _catalog},
            id="variable-and-property",
        ),
    ],
)
def test_get_input_to_variable_names_error(variables, snapshot) -> None:
    """Check the error raised when inputs are fed by several variables."""
    with assert_exception(ValueError, snapshot):
        get_input_to_variable_names(variables, ["density", "young_modulus"])


def _create_space_with_two_catalogs() -> DesignSpace:
    """Create a design space whose two catalog variables feed the same inputs.

    Returns:
        The design space.
    """
    space = _create_space()
    space.add_catalog_variable("spar", _catalog.catalog, value="0")
    return space


def test_get_feeding_variable_names_error_fed_twice(snapshot) -> None:
    """Check the error raised when an input is fed by several variables."""
    space = _create_space_with_two_catalogs()
    with assert_exception(ValueError, snapshot):
        get_feeding_variable_names(space.variables, ["density"])


def test_get_catalog_input_values_error_fed_twice(snapshot) -> None:
    """Check the error raised when an input is fed by several variables."""
    space = _create_space_with_two_catalogs()
    with assert_exception(ValueError, snapshot):
        get_catalog_input_values(space, ["density"])
