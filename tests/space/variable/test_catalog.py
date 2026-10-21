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
"""Tests for the catalog variable."""

from __future__ import annotations

import pickle

import pytest
from numpy import array
from numpy import int64
from numpy.testing import assert_array_equal
from pandas import DataFrame
from pydantic import ValidationError

from gemseo.enum import VariableType
from gemseo.space import Catalog
from gemseo.space.variable import CatalogVariable
from gemseo.util.testing.helper import assert_exception


@pytest.fixture
def table() -> DataFrame:
    """A catalog of materials."""
    return DataFrame(
        {"mass": [2.7, 7.8, 4.5], "cost": [10.0, 5.0, 50.0]},
        index=["aluminium", "steel", "titanium"],
    )


@pytest.fixture
def variable(table) -> CatalogVariable:
    """A catalog variable over a catalog of materials."""
    return CatalogVariable(catalog=table)


def test_fields(variable, table) -> None:
    """Check the fields of a catalog variable."""
    assert variable.type == VariableType.CATALOG
    assert variable.size == 1
    assert variable.coordinate_type is int64
    assert variable.catalog == Catalog(properties=table)
    assert variable.labels == ("aluminium", "steel", "titanium")


@pytest.mark.parametrize("catalog_", ["table", "mapping", "catalog"])
def test_catalog_types(catalog_, table) -> None:
    """Check that the catalog can be a table, a mapping or a catalog."""
    catalogs = {
        "table": table,
        "mapping": {"mass": [2.7, 7.8, 4.5], "cost": [10.0, 5.0, 50.0]},
        "catalog": Catalog(properties=table),
    }
    variable = CatalogVariable(catalog=catalogs[catalog_])
    assert len(variable.catalog) == 3


@pytest.mark.parametrize("labels", [(), ["a", "b", "c"]])
def test_model_dump_round_trip(table, labels) -> None:
    """Check that a dumped catalog variable is validated back into the same one.

    The dump of the catalog is a mapping from its field names to its fields,
    which is read as the fields of a catalog, not as its properties,
    which would give two properties named `properties` and `labels`.
    """
    variable = CatalogVariable(catalog=Catalog(properties=table, labels=labels))

    assert CatalogVariable.model_validate(variable.model_dump()) == variable


def test_json_schema() -> None:
    """Check that the JSON schema of a catalog variable embeds the one of a catalog."""
    schema = CatalogVariable.model_json_schema()

    assert schema["properties"]["catalog"]["$ref"] == "#/$defs/Catalog"
    assert "Catalog" in schema["$defs"]


def test_properties_named_as_the_catalog_fields() -> None:
    """Check that properties named as the catalog fields remain properties."""
    variable = CatalogVariable(catalog={"properties": [1.0, 2.0], "labels": ["x", "y"]})

    assert variable.catalog.to_dataframe().columns.tolist() == [
        "properties",
        "labels",
    ]
    assert_array_equal(variable.catalog.labels, array(["0", "1"]))


def test_one_row_catalog() -> None:
    """Check that a single-alternative catalog gives a degenerate variable.

    Such a variable has a single admissible value and no combinatorics;
    it is accepted as is.
    """
    variable = CatalogVariable(catalog={"mass": 2.7})

    assert variable.find_components_outside_domain(array([0])) == set()
    assert variable.find_components_outside_domain(array([1])) == {0}


@pytest.mark.parametrize(
    ("name", "value"), [("size", 2), ("labels", ("aluminium", "steel"))]
)
def test_derived_member_is_not_settable(table, name, value, snapshot) -> None:
    """Check that the size and the labels derive from the catalog."""
    with assert_exception(ValidationError, snapshot):
        CatalogVariable(catalog=table, **{name: value})


def test_get_out_of_domain_message(variable, snapshot) -> None:
    """Check the wording naming the items when a value is rejected."""
    assert variable._get_out_of_domain_message("x", array([5]), [0]) == snapshot


def test_eq(variable, table) -> None:
    """Check that the catalog takes part in the equality of two variables."""
    assert variable == CatalogVariable(catalog=table)
    assert variable != CatalogVariable(catalog=table.iloc[:2])


def test_pickle(variable) -> None:
    """Check that a variable survives a pickle round-trip with its arrays frozen."""
    restored = pickle.loads(pickle.dumps(variable))

    assert restored == variable
    assert restored.labels == variable.labels
    assert not restored.catalog.labels.flags.writeable
