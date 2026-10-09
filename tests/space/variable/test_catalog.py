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
import warnings

import pytest
from numpy import array
from numpy import inf
from numpy import int64
from numpy import nan
from numpy.testing import assert_array_equal
from pandas import DataFrame
from pydantic import ValidationError

from gemseo.space import Catalog
from gemseo.space.variable import CatalogVariable
from gemseo.space.variable import DataType
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
    assert variable.type == DataType.CATALOG
    assert variable.size == 1
    assert variable.coordinate_type is int64
    assert variable.catalog == Catalog(properties=table)
    # The bounds are derived from the number of alternatives.
    assert_array_equal(variable.lower_bound, array([0]))
    assert_array_equal(variable.upper_bound, array([2]))
    assert variable.lower_bound.dtype == int64


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

    assert_array_equal(variable.lower_bound, array([0]))
    assert_array_equal(variable.upper_bound, array([0]))
    assert variable.find_components_outside_domain(array([0])) == set()
    assert variable.find_components_outside_domain(array([1])) == {0}


@pytest.mark.parametrize("bound", ["lower_bound", "upper_bound"])
def test_bounds_are_not_settable(table, bound, snapshot) -> None:
    """Check that a bound cannot be passed explicitly."""
    with assert_exception(ValidationError, snapshot):
        CatalogVariable(catalog=table, **{bound: 0})


def test_size_is_not_settable(table, snapshot) -> None:
    """Check that a catalog variable is scalar and its size read-only."""
    with assert_exception(ValidationError, snapshot):
        CatalogVariable(catalog=table, size=2)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (array([0]), set()),
        (array([1]), set()),
        (array([2]), set()),
        (array([-1]), {0}),
        (array([3]), {0}),
        (array([1.5]), {0}),
        (array([0.0]), set()),
        (array([None]), set()),
    ],
)
def test_find_components_outside_domain(variable, value, expected) -> None:
    """Check the domain of a catalog variable.

    The domain is the integers from zero to the number of alternatives minus one;
    `None` stands for a value that is not set yet.
    """
    assert variable.find_components_outside_domain(value) == expected


@pytest.mark.parametrize("value", [array([inf]), array([-inf]), array([nan])])
def test_find_components_outside_domain_with_non_finite_value(variable, value) -> None:
    """Check that a non-finite value is out of the domain without any warning.

    An infinite or NaN position must be reported as out of the domain.
    An infinite one must also not reach `mod`,
    which emits a spurious `RuntimeWarning` on it
    (see `IntegerVariable.check_finite_bound_components`);
    `mod` is silent on a NaN, so that case only pins the returned set.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert variable.find_components_outside_domain(value) == {0}


@pytest.mark.parametrize(
    "value",
    [
        array(["steel"]),
        array([b"steel"]),
        array(["steel"], dtype=object),
    ],
)
def test_find_components_outside_domain_with_a_label(variable, value) -> None:
    """Check that a label passed instead of a position is out of the domain.

    A catalog variable's value is the position of an alternative,
    not its label; passing a label must not crash with a raw numpy error.
    """
    assert variable.find_components_outside_domain(value) == {0}


def test_get_normalization_mask(variable) -> None:
    """Check that a catalog variable is never normalized."""
    for enable_integer_normalization in (False, True):
        assert_array_equal(
            variable.get_normalization_mask(enable_integer_normalization),
            array([False]),
        )


def test_get_default_value(variable) -> None:
    """Check that the default value is the position of the first alternative."""
    value = variable.get_default_value()

    assert_array_equal(value, array([0]))
    assert value.dtype == int64


def test_get_out_of_domain_message(variable, snapshot) -> None:
    """Check the wording naming the items when a value is rejected."""
    assert variable._get_out_of_domain_message("x", array([5]), [0]) == snapshot


def test_get_out_of_domain_component_message(variable, snapshot) -> None:
    """Check the wording naming the items when a component is rejected."""
    assert variable._get_out_of_domain_component_message("x", 0, 5) == snapshot


def test_eq(variable, table) -> None:
    """Check that the catalog takes part in the equality of two variables."""
    assert variable == CatalogVariable(catalog=table)
    assert variable != CatalogVariable(catalog=table.iloc[:2])


def test_pickle(variable) -> None:
    """Check that a variable survives a pickle round-trip with its arrays frozen."""
    restored = pickle.loads(pickle.dumps(variable))

    assert restored == variable
    assert not restored.lower_bound.flags.writeable
    assert not restored.catalog.labels.flags.writeable
