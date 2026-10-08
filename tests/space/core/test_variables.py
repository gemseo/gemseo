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
"""Tests for the generic Variables registry."""

from __future__ import annotations

from collections.abc import MutableMapping

import pytest
from numpy.testing import assert_array_equal

from gemseo.space._core.variables import Variables
from gemseo.space.variable import BaseVariable
from gemseo.space.variable import CatalogVariable
from gemseo.space.variable import DataType
from gemseo.space.variable import DiscreteVariable
from gemseo.space.variable import IntegerVariable
from gemseo.space.variable import RealVariable
from gemseo.util.testing.helper import assert_exception


def _create_variable(name: str) -> BaseVariable:
    """Create a variable of the kind encoded by its name.

    Args:
        name: The name of the variable,
            `"x"` for real, `"n"` for integer,
            `"d"` for discrete and `"c"` for catalog.

    Returns:
        The variable.
    """
    if name == "n":
        return IntegerVariable(lower_bound=0, upper_bound=10)
    if name == "d":
        return DiscreteVariable(choices=[1, 2])
    if name == "c":
        return CatalogVariable(catalog={"property": [1, 2]})
    return RealVariable(lower_bound=0.0, upper_bound=1.0)


@pytest.fixture
def variables() -> Variables:
    """A variables with a real and an integer variable."""
    variables = Variables()
    variables["x"] = RealVariable(size=2, lower_bound=0.0, upper_bound=1.0)
    variables["n"] = IntegerVariable(size=1, lower_bound=0, upper_bound=10)
    return variables


def test_mapping_interface(variables) -> None:
    """Check that the variables reads as a name-to-variable mapping."""
    assert isinstance(variables, MutableMapping)
    assert list(variables) == ["x", "n"]
    assert len(variables) == 2
    assert "x" in variables
    assert "missing" not in variables
    assert list(variables.keys()) == ["x", "n"]
    assert [variable.size for variable in variables.values()] == [2, 1]
    assert dict(variables.items()).keys() == {"x", "n"}
    assert variables["x"].type == DataType.REAL
    assert variables.get("missing") is None


def test_getitem_unknown_variable(variables) -> None:
    """Check that indexing an unknown variable raises."""
    with pytest.raises(KeyError):
        variables["missing"]


@pytest.mark.parametrize(
    ("names", "type_", "expected"),
    [
        ((), DataType.REAL, False),
        ((), DataType.INTEGER, False),
        ((), DataType.DISCRETE, False),
        ((), DataType.CATALOG, False),
        (("x",), DataType.REAL, True),
        (("x",), DataType.INTEGER, False),
        (("x",), DataType.DISCRETE, False),
        (("x",), DataType.CATALOG, False),
        (("n",), DataType.INTEGER, True),
        (("x", "n"), DataType.INTEGER, True),
        (("d",), DataType.DISCRETE, True),
        (("x", "d"), DataType.DISCRETE, True),
        (("d",), DataType.CATALOG, False),
        (("c",), DataType.CATALOG, True),
        (("x", "c"), DataType.CATALOG, True),
        (("c",), DataType.DISCRETE, False),
    ],
)
def test_has_variable(names, type_, expected) -> None:
    """Check the detection of a variable of a given type."""
    variables = Variables()
    for name in names:
        variables[name] = _create_variable(name)

    assert variables.has_variables_of_type(type_) is expected


def test_get_integer_components(variables) -> None:
    """Check the integer-component mask."""
    assert_array_equal(variables.get_integer_mask(), [False, False, True])


def test_setitem_insert(variables) -> None:
    """Check that setting a new name appends it and allocates its indices."""
    version = variables.version
    variables["z"] = RealVariable(size=3, lower_bound=0.0, upper_bound=1.0)
    assert list(variables) == ["x", "n", "z"]
    assert variables.size == 6
    assert variables.name_to_indices["z"] == range(3, 6)
    assert variables.version == version + 1


def test_setitem_replace_same_size(variables) -> None:
    """Check that replacing keeps position, size and index ranges."""
    variables["x"] = RealVariable(size=2, lower_bound=-1.0, upper_bound=2.0)
    assert list(variables) == ["x", "n"]
    assert variables.size == 3
    assert variables.name_to_indices["x"] == range(2)
    assert variables.name_to_indices["n"] == range(2, 3)
    assert_array_equal(variables["x"].lower_bound, [-1.0, -1.0])


def test_setitem_replace_resize(variables) -> None:
    """Check that replacing with a different size rebuilds indices and size."""
    variables["x"] = RealVariable(size=4, lower_bound=0.0, upper_bound=1.0)
    assert variables.size == 5
    assert variables.name_to_indices["x"] == range(4)
    assert variables.name_to_indices["n"] == range(4, 5)


def test_delitem(variables) -> None:
    """Check that deleting a variable removes it and rebuilds indices."""
    version = variables.version
    del variables["x"]
    assert list(variables) == ["n"]
    assert variables.size == 1
    assert variables.name_to_indices["n"] == range(1)
    assert variables.version == version + 1


def test_delitem_unknown_variable(variables) -> None:
    """Check that deleting an unknown variable raises."""
    with pytest.raises(KeyError):
        del variables["missing"]


def test_rename(variables) -> None:
    """Check that renaming preserves the order and the index ranges."""
    version = variables.version
    variables.rename("x", "y")
    assert list(variables) == ["y", "n"]
    assert variables.name_to_indices["y"] == range(2)
    assert variables.version == version + 1


def test_rename_unknown_variable(variables) -> None:
    """Check that renaming an unknown variable raises."""
    with pytest.raises(KeyError):
        variables.rename("missing", "y")


def test_rename_collision(variables, snapshot) -> None:
    """Check that renaming to an already registered name raises and does not mutate."""
    version = variables.version
    with assert_exception(ValueError, snapshot):
        variables.rename("x", "n")
    assert list(variables) == ["x", "n"]
    assert variables.size == 3
    assert variables.version == version


def test_rename_same_name_is_noop(variables) -> None:
    """Check that renaming a variable to its own name is a no-op."""
    version = variables.version
    variables.rename("x", "x")
    assert list(variables) == ["x", "n"]
    assert variables.version == version + 1


def test_filter_components(variables) -> None:
    """Check that filtering the components of a variable resizes it."""
    version = variables.version
    variables.filter_components("x", [1])
    assert variables["x"].size == 1
    assert variables.size == 2
    assert variables.name_to_indices["n"] == range(1, 2)
    assert variables.version == version + 1


def test_filter_components_keeping_all(variables) -> None:
    """Check that keeping every component in order shares the variable."""
    variable = variables["x"]
    version = variables.version
    variables.filter_components("x", [0, 1])

    assert variables["x"] is variable
    assert variables.version == version + 1


def test_filter_components_of_a_discrete_variable() -> None:
    """Check that filtering the only component of a discrete variable is an identity."""
    variables = Variables()
    variables["d"] = DiscreteVariable(choices=[2, 4])
    variable = variables["d"]
    variables.filter_components("d", [0])

    assert variables["d"] is variable
    assert_array_equal(variables["d"].choices, [2.0, 4.0])


def test_has_variable_tracks_mutations_of_a_discrete_variable() -> None:
    """Check that the discrete detection stays correct across mutations."""
    variables = Variables()
    variables["x"] = RealVariable(lower_bound=0.0, upper_bound=1.0)
    assert variables.has_variables_of_type(DataType.DISCRETE) is False

    variables["d"] = DiscreteVariable(choices=[1, 2])
    assert variables.has_variables_of_type(DataType.DISCRETE) is True

    # Overwriting a discrete variable with a real one turns it off.
    variables["d"] = RealVariable(lower_bound=0.0, upper_bound=1.0)
    assert variables.has_variables_of_type(DataType.DISCRETE) is False

    # Overwriting a real variable with a discrete one turns it on.
    variables["d"] = DiscreteVariable(choices=[1, 2])
    assert variables.has_variables_of_type(DataType.DISCRETE) is True

    # Removing the last discrete variable turns it off.
    del variables["d"]
    assert variables.has_variables_of_type(DataType.DISCRETE) is False

    # filter_components() preserves the kind of the variable (a discrete
    # variable is always scalar, so only the identity filtering applies),
    # so it must not flip the result either way.
    variables["d"] = DiscreteVariable(choices=[1, 2, 3])
    variables.filter_components("d", [0])
    assert variables.has_variables_of_type(DataType.DISCRETE) is True

    del variables["d"]
    assert variables.has_variables_of_type(DataType.DISCRETE) is False


def test_has_variable_tracks_mutations_of_a_catalog_variable() -> None:
    """Check that the catalog detection stays correct across mutations."""
    variables = Variables()
    variables["x"] = RealVariable(lower_bound=0.0, upper_bound=1.0)
    assert variables.has_variables_of_type(DataType.CATALOG) is False

    variables["c"] = CatalogVariable(catalog={"property": [1, 2]})
    assert variables.has_variables_of_type(DataType.CATALOG) is True

    # Overwriting a catalog variable with a discrete one turns it off,
    # and does not confuse the two types.
    variables["c"] = DiscreteVariable(choices=[1, 2])
    assert variables.has_variables_of_type(DataType.CATALOG) is False
    assert variables.has_variables_of_type(DataType.DISCRETE) is True

    variables["c"] = CatalogVariable(catalog={"property": [1, 2]})
    assert variables.has_variables_of_type(DataType.CATALOG) is True
    assert variables.has_variables_of_type(DataType.DISCRETE) is False

    del variables["c"]
    assert variables.has_variables_of_type(DataType.CATALOG) is False


def test_has_variable_across_mixed_mutations() -> None:
    """Check that the discrete and catalog detections stay exact across mutations.

    The sequence mixes every kind of registry mutation
    (insertion of each kind, renaming, component filtering and deletion),
    including overwriting a name with a variable of a different kind,
    which is the classic way a detection drifts from the registry it describes.
    """
    variables = Variables()

    variables["a"] = RealVariable(lower_bound=0.0, upper_bound=1.0)
    assert variables.has_variables_of_type(DataType.DISCRETE) is False
    assert variables.has_variables_of_type(DataType.CATALOG) is False

    variables["b"] = DiscreteVariable(choices=[1, 2])
    assert variables.has_variables_of_type(DataType.DISCRETE) is True
    assert variables.has_variables_of_type(DataType.CATALOG) is False

    variables["c"] = CatalogVariable(catalog={"property": [1, 2]})
    assert variables.has_variables_of_type(DataType.DISCRETE) is True
    assert variables.has_variables_of_type(DataType.CATALOG) is True

    # Renaming does not add or remove a variable, so the detections are unaffected.
    variables.rename("c", "cat")
    assert variables.has_variables_of_type(DataType.DISCRETE) is True
    assert variables.has_variables_of_type(DataType.CATALOG) is True

    # Filtering the only component of a scalar discrete variable is an
    # identity, so it must not flip either detection.
    variables.filter_components("b", [0])
    assert variables.has_variables_of_type(DataType.DISCRETE) is True
    assert variables.has_variables_of_type(DataType.CATALOG) is True

    # Overwriting "b" (discrete) with a catalog variable of the same name
    # must turn the discrete detection off and the catalog one on.
    variables["b"] = CatalogVariable(catalog={"property": [3, 4]})
    assert variables.has_variables_of_type(DataType.DISCRETE) is False
    assert variables.has_variables_of_type(DataType.CATALOG) is True

    del variables["cat"]
    assert variables.has_variables_of_type(DataType.DISCRETE) is False
    assert variables.has_variables_of_type(DataType.CATALOG) is True

    del variables["b"]
    assert variables.has_variables_of_type(DataType.DISCRETE) is False
    assert variables.has_variables_of_type(DataType.CATALOG) is False
    assert list(variables) == ["a"]


def test_catalog_variable_is_integer() -> None:
    """Check that the integer mask marks a catalog variable as integer.

    It is not an integer variable,
    but its value is a position in its catalog, and so a whole number.
    """
    variables = Variables()
    variables["c"] = CatalogVariable(catalog={"property": [1, 2]})

    assert variables.has_variables_of_type(DataType.INTEGER) is False
    assert_array_equal(variables.get_integer_mask(), [True])


def test_filter_components_of_a_catalog_variable() -> None:
    """Check that filtering the only component of a catalog variable is identity."""
    variables = Variables()
    variables["c"] = CatalogVariable(catalog={"property": [1, 2]})
    variable = variables["c"]
    variables.filter_components("c", [0])

    assert variables["c"] is variable
    assert variables["c"].catalog == variable.catalog
