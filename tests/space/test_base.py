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
"""Tests for the BaseVariableSpace abstract class.

The behavior of the base class is exercised through its concrete subclasses,
[DesignSpace][gemseo.space.design.DesignSpace] and
[RandomSpace][gemseo.space.random.RandomSpace]; only what is not reachable
through them is tested here.
"""

from __future__ import annotations

import pickle
from copy import deepcopy

import pytest
from numpy import array
from numpy import inf
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from gemseo.space._core.variables import UnknownVariableError
from gemseo.space._core.variables import Variables
from gemseo.space._design.variables import DesignVariables
from gemseo.space._random.variables import RandomVariables
from gemseo.space._random.variables_view import RandomVariablesView
from gemseo.space.base import BaseVariableSpace
from gemseo.space.design import DesignSpace
from gemseo.space.random import RandomSpace
from gemseo.space.variables_view import VariablesView
from gemseo.uncertainty.distribution.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.triangular_settings import (
    OTTriangularDistribution_Settings,
)
from gemseo.util.read_only_mapping import ReadOnlyMapping
from gemseo.util.testing.helper import assert_exception

OT_NORMAL = OTNormalDistribution_Settings()


def add_variable(space: BaseVariableSpace, name: str) -> None:
    """Add a variable to a space, whatever its type.

    Args:
        space: The space of variables.
        name: The name of the variable.
    """
    if isinstance(space, RandomSpace):
        space.add_variable(name, OT_NORMAL)
    else:
        space.add_variable(name)


@pytest.fixture(params=[DesignSpace, RandomSpace])
def space(request) -> BaseVariableSpace:
    """A space of variables with a single variable named ``"x"``."""
    space = request.param()
    add_variable(space, "x")
    return space


def test_is_abstract() -> None:
    """Check that the base class cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseVariableSpace()


@pytest.mark.parametrize(
    "method_name",
    [
        "add_variable",
        "get_pretty_table",
        "reference_value",
        "transform_vect",
        "untransform_vect",
    ],
)
def test_abstract_methods(method_name) -> None:
    """Check the abstract methods of the base class."""
    assert method_name in BaseVariableSpace.__abstractmethods__


def build_bounded_design_space() -> DesignSpace:
    """Return a design space with a single variable bounded by 0 and 2.

    Returns:
        The design space.
    """
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=2.0)
    return space


def build_normal_random_space() -> RandomSpace:
    """Return a random space with a single standard normal random variable.

    Returns:
        The random space.
    """
    space = RandomSpace()
    space.add_variable("x", OT_NORMAL)
    return space


@pytest.mark.parametrize(
    ("build_space", "value"),
    [(build_bounded_design_space, 1.0), (build_normal_random_space, 0.0)],
)
def test_transform_vect_keyword_argument(build_space, value) -> None:
    """Check that the mapping to the unit hypercube accepts the ``x_vect`` keyword.

    [BaseVariableSpace.transform_vect][gemseo.space.base.BaseVariableSpace.transform_vect]
    and
    [BaseVariableSpace.untransform_vect][gemseo.space.base.BaseVariableSpace.untransform_vect]
    are abstract,
    so every space must name their first parameter ``x_vect``,
    to preserve the Liskov substitution principle.
    """
    space = build_space()
    vector = array([value])
    unit_vector = space.transform_vect(x_vect=vector)
    assert_allclose(unit_vector, array([0.5]))
    assert_allclose(space.untransform_vect(x_vect=unit_vector), vector)


def test_add_variable_duplicate_name(space, snapshot) -> None:
    """Check the error raised when adding a variable whose name already exists."""
    with assert_exception(ValueError, snapshot):
        add_variable(space, "x")


@pytest.mark.parametrize(
    ("cls", "registry_class"),
    [
        (BaseVariableSpace, Variables),
        (DesignSpace, DesignVariables),
        (RandomSpace, RandomVariables),
    ],
)
def test_variables_class(cls, registry_class) -> None:
    """Check the class of the registry used by a space of variables."""
    assert cls._variables_class is registry_class


@pytest.mark.parametrize("cls", [DesignSpace, RandomSpace])
def test_subclasses(cls) -> None:
    """Check that the concrete spaces derive from the base class."""
    assert issubclass(cls, BaseVariableSpace)
    assert cls().name == ""
    assert isinstance(cls()._variables, cls._variables_class)


def test_filter_with_an_iterator(space) -> None:
    """Check that the names of the variables to be kept can be given as an iterator.

    The names are read twice, once to validate them and once to select the
    variables to be removed, so an iterator must be materialized first;
    otherwise the validation exhausts it and every variable is removed.
    """
    add_variable(space, "y")

    space.filter(name for name in ("x",))

    assert list(space.variables) == ["x"]


def test_filter_dimensions_hook() -> None:
    """Check that the base class validates and delegates the dimension filtering."""
    # Only a design space has something to do beyond filtering the registry,
    # namely resizing the current value.
    assert RandomSpace._filter_dimensions is BaseVariableSpace._filter_dimensions
    assert DesignSpace._filter_dimensions is not BaseVariableSpace._filter_dimensions


@pytest.mark.parametrize("cls", [DesignSpace, RandomSpace])
def test_reference_value_of_an_empty_space(cls) -> None:
    """Check that an empty space of variables defines no reference value."""
    assert cls().reference_value == {}


def test_reference_value_of_a_design_space() -> None:
    """Check that the reference value of a design space is its current value."""
    space = DesignSpace()
    space.add_variable("x", size=2)
    space.add_variable("y")
    space.set_current_variable("x", array([1.0, 2.0]))

    # The current value is partial, so the design space defines no reference value.
    assert space.reference_value == {}

    space.set_current_variable("y", array([3.0]))
    reference_value = space.reference_value
    assert list(reference_value) == ["x", "y"]
    assert_equal(reference_value["x"], array([1.0, 2.0]))
    assert_equal(reference_value["y"], array([3.0]))


def test_reference_value_of_a_random_space() -> None:
    """Check that the reference value of a random space is the mean of its variables."""
    space = RandomSpace()
    space.add_variable("x", OT_NORMAL, OT_NORMAL)
    space.add_variable(
        "y", OTTriangularDistribution_Settings(minimum=0.0, mode=1.0, maximum=2.0)
    )

    reference_value = space.reference_value
    assert list(reference_value) == ["x", "y"]
    assert_allclose(reference_value["x"], array([0.0, 0.0]))
    assert_allclose(reference_value["y"], array([1.0]))


@pytest.mark.parametrize("cls", [DesignSpace, RandomSpace])
def test_current_value_of_an_empty_space(cls) -> None:
    """Check that an empty space of variables has an empty current value."""
    assert cls()._current_value == {}


def test_current_value_of_a_random_space() -> None:
    """Check that a random space has no current value and ignores a write."""
    space = RandomSpace()
    space.add_variable("x", OT_NORMAL)

    space._current_value = {"x": array([123.0])}

    assert space._current_value == {}


def test_to_complex_of_a_random_space() -> None:
    """Check that a random space has no current value and so nothing to cast."""
    space = RandomSpace()
    space.add_variable("x", OT_NORMAL)

    space._to_complex()

    assert space._current_value == {}


def test_current_value_of_a_design_space_covers_every_variable() -> None:
    """Check that the current value of a design space covers all its variables.

    A variable without a value is mapped to `None`,
    unlike
    [get_current_value][gemseo.space.design.DesignSpace.get_current_value],
    which omits it,
    so that writing the mapping back restores the current value exactly.
    """
    space = DesignSpace()
    space.add_variable("x", size=2)
    space.add_variable("y")
    assert space._current_value == {"x": None, "y": None}

    space.set_current_variable("x", array([1.0, 2.0]))
    assert list(space._current_value) == ["x", "y"]
    assert_equal(space._current_value["x"], array([1.0, 2.0]))
    assert space._current_value["y"] is None
    assert list(space.get_current_value(as_dict=True)) == ["x"]


def test_set_current_value_of_a_partially_valued_design_space() -> None:
    """Check that a partial current value of a design space round-trips.

    The variables without a value keep their `None` marker,
    which
    [get_current_value][gemseo.space.design.DesignSpace.get_current_value]
    would have dropped, making the mapping incomplete
    and its writing back an error.
    """
    space = DesignSpace()
    space.add_variable("x", size=2)
    space.add_variable("y")
    space.set_current_variable("x", array([1.0, 2.0]))
    # The current value of a design space is a live view, not a copy.
    current_value = deepcopy(space._current_value)

    space.set_current_variable("x", array([3.0, 4.0]))
    space.set_current_variable("y", array([5.0]))
    space._current_value = current_value

    assert not space.has_current_value
    assert_equal(space._current_value["x"], array([1.0, 2.0]))
    assert space._current_value["y"] is None


def test_set_current_value_of_a_design_space_defining_none() -> None:
    """Check that writing back the current value of an unvalued design space clears it.

    Every variable is mapped to `None`,
    so the write marks them all as having no value again.
    """
    space = DesignSpace()
    space.add_variable("x")
    current_value = deepcopy(space._current_value)

    space.set_current_value({"x": array([0.0])})
    space._current_value = current_value

    assert space.get_current_value(as_dict=True) == {}


def test_set_current_value_of_a_design_space_ignores_unknown_names() -> None:
    """Check that writing the current value of a design space ignores unknown names.

    A formulation filters the design space of its problem
    after the problem stored the current value to restore on reset,
    so the mapping can name variables that the space no longer has.
    """
    space = DesignSpace()
    space.add_variable("x", value=1.0)
    space.add_variable("y", value=2.0)
    current_value = deepcopy(space._current_value)

    space.filter(["x"])
    space.set_current_variable("x", array([3.0]))
    space._current_value = current_value

    assert_equal(space.get_current_value(), array([1.0]))


@pytest.mark.parametrize("cls", [DesignSpace, RandomSpace])
def test_check_an_empty_space(cls, snapshot) -> None:
    """Check that an empty space of variables cannot pass the check."""
    with assert_exception(ValueError, snapshot):
        cls().check()


@pytest.mark.parametrize("cls", [DesignSpace, RandomSpace])
def test_check_a_non_empty_space(cls) -> None:
    """Check that a space of variables with a variable passes the check."""
    space = cls()
    add_variable(space, "x")

    space.check()


def test_random_space_is_not_a_design_space() -> None:
    """Check that a random space is not a design space.

    Bounds, current value, normalization and serialization
    are specific to a design space,
    so every consumer requiring one tests the space with `isinstance`.
    This invariant is what makes those tests correct.
    """
    assert not issubclass(RandomSpace, DesignSpace)


def test_render_footer_is_empty_by_default() -> None:
    """Check that a space of variables has no footer by default."""
    space = DesignSpace()
    space.add_variable("x")
    table = space.get_pretty_table(with_index=True, capitalize=True).get_string()
    assert space._render_footer() == ""
    assert str(space) == f"Design space:\n{table}"


@pytest.mark.parametrize(
    ("cls", "view_class"),
    [
        (BaseVariableSpace, VariablesView),
        (DesignSpace, VariablesView),
        (RandomSpace, RandomVariablesView),
    ],
)
def test_variables_view_class(cls, view_class) -> None:
    """Check the class of the read-only view used by a space of variables."""
    assert cls._variables_view_class is view_class


def test_variables_view(space) -> None:
    """Check that the view gives access to the variables of the space."""
    view = space.variables
    assert isinstance(view, space._variables_view_class)
    assert isinstance(view, ReadOnlyMapping)
    assert view is space.variables
    assert list(view) == ["x"]
    assert len(view) == 1
    assert "x" in view
    assert view["x"] is space._variables["x"]
    assert list(view.items()) == [("x", space._variables["x"])]
    assert repr(view).startswith(f"{space._variables_view_class.__name__}(")


def test_variables_view_name_to_indices(space) -> None:
    """Check that the view gives access to the indices of the variables."""
    add_variable(space, "y")
    name_to_indices = space.variables.name_to_indices
    assert name_to_indices is space._variables.name_to_indices
    assert dict(name_to_indices) == {"x": range(1), "y": range(1, 2)}
    with pytest.raises(TypeError, match="does not support item assignment"):
        name_to_indices["x"] = range(2)


def test_variables_view_has_integer_variables(space) -> None:
    """Check that the view tells whether a variable is of integer type.

    This is how a consumer typed on the base class reads it, whatever the space;
    the random variables are always of float type.
    """
    assert not space.variables.has_integer_variables

    if isinstance(space, DesignSpace):
        space.add_variable("n", type_=DesignSpace.DesignVariableType.INTEGER)
        assert space.variables.has_integer_variables


def test_variables_view_is_live(space) -> None:
    """Check that the view reflects the mutations of the space."""
    add_variable(space, "y")
    assert list(space.variables) == ["x", "y"]
    space.rename_variable("y", "z")
    assert list(space.variables) == ["x", "z"]
    space.remove_variable("x")
    assert list(space.variables) == ["z"]


def test_variables_view_unknown_name(space, snapshot) -> None:
    """Check the error raised when reading a variable that does not exist."""
    with assert_exception(UnknownVariableError, snapshot):
        space.variables["y"]


def test_variables_view_forbids_item_assignment(space) -> None:
    """Check that a variable cannot be replaced through the view."""
    with pytest.raises(TypeError, match="does not support item assignment"):
        space.variables["x"] = space._variables["x"]


def test_variables_view_forbids_item_deletion(space) -> None:
    """Check that a variable cannot be deleted through the view."""
    with pytest.raises(TypeError, match="does not support item deletion"):
        del space.variables["x"]


@pytest.mark.parametrize(
    "method_name",
    [
        "pop",
        "popitem",
        "clear",
        "update",
        "setdefault",
        "rename",
        "filter_components",
        "bump_version",
        "add_copula",
    ],
)
def test_variables_view_has_no_mutator(space, method_name) -> None:
    """Check that the view exposes no method mutating the registry."""
    assert not hasattr(space.variables, method_name)


def test_variables_view_cannot_be_replaced(space) -> None:
    """Check that the view of a space cannot be rebound."""
    with pytest.raises(AttributeError):
        space.variables = None


def copy_with_pickle(space: BaseVariableSpace) -> BaseVariableSpace:
    """Copy a space of variables through a pickle round trip.

    Args:
        space: The space of variables.

    Returns:
        The copy of the space of variables.
    """
    return pickle.loads(pickle.dumps(space))


@pytest.mark.parametrize("copy_space", [deepcopy, copy_with_pickle])
def test_variables_view_after_copy(space, copy_space) -> None:
    """Check that the view of a copied space views the registry of this copy."""
    other_space = copy_space(space)
    assert other_space.variables is not space.variables
    add_variable(other_space, "y")
    assert list(other_space.variables) == ["x", "y"]
    assert list(space.variables) == ["x"]


@pytest.mark.parametrize(
    ("bound_name", "stale_bound"), [("lower_bound", -inf), ("upper_bound", inf)]
)
def test_variables_view_replaces_a_changed_variable(bound_name, stale_bound) -> None:
    """Check that changing a variable replaces it instead of mutating it.

    Only a design space can change the bounds of a variable;
    the bounds of a random variable derive from its distribution.
    """
    space = DesignSpace()
    space.add_variable("x")
    variable = space.variables["x"]

    getattr(space, f"set_{bound_name}")("x", 1.0)

    assert space.variables["x"] is not variable
    assert_equal(getattr(space.variables["x"], bound_name), array([1.0]))
    assert_equal(getattr(variable, bound_name), array([stale_bound]))


@pytest.mark.parametrize("bound_name", ["lower_bound", "upper_bound"])
def test_variables_view_gives_read_only_bounds(space, bound_name) -> None:
    """Check that a bound read through the view cannot be mutated in place."""
    bound = getattr(space.variables["x"], bound_name)

    assert not bound.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        bound[0] = 0.0


@pytest.mark.parametrize(
    "copy_space", [lambda space: space, deepcopy, copy_with_pickle]
)
@pytest.mark.parametrize("bound_name", ["lower_bound", "upper_bound"])
def test_variables_view_gives_unfreezable_bounds(
    bound_name, copy_space, snapshot
) -> None:
    """Check that a bound read through the view cannot be unfrozen."""
    space = DesignSpace()
    space.add_variable("x")
    bound = getattr(copy_space(space).variables["x"], bound_name)

    assert not bound.flags.writeable

    with assert_exception(ValueError, snapshot):
        bound.setflags(write=True)
