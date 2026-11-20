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
"""Tests for the one-hot encoding of a design space."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

import pytest
from numpy import array
from numpy import nan
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array

from gemseo.core.function.array_function import ArrayFunction
from gemseo.enum import VariableType
from gemseo.problem.mdo.sellar import SellarDesignSpace
from gemseo.space import DesignSpace
from gemseo.space.encoding import SpaceOneHotEncoding
from gemseo.util.derivative.check.function import FunctionJacobianChecker
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.util.typing import RealArray


@pytest.fixture
def space() -> DesignSpace:
    """A design space with real, integer, discrete and catalog variables.

    The original coordinates are `[x0, x1, i, d, c, u]`
    and the encoded ones `[x0, x1, i, d0, d1, d2, c0, c1, c2, u0, u1]`;
    `u` is a catalog variable without value.
    """
    space = DesignSpace("space")
    space.add_real_variable("x", 2, lower_bound=0.0, upper_bound=10.0, value=[2.0, 3.0])
    space.add_integer_variable("i", lower_bound=0, upper_bound=10, value=4)
    space.add_discrete_variable("d", [1, 3, 8], value=3)
    space.add_catalog_variable("c", {"density": [1.0, 2.0, 3.0]}, value=2)
    space.add_catalog_variable("u", {"density": [4.0, 5.0]})
    return space


@pytest.fixture
def encoding(space) -> SpaceOneHotEncoding:
    """The encoding."""
    return SpaceOneHotEncoding(space)


def test_encoded_space_variables(space, encoding) -> None:
    """Check the variables of the encoded space, the carried ones being shared."""
    encoded_space = encoding.encoded_space

    assert list(encoded_space) == ["x", "i", "d_weights", "c_weights", "u_weights"]
    assert encoded_space.name == "space"
    variables = encoded_space.variables
    assert [variable.type for variable in variables.values()] == [
        VariableType.REAL,
        VariableType.INTEGER,
        VariableType.INTEGER,
        VariableType.INTEGER,
        VariableType.INTEGER,
    ]
    assert [variable.size for variable in variables.values()] == [2, 1, 3, 3, 2]
    assert variables["x"] is space.variables["x"]
    assert variables["i"] is space.variables["i"]


@pytest.mark.parametrize(
    ("name", "bounds"),
    [
        pytest.param(
            "get_lower_bounds", [0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], id="lower"
        ),
        pytest.param(
            "get_upper_bounds", [10.0, 10.0, 10, 1, 1, 1, 1, 1, 1, 1, 1], id="upper"
        ),
    ],
)
def test_encoded_space_bounds(encoding, name, bounds) -> None:
    """Check the bounds of the encoded space, those of the weights being 0 and 1."""
    encoded_space = encoding.encoded_space

    assert_array_equal(getattr(encoded_space, name)(), array(bounds))


def test_encoded_space_current_value(encoding) -> None:
    """Check that the current value of an encoded variable is its vertex."""
    current_value = encoding.encoded_space._current_value

    assert_array_equal(current_value["x"], [2.0, 3.0])
    assert_array_equal(current_value["i"], [4])
    assert_array_equal(current_value["d_weights"], [0.0, 1.0, 0.0])
    assert_array_equal(current_value["c_weights"], [0.0, 0.0, 1.0])
    assert current_value["u_weights"] is None


def test_encoded_space_leaves_the_original_space_untouched(space, encoding) -> None:
    """Check that building the encoded space leaves the original space untouched."""
    _ = encoding.encoded_space

    assert space.variables["d"].type == VariableType.DISCRETE
    assert_array_equal(space.get_current_value(["d"]), [3.0])


@pytest.mark.parametrize(
    ("encode_discrete", "encode_catalog", "encoded_names", "dimension"),
    [
        pytest.param(True, True, {"d", "c", "u"}, 11, id="both"),
        pytest.param(False, True, {"c", "u"}, 9, id="catalog only"),
        pytest.param(True, False, {"d"}, 8, id="discrete only"),
    ],
)
def test_encoded_kinds(
    space, encode_discrete, encode_catalog, encoded_names, dimension
) -> None:
    """Check the encoded kinds of variables, either of which can be left unencoded."""
    encoding = SpaceOneHotEncoding(
        space,
        encode_discrete=encode_discrete,
        encode_catalog=encode_catalog,
    )
    assert encoding.encoded_names == encoded_names
    assert encoding.encoded_space.dimension == dimension


def _create_real_space() -> DesignSpace:
    """Create a design space with a real variable.

    Returns:
        The design space.
    """
    space = DesignSpace()
    space.add_real_variable("x", value=1.0)
    return space


def _create_categorical_space() -> DesignSpace:
    """Create a design space with a categorical variable.

    Returns:
        The design space.
    """
    space = DesignSpace()
    space.add_categorical_variable("k", ("a", "b", "c"), value="b")
    return space


def _create_catalog_space() -> DesignSpace:
    """Create a design space with a catalog variable.

    Returns:
        The design space.
    """
    space = DesignSpace()
    space.add_catalog_variable("v", {"density": [1.0, 3.0, 8.0]}, value=1)
    return space


@pytest.mark.parametrize(
    ("space", "options"),
    [
        pytest.param(_create_real_space(), {}, id="real"),
        pytest.param(_create_categorical_space(), {}, id="categorical"),
        pytest.param(
            _create_catalog_space(),
            {"encode_catalog": False},
            id="catalog not encoded",
        ),
    ],
)
def test_nothing_encoded(space, options) -> None:
    """Check that the encoded space is the original one when nothing is encoded."""
    encoding = SpaceOneHotEncoding(space, **options)

    assert encoding.encoded_names == frozenset()
    assert encoding.encoded_space is space


@pytest.fixture
def real_space() -> DesignSpace:
    """A design space with a real variable."""
    return _create_real_space()


@pytest.mark.parametrize("name", ["encode", "decode"])
def test_identity_when_nothing_is_encoded(real_space, name) -> None:
    """Check that a map returns its argument when nothing is encoded."""
    value = array([1.0])

    assert getattr(SpaceOneHotEncoding(real_space), name)(value) is value


def test_weights_name_taken(snapshot) -> None:
    """Check that the weights cannot take the name of a variable carried over."""
    space = DesignSpace()
    space.add_discrete_variable("d", [1, 3, 8], value=3)
    space.add_real_variable("d_weights")
    with assert_exception(ValueError, snapshot):
        SpaceOneHotEncoding(space)


def test_encoded_space_of_subclass() -> None:
    """Check the encoded space of a subclass with another constructor signature."""
    space = SellarDesignSpace(n=1, add_couplings=True)
    space.add_discrete_variable("c", [1, 3, 8], value=3)
    encoded_space = SpaceOneHotEncoding(space).encoded_space

    assert type(encoded_space) is DesignSpace
    assert list(encoded_space) == [*list(space)[:-1], "c_weights"]
    assert_array_equal(encoded_space.get_current_value(["c_weights"]), [0, 1, 0])


@pytest.mark.parametrize(
    "enable", [False, True], ids=["normalization disabled", "normalization enabled"]
)
def test_encoded_space_integer_normalization(space, enable) -> None:
    """Check that the normalization of the integer variables is kept."""
    space.enable_integer_variables_normalization = enable
    encoded_space = SpaceOneHotEncoding(space).encoded_space

    assert encoded_space.enable_integer_variables_normalization is enable


@pytest.fixture
def encoding_of_changed_space() -> SpaceOneHotEncoding:
    """An encoding of a space to which a variable is added after the encoding.

    The encoding is built on `[x, d]`, with the encoded coordinates `[x, d0, d1, d2]`,
    then the discrete variable `e` is added to the space.
    """
    space = DesignSpace()
    space.add_real_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_discrete_variable("d", [1.0, 2.0, 3.0], value=2.0)
    encoding = SpaceOneHotEncoding(space)
    space.add_discrete_variable("e", [1.0, 2.0], value=1.0)
    return encoding


def test_encoded_space_ignores_a_later_variable(encoding_of_changed_space) -> None:
    """Check that the encoded space ignores a variable added after the encoding."""
    encoded_space = encoding_of_changed_space.encoded_space

    assert list(encoded_space) == ["x", "d_weights"]


def test_encode_ignores_a_later_variable(encoding_of_changed_space) -> None:
    """Check that the encoding ignores a variable added after the encoding."""
    encoded_value = encoding_of_changed_space.encode(array([0.5, 2.0]))

    assert_array_equal(encoded_value, [0.5, 0, 1, 0])


def test_constraints_ignore_a_later_variable(encoding_of_changed_space) -> None:
    """Check that the constraints ignore a variable added after the encoding."""
    constraint = encoding_of_changed_space.create_constraints()[0]

    assert constraint.jac(array([0.5, 0, 1, 0])).shape == (1, 4)


_original_value: Final[list[float]] = [2.0, 3.0, 4.0, 8.0, 1.0, 0.0]
"""A value of the original space."""

_encoded_value: Final[list[float]] = [2.0, 3.0, 4.0, 0, 0, 1, 0, 1, 0, 1, 0]
"""The image of `_original_value` in the encoded space."""

_values: Final = pytest.mark.parametrize(
    ("original_value", "encoded_value"),
    [
        (_original_value, _encoded_value),
        (
            [_original_value, [0.0, 1.0, 5.0, 1.0, 2.0, 1.0]],
            [_encoded_value, [0.0, 1.0, 5.0, 1, 0, 0, 0, 0, 1, 0, 1]],
        ),
    ],
    ids=["single", "batch"],
)
"""The pairs of a value of the original space and its image in the encoded space."""


@_values
def test_encode(encoding, original_value, encoded_value) -> None:
    """Check that a value is encoded by the vertices of its choices."""
    assert_array_equal(encoding.encode(array(original_value)), encoded_value)


@_values
def test_decode(encoding, original_value, encoded_value) -> None:
    """Check that vertices are decoded as their choices."""
    assert_array_equal(encoding.decode(array(encoded_value)), original_value)


def test_decoding_is_the_largest_weight(encoding) -> None:
    """Check the decoding of weights that are not a vertex, ties included."""
    encoded_value = array([2.0, 3.0, 4.0, 0.2, 0.7, 0.6, 0.5, 0.5, 0.1, 0.0, 0.0])

    # The weights of `d` peak at the second choice,
    # those of `c` tie between the first two alternatives and `u` has only ties.
    assert_array_equal(
        encoding.decode(encoded_value),
        array([2.0, 3.0, 4.0, 3.0, 0.0, 0.0]),
    )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([2.0, 3.0, 4.0, 2.0, 1.4, 7.0], id="not choices"),
        pytest.param([2.0, 3.0, 4.0, nan, 1.0, 0.0], id="nan"),
        pytest.param(
            [[2.0, 3.0, 4.0, 8.0, 1.0, 0.0], [2.0, 3.0, 4.0, 2.0, 1.0, 0.0]],
            id="batch",
        ),
    ],
)
def test_encode_outside_domain(encoding, snapshot, value) -> None:
    """Check that a value that is not a choice cannot be encoded."""
    with assert_exception(ValueError, snapshot):
        encoding.encode(array(value))


@pytest.mark.parametrize(
    "indices",
    [
        pytest.param([4], id="discrete"),
        pytest.param([7], id="catalog"),
        pytest.param([4, 7], id="discrete and catalog"),
    ],
)
def test_decode_nan_weight(encoding, snapshot, indices) -> None:
    """Check that weights with a NaN component cannot be decoded."""
    value = array(_encoded_value)
    value[indices] = nan
    with assert_exception(ValueError, snapshot):
        encoding.decode(value)


def test_constraint_names_and_types(encoding) -> None:
    """Check the names, types and input names of the constraints.

    The constraints come in the order of the space
    and their inputs are named after the components of the encoded space.
    """
    constraints = encoding.create_constraints()

    assert [(c.name, c.f_type) for c in constraints] == [
        ("d_one_hot", ArrayFunction.FunctionType.EQ),
        ("c_one_hot", ArrayFunction.FunctionType.EQ),
        ("u_one_hot", ArrayFunction.FunctionType.EQ),
    ]
    input_names = encoding.encoded_space.get_indexed_variable_names()
    assert all(c.input_names == input_names for c in constraints)


@pytest.mark.xfail(
    strict=True,
    reason="#2150: LinearFunction prints a negative constant with two minus signs.",
)
def test_constraint_expression_names_the_weights(encoding) -> None:
    """Check that the expression of a constraint sums the weights."""
    constraint = encoding.create_constraints()[0]

    assert constraint.expr == "d_weights[0] + d_weights[1] + d_weights[2] - 1.00e+00"


def test_no_constraint(real_space) -> None:
    """Check that there is no constraint when nothing is encoded."""
    assert SpaceOneHotEncoding(real_space).create_constraints() == ()


@pytest.fixture
def constraint(encoding) -> ArrayFunction:
    """The constraint of the discrete variable `d`."""
    return encoding.create_constraints()[0]


def _create_encoded_value(weights: Sequence[float]) -> RealArray:
    """Create a value of the encoded space from the weights of `d`.

    Args:
        weights: The weights of `d`.

    Returns:
        The value of the encoded space,
        with the vertices of the first choices for the other encoded variables.
    """
    return array([2.0, 3.0, 4.0, *weights, 1, 0, 0, 1, 0])


@pytest.mark.parametrize(
    ("weights", "value"),
    [
        pytest.param([0, 1, 0], 0, id="one-hot"),
        pytest.param([0, 0, 0], -1, id="zero"),
        pytest.param([1, 1, 0], 1, id="two ones"),
    ],
)
def test_constraint_value(constraint, weights, value) -> None:
    """Check the value of the constraint, which reads only the weights of `d`."""
    assert_array_equal(constraint.evaluate(_create_encoded_value(weights)), [value])


def test_constraint_jacobian_is_sparse(constraint) -> None:
    """Check that the Jacobian of the constraint is a sparse array."""
    value = _create_encoded_value([0.2, 0.3, 0.5])

    assert isinstance(constraint.jac(value), csr_array)


def test_constraint_jacobian(constraint) -> None:
    """Check the Jacobian of the constraint against finite differences."""
    value = _create_encoded_value([0.2, 0.3, 0.5])

    assert FunctionJacobianChecker(constraint).check(value, atol=1e-6)
