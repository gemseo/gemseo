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
"""Tests for the RandomVariable model."""

from __future__ import annotations

import pickle
from copy import copy
from copy import deepcopy

import pytest
from numpy import inf
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from gemseo.space.variable import BaseDeterministicVariable
from gemseo.space.variable import BaseNumericVariable
from gemseo.space.variable import BaseVariable
from gemseo.space.variable import DataType
from gemseo.space.variable import RealVariable
from gemseo.space.variable import Variable
from gemseo.space.variable.random import RandomVariable
from gemseo.uncertainty.distribution.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.distribution.scipy.normal_settings import (
    SPNormalDistribution_Settings,
)
from gemseo.uncertainty.distribution.scipy.uniform_settings import (
    SPUniformDistribution_Settings,
)
from gemseo.util.testing.helper import assert_exception

NORMAL = SPNormalDistribution_Settings(mu=1.0, sigma=2.0)
UNIFORM = SPUniformDistribution_Settings(minimum=0.0, maximum=1.0)


@pytest.fixture
def variable() -> RandomVariable:
    """A random variable of size 2 with different marginals."""
    return RandomVariable(distribution_settings=(NORMAL, UNIFORM))


def test_derived_fields(variable) -> None:
    """Check that size, type and bounds derive from the distribution."""
    assert variable.size == 2
    assert variable.type == DataType.REAL
    assert_array_equal(variable.lower_bound, [-inf, 0.0])
    assert_array_equal(variable.upper_bound, [inf, 1.0])
    assert variable.distribution.dimension == 2
    assert variable.distribution_settings == (NORMAL, UNIFORM)


def test_size_is_the_number_of_marginals(variable) -> None:
    """Check that the size is read from the settings, not from the distribution."""
    assert variable.size == variable.distribution.dimension

    new_variable = RandomVariable(distribution_settings=(NORMAL, UNIFORM))
    assert new_variable.size == 2
    # The joint probability distribution has not been built to compute the size.
    assert set(new_variable.__dict__) == {"distribution_settings"}


def test_bounds_are_read_only(variable, snapshot) -> None:
    """Check that the bounds cannot be mutated through the variable."""
    for bound in (variable.lower_bound, variable.upper_bound):
        assert not bound.flags.writeable
        with assert_exception(ValueError, snapshot):
            bound[0] = 0.0

    with assert_exception(ValueError, snapshot):
        variable.lower_bound.setflags(write=True)


def test_bounds_leave_the_distribution_alone(variable) -> None:
    """Check that reading a bound does not touch the distribution.

    The distribution is shared with every copy of the variable,
    so the variable copies its bound arrays instead of freezing them in place.
    """
    distribution = variable.distribution

    assert_array_equal(variable.lower_bound, distribution.math_lower_bound)
    assert_array_equal(variable.upper_bound, distribution.math_upper_bound)
    assert distribution.math_lower_bound.flags.writeable
    assert distribution.math_upper_bound.flags.writeable


def test_bounds_are_the_support(variable) -> None:
    """Check that the bounds of the variable are the limits of the support."""
    support = variable.distribution.support
    assert_array_equal(support[:, 0], variable.lower_bound)
    assert_array_equal(support[:, 1], variable.upper_bound)


def test_is_a_variable(variable) -> None:
    """Check that a random variable is a variable whose components are numbers."""
    assert isinstance(variable, BaseVariable)
    assert isinstance(variable, BaseNumericVariable)
    assert not isinstance(variable, BaseDeterministicVariable)
    assert not isinstance(variable, Variable)


@pytest.mark.parametrize(
    "name",
    [
        "distribution_settings",
        "size",
        "lower_bound",
        "upper_bound",
        "distribution",
    ],
)
def test_immutability(variable, name, snapshot) -> None:
    """Check that neither the settings nor the derived attributes can be set."""
    with assert_exception(ValidationError, snapshot):
        setattr(variable, name, 3)


def test_type_is_a_class_variable(variable, snapshot) -> None:
    """Check that the type is shared by the class and cannot be set on an instance."""
    assert RandomVariable.type == DataType.REAL
    with assert_exception(AttributeError, snapshot):
        variable.type = DataType.INTEGER


@pytest.mark.parametrize(
    "field", ["size", "type", "lower_bound", "upper_bound", "distribution", "foo"]
)
def test_derived_fields_cannot_be_passed(field, snapshot) -> None:
    """Check that the settings are the only argument of a random variable."""
    with assert_exception(ValidationError, snapshot):
        RandomVariable(distribution_settings=(NORMAL,), **{field: 3})


def test_missing_settings(snapshot) -> None:
    """Check that the settings of the marginal distributions are required."""
    with assert_exception(ValidationError, snapshot):
        RandomVariable()


def test_empty_settings(snapshot) -> None:
    """Check that a random variable has at least one component."""
    with assert_exception(ValidationError, snapshot):
        RandomVariable(distribution_settings=())


def test_model_copy(variable) -> None:
    """Check that only the settings can be updated by model_copy."""
    assert variable.model_copy() is variable

    new_variable = variable.model_copy(update={"distribution_settings": (UNIFORM,)})
    assert new_variable.distribution_settings == (UNIFORM,)
    assert new_variable.size == 1
    assert_array_equal(new_variable.lower_bound, [0.0])
    assert new_variable.distribution.dimension == 1
    # The original variable is left alone.
    assert variable.distribution_settings == (NORMAL, UNIFORM)
    assert variable.size == 2


def test_model_copy_with_derived_field(variable, snapshot) -> None:
    """Check that model_copy cannot update a field deriving from the settings."""
    with assert_exception(ValidationError, snapshot):
        variable.model_copy(update={"size": 5})


def test_filter_components_keeping_all(variable) -> None:
    """Check that keeping every component in order returns the variable itself."""
    assert variable.filter_components(range(variable.size)) is variable


def test_filter_components(variable) -> None:
    """Check that keeping some components returns a variable built from them."""
    filtered_variable = variable.filter_components([1])
    assert filtered_variable is not variable
    assert filtered_variable.distribution_settings == (UNIFORM,)


def test_mixed_libraries(snapshot) -> None:
    """Check that a random variable cannot mix libraries."""
    with assert_exception(ValueError, snapshot):
        RandomVariable(distribution_settings=(NORMAL, OTNormalDistribution_Settings()))


def test_equality(variable) -> None:
    """Check the equality of two random variables."""
    assert variable == RandomVariable(distribution_settings=(NORMAL, UNIFORM))
    assert variable != RandomVariable(distribution_settings=(UNIFORM, NORMAL))
    assert variable != RandomVariable(distribution_settings=(NORMAL,))
    assert variable != RealVariable(size=2)


def test_copy_is_shared(variable) -> None:
    """Check that a copy of a random variable shares its distribution.

    A random variable is frozen and its distribution is only read,
    so the copy can be the variable itself, as for any variable.
    """
    assert copy(variable) is variable
    assert deepcopy(variable) is variable
    assert variable.distribution is deepcopy(variable).distribution


def test_pickle(variable) -> None:
    """Check that the distribution is rebuilt when unpickling."""
    restored = pickle.loads(pickle.dumps(variable))
    # The distribution is not pickled but built again on demand.
    assert set(restored.__dict__) == {"distribution_settings"}
    assert restored == variable
    assert restored.size == 2
    assert_almost_equal(restored.distribution.mean, variable.distribution.mean)
    assert_array_equal(restored.lower_bound, variable.lower_bound)
    assert restored.distribution is not variable.distribution
