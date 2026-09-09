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
"""Tests for the legacy variable kept for the pre-hierarchy pickles."""

from __future__ import annotations

import pickle

import pytest
from numpy import array
from numpy import inf

from gemseo.space.variable import BaseVariable
from gemseo.space.variable import ContinuousVariable
from gemseo.space.variable._legacy import Variable
from gemseo.space.variable.factory import VariableFactory
from tests.space.variable.utils import KINDS


@pytest.mark.parametrize("kind", KINDS)
def test_unpickle_legacy_variable(kind) -> None:
    """Check that a variable pickled before the hierarchy is restored as its kind."""
    legacy = Variable(
        size=2,
        type=kind.type,
        lower_bound=array([0.0, 0.0]),
        upper_bound=array([1.0, 2.0]),
    )

    with pytest.warns(DeprecationWarning, match=kind.__name__):
        restored = pickle.loads(pickle.dumps(legacy))

    assert type(restored) is kind
    assert restored == kind(
        size=2, lower_bound=array([0.0, 0.0]), upper_bound=array([1.0, 2.0])
    )
    # The restored variable has been validated as a new one:
    # its bounds are frozen so that they cannot be mutated in place.
    assert not restored.lower_bound.flags.writeable
    assert not restored.upper_bound.flags.writeable


def test_unpickle_legacy_variable_with_default_fields() -> None:
    """Check that a legacy variable with no field is restored with the defaults."""
    with pytest.warns(
        DeprecationWarning,
        match="The class 'gemseo.space.variable.Variable' is deprecated",
    ):
        restored = pickle.loads(pickle.dumps(Variable()))

    assert restored == ContinuousVariable(size=1, lower_bound=-inf, upper_bound=inf)


def test_legacy_variable_is_not_a_kind() -> None:
    """Check that the legacy variable is out of the hierarchy and of its factory.

    Otherwise the factory would find two classes pinning the float data type.
    """
    assert not issubclass(Variable, BaseVariable)
    assert "Variable" not in VariableFactory().class_names
