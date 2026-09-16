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
"""Tests for the RandomVariables registry."""

from __future__ import annotations

import pytest
from openturns import NormalCopula

from gemseo.space._core.variables import UnknownVariableError
from gemseo.space._core.variables import Variables
from gemseo.space._random.variables import RandomVariables
from gemseo.space.variable.random import RandomVariable
from gemseo.uncertainty.distribution.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.uncertainty.distribution.scipy.normal_settings import (
    SPNormalDistribution_Settings,
)
from gemseo.util.testing.helper import assert_exception

OT_NORMAL = OTNormalDistribution_Settings()
OT_UNIFORM = OTUniformDistribution_Settings()
SP_NORMAL = SPNormalDistribution_Settings()


@pytest.fixture
def registry() -> RandomVariables:
    """A registry with two random variables based on OpenTURNS."""
    registry = RandomVariables()
    registry["x"] = RandomVariable(distribution_settings=(OT_NORMAL, OT_NORMAL))
    registry["y"] = RandomVariable(distribution_settings=(OT_NORMAL,))
    return registry


def test_is_a_registry(registry) -> None:
    """Check that the registry is a registry of variables."""
    assert isinstance(registry, Variables)
    assert list(registry) == ["x", "y"]
    assert registry.size == 3
    assert registry.name_to_indices["y"] == range(2, 3)


def test_joint_distribution(registry) -> None:
    """Check that the joint distribution covers all the components."""
    assert registry.distribution.dimension == 3


def test_empty_registry() -> None:
    """Check that the joint distribution of an empty registry is undefined."""
    assert RandomVariables().distribution is None


def test_single_version_bump_per_mutation(registry) -> None:
    """Check that a mutation bumps the version exactly once."""
    version = registry.version
    registry["z"] = RandomVariable(distribution_settings=(OT_NORMAL,))
    assert registry.version == version + 1
    del registry["z"]
    assert registry.version == version + 2
    registry.rename("x", "u")
    assert registry.version == version + 3


def test_setitem_replaces(registry) -> None:
    """Check that assigning an existing name replaces the random variable."""
    registry["x"] = RandomVariable(distribution_settings=(OT_NORMAL,))
    assert list(registry) == ["x", "y"]
    assert registry["x"].size == 1
    assert registry.size == 2
    assert registry.distribution.dimension == 2


def test_delitem_rebuilds_the_joint_distribution(registry) -> None:
    """Check that deleting a random variable rebuilds the joint distribution."""
    del registry["x"]
    assert registry.distribution.dimension == 1


def test_mixed_libraries(registry, snapshot) -> None:
    """Check that the registry cannot mix distribution libraries."""
    with assert_exception(ValueError, snapshot):
        registry["z"] = RandomVariable(distribution_settings=(SP_NORMAL,))


def test_add_copula(registry) -> None:
    """Check that a copula is stored and taken into account."""
    registry.add_copula(("x", "y"), NormalCopula(3))
    assert registry.copulas == ((("x", "y"), registry.copulas[0][1]),)
    assert registry.distribution.dimension == 3


def test_add_copula_empty_names(registry, snapshot) -> None:
    """Check the error raised when no random variable name is passed.

    A copula with no name must be rejected before anything is recorded,
    so that the registry is not left with a copula that does not validate
    against any variable.
    """
    with assert_exception(ValueError, snapshot):
        registry.add_copula((), NormalCopula(2))

    assert registry.copulas == ()
    assert registry.distribution.dimension == 3

    # A later mutation succeeds, instead of failing because of a bogus copula.
    registry["z"] = RandomVariable(distribution_settings=(OT_NORMAL,))
    assert list(registry) == ["x", "y", "z"]
    assert registry.distribution.dimension == 4


def test_add_copula_unknown_name(registry, snapshot) -> None:
    """Check the error raised when the random variable does not exist."""
    with assert_exception(ValueError, snapshot):
        registry.add_copula("z", NormalCopula(2))


def test_add_copula_already_has_copula(registry, snapshot) -> None:
    """Check the error raised when the random variable already has a copula."""
    registry.add_copula(("x", "y"), NormalCopula(3))
    with assert_exception(ValueError, snapshot):
        registry.add_copula("x", NormalCopula(3))


def test_add_copula_without_dependency_support(snapshot) -> None:
    """Check the error raised when the library does not support dependency."""
    registry = RandomVariables()
    registry["x"] = RandomVariable(distribution_settings=(SP_NORMAL,))
    with assert_exception(ValueError, snapshot):
        registry.add_copula("x", NormalCopula(2))


def test_add_copula_dimension_mismatch(registry, snapshot) -> None:
    """Check the error raised when the copula dimension does not match.

    A copula whose dimension does not match that of the covered variables
    must leave the registry unchanged,
    so that later mutations are not affected by the failed call.
    """
    with assert_exception(ValueError, snapshot):
        registry.add_copula("y", NormalCopula(2))

    assert registry.copulas == ()
    assert registry.distribution.dimension == 3

    # A later mutation succeeds, instead of re-raising the same error.
    registry["z"] = RandomVariable(distribution_settings=(OT_NORMAL,))
    assert list(registry) == ["x", "y", "z"]
    del registry["x"]
    assert list(registry) == ["y", "z"]
    registry.rename("y", "u")
    assert list(registry) == ["u", "z"]


def test_setitem_failure_leaves_registry_unchanged(registry, snapshot) -> None:
    """Check that a failure to rebuild the distribution leaves the registry as is.

    Replacing "y" with a random variable of a different size invalidates the
    copula covering "x" and "y", whose dimension no longer matches; the
    registry, including "y" itself, must be left unchanged by the failure.
    """
    registry.add_copula(("x", "y"), NormalCopula(3))
    library_name = registry._RandomVariables__distribution_library_name
    distribution = registry.distribution

    with assert_exception(ValueError, snapshot):
        registry["y"] = RandomVariable(distribution_settings=(OT_NORMAL, OT_NORMAL))

    assert registry["y"].size == 1
    assert registry.copulas == ((("x", "y"), registry.copulas[0][1]),)
    assert registry.distribution is distribution
    assert registry._RandomVariables__distribution_library_name == library_name

    # A later mutation succeeds, instead of re-raising the same error.
    registry["z"] = RandomVariable(distribution_settings=(OT_NORMAL,))
    assert list(registry) == ["x", "y", "z"]
    assert registry.distribution.dimension == 4


def test_delitem_removes_the_copulas(registry) -> None:
    """Check that deleting a random variable removes its copulas."""
    registry.add_copula(("x", "y"), NormalCopula(3))
    del registry["x"]
    assert registry.copulas == ()


def test_delitem_keeps_the_other_copula_and_remaps_its_indices() -> None:
    """Check that a surviving copula's indices are remapped after a deletion.

    With two copulas, deleting a variable covered by only one of them must
    remove that copula while keeping the other, with its indices remapped to
    the positions of its variables in the remaining registry.
    """
    registry = RandomVariables()
    registry["a"] = RandomVariable(distribution_settings=(OT_NORMAL,))
    registry["b"] = RandomVariable(distribution_settings=(OT_NORMAL,))
    registry["c"] = RandomVariable(distribution_settings=(OT_NORMAL,))
    registry["d"] = RandomVariable(distribution_settings=(OT_NORMAL,))
    registry.add_copula(("a", "c"), NormalCopula(2))
    registry.add_copula(("b", "d"), NormalCopula(2))

    del registry["a"]

    assert list(registry) == ["b", "c", "d"]
    assert registry.copulas == ((("b", "d"), registry.copulas[0][1]),)
    assert repr(registry.distribution) == (
        "OTJointDistribution("
        "Normal(mu=0.0, sigma=1.0), "
        "Normal(mu=0.0, sigma=1.0), "
        "Normal(mu=0.0, sigma=1.0); "
        "MarginalDistribution("
        "distribution=BlockIndependentCopula("
        "NormalCopula(R = [[ 1 0 ]\n [ 0 1 ]]), "
        "IndependentCopula(dimension = 1)), "
        "indices=[0,2,1]))"
    )


def test_delitem_last_variable(registry) -> None:
    """Check that the joint distribution is undefined once the registry is empty."""
    del registry["x"]
    del registry["y"]
    assert registry.distribution is None


def test_rename_renames_the_copulas(registry) -> None:
    """Check that renaming a random variable renames it in its copula."""
    registry.add_copula(("x", "y"), NormalCopula(3))
    registry.rename("x", "u")
    assert registry.copulas[0][0] == ("u", "y")


def test_rename_variable_without_copula(registry) -> None:
    """Check that renaming a random variable leaves the copulas of the others alone."""
    registry["z"] = RandomVariable(distribution_settings=(OT_NORMAL,))
    registry.add_copula(("x", "y"), NormalCopula(3))
    registry.rename("z", "w")
    assert registry.copulas == ((("x", "y"), registry.copulas[0][1]),)


def test_rename_collision(registry, snapshot) -> None:
    """Check that renaming to an existing name raises and leaves the copulas alone."""
    registry.add_copula(("x", "y"), NormalCopula(3))
    with assert_exception(ValueError, snapshot):
        registry.rename("x", "y")
    assert list(registry) == ["x", "y"]
    assert registry.copulas == ((("x", "y"), registry.copulas[0][1]),)


def test_rename_same_name_is_noop(registry) -> None:
    """Check that renaming a random variable to its own name is a no-op."""
    registry.add_copula(("x", "y"), NormalCopula(3))
    registry.rename("x", "x")
    assert list(registry) == ["x", "y"]
    assert registry.copulas == ((("x", "y"), registry.copulas[0][1]),)


def test_filter_components(registry) -> None:
    """Check that filtering rebuilds the random variable from the kept components."""
    registry["x"] = RandomVariable(
        distribution_settings=(OT_NORMAL, OT_UNIFORM, OT_NORMAL)
    )
    version = registry.version
    registry.filter_components("x", [0, 2])
    assert registry["x"].distribution_settings == (OT_NORMAL, OT_NORMAL)
    assert registry["x"].size == 2
    assert registry.size == 3
    assert registry.name_to_indices["y"] == range(2, 3)
    assert registry.distribution.dimension == 3
    assert registry.version == version + 1


def test_filter_components_removes_the_copulas(registry) -> None:
    """Check that filtering a random variable removes the copula covering it."""
    registry.add_copula(("x", "y"), NormalCopula(3))
    registry.filter_components("x", [0])
    assert registry.copulas == ()
    assert registry.distribution.dimension == 2


def test_filter_components_keeping_all_keeps_the_copulas(registry) -> None:
    """Check that keeping every component leaves the copula covering it alone."""
    registry.add_copula(("x", "y"), NormalCopula(3))
    version = registry.version
    distribution = registry.distribution

    registry.filter_components("x", [0, 1])

    assert registry["x"].size == 2
    assert registry.copulas == ((("x", "y"), registry.copulas[0][1]),)
    assert registry.distribution is distribution
    assert registry.version == version


def test_filter_components_keeps_the_other_copulas(registry) -> None:
    """Check that filtering a random variable leaves the copulas of the others alone."""
    registry["z"] = RandomVariable(distribution_settings=(OT_NORMAL,))
    registry.add_copula(("y", "z"), NormalCopula(2))
    registry.filter_components("x", [0])
    assert registry.copulas == ((("y", "z"), registry.copulas[0][1]),)


def test_filter_components_empty(registry, snapshot) -> None:
    """Check the error raised when no component is to be kept."""
    with assert_exception(ValueError, snapshot):
        registry.filter_components("x", [])


def test_filter_components_unknown_name(registry, snapshot) -> None:
    """Check the error raised when the random variable does not exist."""
    with assert_exception(UnknownVariableError, snapshot):
        registry.filter_components("z", [0])
