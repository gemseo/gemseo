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
"""Tests for the RandomSpace class."""

from __future__ import annotations

import pickle
import sys
from types import ModuleType

import pytest
from numpy import array
from numpy import corrcoef
from numpy import inf
from numpy import isnan
from numpy import unique
from numpy.random import default_rng
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from openturns import CorrelationMatrix
from openturns import NormalCopula
from openturns import RandomGenerator

from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.doe.factory import doe_library_factory
from gemseo.doe.openturns.settings.ot_monte_carlo import OT_MONTE_CARLO_Settings
from gemseo.scenario.evaluation import EvaluationScenario
from gemseo.space._core.variables import UnknownVariableError
from gemseo.space.base import BaseVariableSpace
from gemseo.space.design import DesignSpace
from gemseo.space.random import RandomSpace
from gemseo.space.variable import DataType
from gemseo.space.variable.random import RandomVariable
from gemseo.uncertainty.distribution.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.finite_discrete_settings import (
    OTFiniteDiscreteDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.uncertainty.distribution.scipy.distribution_settings import (
    SPDistribution_Settings,
)
from gemseo.uncertainty.distribution.scipy.normal_settings import (
    SPNormalDistribution_Settings,
)
from gemseo.uncertainty.sensitivity.correlation import CorrelationAnalysis
from gemseo.util.testing.helper import assert_exception

OT_NORMAL = OTNormalDistribution_Settings(mu=1.0, sigma=2.0)
OT_UNIFORM = OTUniformDistribution_Settings(minimum=0.0, maximum=1.0)
SP_NORMAL = SPNormalDistribution_Settings()


@pytest.fixture
def space() -> RandomSpace:
    """A random space with a random vector and a scalar random variable."""
    space = RandomSpace("u")
    space.add_variable("x", *[OT_UNIFORM] * 2)
    space.add_variable("y", OT_NORMAL)
    return space


@pytest.fixture
def scalar_space() -> RandomSpace:
    """A random space with two scalar random variables."""
    space = RandomSpace()
    space.add_variable("x", OT_UNIFORM)
    space.add_variable("y", OT_NORMAL)
    return space


def _create_correlated_copula() -> NormalCopula:
    """Create a copula correlating two components.

    Returns:
        The copula.
    """
    correlation = CorrelationMatrix(3)
    correlation[0, 2] = 0.8
    return NormalCopula(correlation)


def test_is_a_variable_space(space) -> None:
    """Check that a random space is a space of variables."""
    assert isinstance(space, BaseVariableSpace)
    assert not isinstance(space, DesignSpace)


def test_container_behavior(space) -> None:
    """Check the container behavior inherited from the base class."""
    assert space.name == "u"
    assert list(space.variables) == ["x", "y"]
    assert space.dimension == 3
    assert len(space) == 2
    assert list(space) == ["x", "y"]
    assert "x" in space
    assert "z" not in space
    assert space.variables["x"].size == 2
    assert space.variables["x"].type == DataType.REAL
    assert space.variables["y"].size == 1
    assert space.variables["y"].type == DataType.REAL
    assert space.get_indexed_variable_names() == ["x[0]", "x[1]", "y"]
    assert_array_equal(space.get_variables_indexes(["y"]), [2])


@pytest.mark.parametrize(
    "name",
    [
        "variable_names",
        "variable_sizes",
        "variable_types",
        "name_to_indices",
        "has_integer_variables",
        "get_size",
        "get_type",
    ],
)
@pytest.mark.parametrize("space_class", [DesignSpace, RandomSpace])
def test_no_accessors(space_class, name) -> None:
    """Check that a space does not duplicate its registry of variables."""
    assert not hasattr(space_class(), name)


def test_conversions(space) -> None:
    """Check the array/dictionary conversions inherited from the base class."""
    full_value = array([1.0, 2.0, 3.0])
    name_to_value = space.convert_array_to_dict(full_value)
    assert_array_equal(name_to_value["x"], [1.0, 2.0])
    assert_array_equal(space.convert_dict_to_array(name_to_value), full_value)


def test_registry(space) -> None:
    """Check the public registry of random variables."""
    registry = space.variables
    assert registry.distribution.dimension == 3
    assert registry["x"].distribution.dimension == 2
    assert registry["x"].distribution_settings == (OT_UNIFORM, OT_UNIFORM)
    assert_array_equal(registry["x"].distribution.support, [[0.0, 1.0], [0.0, 1.0]])
    assert_array_equal(registry["y"].distribution.support, [[-inf, inf]])
    assert registry["y"].distribution.range.shape == (1, 2)


def test_registry_bounds_are_the_support(space) -> None:
    """Check that the bounds of an entry are the limits of the support."""
    for random_variable in space.variables.values():
        support = random_variable.distribution.support
        assert_array_equal(support[:, 0], random_variable.lower_bound)
        assert_array_equal(support[:, 1], random_variable.upper_bound)


def test_add_variable_with_heterogeneous_marginals() -> None:
    """Check a random vector whose components have different distributions."""
    space = RandomSpace()
    space.add_variable("x", OT_UNIFORM, OT_NORMAL)
    assert space.variables["x"].size == 2
    marginals = space.variables["x"].distribution.marginals
    assert marginals[0].mean == pytest.approx(0.5)
    assert marginals[1].mean == pytest.approx(1.0)


def test_add_variable_with_iid_components() -> None:
    """Check a random variable whose components are identically distributed."""
    space = RandomSpace()
    space.add_variable("x", *[OT_UNIFORM] * 3)
    assert space.variables["x"].size == 3
    marginals = space.variables["x"].distribution.marginals
    assert [marginal.mean for marginal in marginals] == pytest.approx([0.5] * 3)


def test_add_variable_duplicate_name(space, snapshot) -> None:
    """Check the error raised when the random variable already exists."""
    with assert_exception(ValueError, snapshot):
        space.add_variable("x", OT_NORMAL)


def test_add_variable_without_settings(space, snapshot) -> None:
    """Check the error raised when no distribution settings are passed."""
    with assert_exception(ValueError, snapshot):
        space.add_variable("z")


def test_add_variable_mixed_libraries(space, snapshot) -> None:
    """Check the error raised when mixing distribution libraries."""
    with assert_exception(ValueError, snapshot):
        space.add_variable("z", SP_NORMAL)


def test_registry_is_read_only(space) -> None:
    """Check that a random variable cannot be replaced through the registry."""
    with pytest.raises(TypeError, match="does not support item assignment"):
        space.variables["x"] = RandomVariable(distribution_settings=(OT_NORMAL,))


def test_replace_variable(space) -> None:
    """Check that a random variable is replaced by removing and adding it again."""
    space.remove_variable("x")
    space.add_variable("x", OT_NORMAL)
    assert list(space.variables) == ["y", "x"]
    assert space.variables["x"].size == 1
    assert space.dimension == 2


def test_add_copula(space) -> None:
    """Check that a copula correlates the samples."""
    space.add_copula(("x", "y"), NormalCopula(3))
    assert space.variables.copulas[0][0] == ("x", "y")
    assert space.variables.distribution.dimension == 3


def test_add_copula_single_name(space) -> None:
    """Check that a copula covering a single random vector takes its name alone."""
    space.add_copula("x", NormalCopula(2))
    assert space.variables.copulas[0][0] == ("x",)
    assert space.variables.distribution.dimension == 3


def test_add_copula_names_iterator(space) -> None:
    """Check that the names of a copula can be passed as an iterator.

    The names are read for the checks and stored with the copula,
    so an iterator must not be exhausted in between.
    """
    space.add_copula((name for name in ("x", "y")), NormalCopula(3))
    assert space.variables.copulas[0][0] == ("x", "y")
    assert space.variables.distribution.dimension == 3


def test_add_copula_empty_names(space, snapshot) -> None:
    """Check the error raised when no random variable name is passed.

    A copula with no name must be rejected before anything is recorded,
    so that the space is not left with variables and a bogus copula that
    breaks every later mutation.
    """
    with assert_exception(ValueError, snapshot):
        space.add_copula((), NormalCopula(2))

    assert space.variables.copulas == ()
    assert space.variables.distribution.dimension == 3

    # A later mutation succeeds, instead of failing because of a bogus copula.
    space.add_variable("z", OT_NORMAL)
    assert list(space.variables) == ["x", "y", "z"]


def test_add_copula_errors(space, snapshot) -> None:
    """Check the error raised when the random variable does not exist."""
    with assert_exception(ValueError, snapshot):
        space.add_copula("z", NormalCopula(2))


def test_add_copula_already_has_copula(space, snapshot) -> None:
    """Check the error raised when the random variable already has a copula."""
    space.add_copula(("x", "y"), NormalCopula(3))
    with assert_exception(ValueError, snapshot):
        space.add_copula("y", NormalCopula(2))


def test_add_copula_without_dependency_support(snapshot) -> None:
    """Check the error raised when the library does not support dependency."""
    space = RandomSpace()
    space.add_variable("x", SP_NORMAL)
    with assert_exception(ValueError, snapshot):
        space.add_copula("x", NormalCopula(2))


def test_add_copula_dimension_mismatch(scalar_space, snapshot) -> None:
    """Check the error raised when the copula dimension does not match.

    A copula whose dimension does not match that of the covered variables
    must leave the space unchanged,
    so that later mutations are not affected by the failed call.
    """
    with assert_exception(ValueError, snapshot):
        scalar_space.add_copula("x", NormalCopula(2))

    assert scalar_space.variables.copulas == ()
    assert scalar_space.variables.distribution.dimension == 2

    # Later mutations succeed, instead of re-raising the same error.
    scalar_space.add_variable("z", OT_NORMAL)
    assert list(scalar_space.variables) == ["x", "y", "z"]
    scalar_space.remove_variable("x")
    assert list(scalar_space.variables) == ["y", "z"]
    scalar_space.rename_variable("y", "u")
    assert list(scalar_space.variables) == ["u", "z"]


def test_compute_samples(space) -> None:
    """Check the computation of samples, as an array and as a dictionary."""
    samples = space.compute_samples(5)
    assert samples.shape == (5, 3)
    samples = space.compute_samples(5, as_dict=True)
    assert samples.keys() == {"x", "y"}
    assert samples["x"].shape == (5, 2)
    assert samples["y"].shape == (5, 1)


@pytest.mark.parametrize(
    ("settings", "samples"),
    [
        (
            [
                OTDistribution_Settings(
                    interfaced_distribution="Dirac", parameters=(1,)
                ),
                OTDistribution_Settings(
                    interfaced_distribution="Dirac", parameters=(2,)
                ),
            ],
            array([[1, 2]] * 4),
        ),
        (
            [OTDistribution_Settings(interfaced_distribution="Dirac", parameters=(1,))],
            array([[1]] * 4),
        ),
    ],
)
def test_ot_interfaced_distribution_samples(settings, samples) -> None:
    """Check the samples of a random variable based on an interfaced OT distribution."""
    space = RandomSpace()
    space.add_variable("x", *settings)
    assert_array_equal(space.compute_samples(4), samples)


@pytest.mark.parametrize(
    ("settings", "upper_bound"),
    [
        (
            [
                SPDistribution_Settings(
                    interfaced_distribution="uniform", parameters={"scale": 1}
                )
            ],
            [1],
        ),
        (
            [
                SPDistribution_Settings(
                    interfaced_distribution="uniform", parameters={"scale": 1}
                ),
                SPDistribution_Settings(
                    interfaced_distribution="uniform", parameters={"scale": 2}
                ),
            ],
            [1, 2],
        ),
    ],
)
def test_sp_interfaced_distribution_upper_bound(settings, upper_bound) -> None:
    """Check the support of a random variable based on an interfaced SP distribution."""
    space = RandomSpace()
    space.add_variable("x", *settings)
    assert_array_equal(
        space.variables.distribution.math_upper_bound, array(upper_bound)
    )


@pytest.mark.parametrize(
    ("settings_class", "distribution", "parameters", "string_representation"),
    [
        (OTDistribution_Settings, "Uniform", (), "Uniform()"),
        (OTDistribution_Settings, "Uniform", (2, 4), "Uniform(2, 4)"),
        (SPDistribution_Settings, "uniform", {}, "uniform()"),
        (
            SPDistribution_Settings,
            "uniform",
            {"scale": 2, "loc": 2},
            "uniform(scale=2, loc=2)",
        ),
    ],
)
def test_interfaced_distribution_repr(
    settings_class, distribution, parameters, string_representation
) -> None:
    """Check the marginal of a random variable based on an interfaced distribution."""
    space = RandomSpace()
    space.add_variable(
        "x",
        settings_class(interfaced_distribution=distribution, parameters=parameters),
    )
    marginal = space.variables["x"].distribution.marginals[0]
    assert str(marginal) == string_representation


def test_compute_samples_with_discrete_marginal() -> None:
    """Check that compute_samples supports a discrete marginal distribution.

    The samples of the discrete variable must take the values of its support only,
    and no sample must be NaN.
    """
    space = RandomSpace()
    space.add_variable(
        "x", OTFiniteDiscreteDistribution_Settings(value_to_weight={0: 1, 1: 2})
    )
    space.add_variable("y", OT_UNIFORM)

    RandomGenerator.SetSeed(0)
    samples = space.compute_samples(10_000)

    assert_array_equal(unique(samples[:, 0]), [0.0, 1.0])
    assert not isnan(samples).any()
    assert_allclose(samples.mean(0), [0.666, 0.499], atol=1e-3)


def test_remove_variable(space) -> None:
    """Check that removing a variable updates the joint distribution."""
    space.remove_variable("x")
    assert list(space.variables) == ["y"]
    assert space.dimension == 1
    assert space.variables.distribution.dimension == 1


def test_remove_variable_keeps_the_other_copula_and_remaps_its_indices() -> None:
    """Check that a surviving copula's indices are remapped after a removal.

    With two copulas, removing a variable covered by only one of them must
    remove that copula while keeping the other, with its indices remapped to
    the positions of its variables in the remaining space.
    """
    space = RandomSpace()
    space.add_variable("a", OT_NORMAL)
    space.add_variable("b", OT_NORMAL)
    space.add_variable("c", OT_NORMAL)
    space.add_variable("d", OT_NORMAL)
    space.add_copula(("a", "c"), NormalCopula(2))
    space.add_copula(("b", "d"), NormalCopula(2))

    space.remove_variable("a")

    assert list(space.variables) == ["b", "c", "d"]
    assert space.variables.copulas == ((("b", "d"), space.variables.copulas[0][1]),)
    assert repr(space.variables.distribution) == (
        "OTJointDistribution("
        "Normal(mu=1.0, sigma=2.0), "
        "Normal(mu=1.0, sigma=2.0), "
        "Normal(mu=1.0, sigma=2.0); "
        "MarginalDistribution("
        "distribution=BlockIndependentCopula("
        "NormalCopula(R = [[ 1 0 ]\n [ 0 1 ]]), "
        "IndependentCopula(dimension = 1)), "
        "indices=[0,2,1]))"
    )


def test_remove_all_variables(space, snapshot) -> None:
    """Check that an emptied random space behaves like an empty one."""
    space.remove_variable("x")
    space.remove_variable("y")
    assert space.variables.distribution is None
    with assert_exception(ValueError, snapshot):
        space.compute_samples()


def test_remove_all_variables_then_add_other_library() -> None:
    """Check that emptying a space resets the distribution library constraint."""
    space = RandomSpace()
    space.add_variable("x", OT_NORMAL)
    space.remove_variable("x")
    assert len(space) == 0
    space.add_variable("y", SP_NORMAL)
    assert space.variables["y"].distribution_settings == (SP_NORMAL,)


def test_filter_to_empty_then_add_other_library() -> None:
    """Check that filtering out all variables resets the distribution library."""
    space = RandomSpace()
    space.add_variable("x", OT_NORMAL)
    space.filter([])
    assert len(space) == 0
    space.add_variable("y", SP_NORMAL)
    assert space.variables["y"].distribution_settings == (SP_NORMAL,)


def test_rename_variable(space) -> None:
    """Check that renaming a variable preserves the order and the distributions."""
    space.rename_variable("x", "z")
    assert list(space.variables) == ["z", "y"]
    assert space.variables["z"].distribution.dimension == 2


def test_rename_variable_collision(space, snapshot) -> None:
    """Check that renaming to an existing name raises and leaves the space intact."""
    with assert_exception(ValueError, snapshot):
        space.rename_variable("x", "y")
    assert list(space.variables) == ["x", "y"]
    assert space.dimension == 3
    assert dict(space.variables.name_to_indices) == {
        "x": range(2),
        "y": range(2, 3),
    }
    assert space.compute_samples(2).shape == (2, 3)


def test_rename_variable_same_name_is_noop(space) -> None:
    """Check that renaming a variable to its own name is a no-op."""
    space.rename_variable("x", "x")
    assert list(space.variables) == ["x", "y"]


def test_rename_variable_preserves_the_marginal_distributions() -> None:
    """Check that renaming a variable does not swap the marginal distributions.

    Renaming a variable other than the last one must not move it
    in the mapping of the random variables,
    otherwise the joint probability distribution
    rebuilt by a subsequent mutation of the random space
    would give each random variable the marginal distribution of another one.
    """
    space = RandomSpace()
    space.add_variable(
        "u", OTDistribution_Settings(interfaced_distribution="Dirac", parameters=(1,))
    )
    space.add_variable(
        "v", OTDistribution_Settings(interfaced_distribution="Dirac", parameters=(2,))
    )
    space.rename_variable("u", "z")
    space.add_variable(
        "w", OTDistribution_Settings(interfaced_distribution="Dirac", parameters=(3,))
    )

    assert list(space.variables) == ["z", "v", "w"]
    assert_array_equal(space.compute_samples(2), array([[1, 2, 3]] * 2))


def test_filter(space) -> None:
    """Check that filtering keeps a subset of variables."""
    other = space.filter(["y"], copy=True)
    assert list(other.variables) == ["y"]
    assert list(space.variables) == ["x", "y"]
    assert other.variables.distribution.dimension == 1


def test_filter_dimensions(space) -> None:
    """Check that filtering keeps a subset of components of a random variable."""
    other = space.filter_dimensions("x", [1])
    assert other is space
    assert space.dimension == 2
    assert space.variables["x"].size == 1
    assert space.variables["y"].size == 1
    assert space.variables["x"].distribution_settings == (OT_UNIFORM,)
    assert space.variables.distribution.dimension == 2
    assert space.compute_samples(3).shape == (3, 2)


def test_filter_dimensions_with_copula(space) -> None:
    """Check that filtering a random variable removes the copula covering it."""
    space.add_copula(("x", "y"), NormalCopula(3))
    space.filter_dimensions("x", [0])
    assert space.variables.copulas == ()
    assert space.variables.distribution.dimension == 2


@pytest.mark.parametrize(
    ("name", "dimensions"),
    [
        ("x", [2]),
        ("x", [2, 3]),
        ("x", []),
        ("z", [0]),
        ("x", [-1]),
        ("x", [1, 1]),
    ],
)
def test_filter_dimensions_errors(space, name, dimensions, snapshot) -> None:
    """Check the errors raised when the components cannot be filtered."""
    with assert_exception((ValueError, UnknownVariableError), snapshot):
        space.filter_dimensions(name, dimensions)

    assert space.dimension == 3


def test_filter_dimensions_error_keeps_copula_and_dimension(space, snapshot) -> None:
    """Check that a failing filter_dimensions leaves the copulas untouched.

    A failure must leave the registry and its copulas as they were,
    instead of removing the copula covering the variable
    before the replacement variable is built and fails to be substituted.
    """
    space.add_copula(("x", "y"), NormalCopula(3))
    copulas_before = space.variables.copulas
    dimension_before = space.dimension

    with assert_exception(ValueError, snapshot):
        space.filter_dimensions("x", [-5])

    assert space.variables.copulas == copulas_before
    assert space.dimension == dimension_before
    assert space.variables["x"].size == 2


def test_equality(space) -> None:
    """Check the equality of two random spaces."""
    other = RandomSpace("u")
    other.add_variable("x", *[OT_UNIFORM] * 2)
    other.add_variable("y", OT_NORMAL)
    assert space == other
    other.remove_variable("y")
    assert space != other


def test_equality_with_copula(space) -> None:
    """Check that a copula is taken into account when comparing random spaces."""
    other = RandomSpace("u")
    other.add_variable("x", *[OT_UNIFORM] * 2)
    other.add_variable("y", OT_NORMAL)
    space.add_copula(("x", "y"), NormalCopula(3))
    assert space != other
    other.add_copula(("x", "y"), NormalCopula(3))
    assert space == other


def test_str(space, snapshot) -> None:
    """Check the string representation of a random space."""
    assert str(space) == snapshot


def test_repr(space, snapshot) -> None:
    """Check the representation of a random space."""
    assert repr(space) == snapshot


def test_str_with_transformation(snapshot) -> None:
    """Check the view when a probability distribution has a transformation."""
    space = RandomSpace()
    space.add_variable("x", OT_NORMAL)
    space.add_variable("y", OTNormalDistribution_Settings(transformation="x+2"))
    assert str(space) == snapshot


def test_repr_html(space) -> None:
    """Check that the HTML representation is titled with the random space."""
    assert "Random space" in space._repr_html_()


def test_str_with_copula(space, snapshot) -> None:
    """Check that the copulas are rendered under the tabular view."""
    space.add_copula(("x", "y"), _create_correlated_copula())
    assert str(space) == snapshot


def test_str_with_block_independent_copulas(snapshot) -> None:
    """Check that every block of random variables linked by a copula is rendered."""
    space = RandomSpace("u")
    for name in ("a", "b", "c", "d"):
        space.add_variable(name, OT_NORMAL)

    space.add_copula(("a", "c"), NormalCopula(2))
    space.add_copula(("b", "d"), NormalCopula(2))
    assert str(space) == snapshot


def test_repr_html_with_copula(space) -> None:
    """Check that the HTML representation includes the copulas."""
    space.add_copula(("x", "y"), _create_correlated_copula())
    assert "Copulas: (x, y) -> NormalCopula" in space._repr_html_()


def test_render_footer_without_copula(space) -> None:
    """Check that a random space without copula has no footer."""
    assert space._render_footer() == ""


def test_get_pretty_table_ignores_copulas(space) -> None:
    """Check that a copula does not change the tabular view.

    A copula relates several random variables
    while the tabular view has one row per component,
    so the copulas are rendered in the footer instead.
    """
    table = space.get_pretty_table().get_string()
    space.add_copula(("x", "y"), _create_correlated_copula())
    assert space.get_pretty_table().get_string() == table


def test_get_pretty_table(space, snapshot) -> None:
    """Check the tabular view of a random space with its default settings.

    The string representations pass ``with_index=True`` and ``capitalize=True``,
    so the default settings are only exercised here.
    """
    assert space.get_pretty_table().get_string() == snapshot


def test_get_pretty_table_with_index(space, snapshot) -> None:
    """Check that the components of a random vector are indexed on demand."""
    assert space.get_pretty_table(with_index=True).get_string() == snapshot


def test_get_pretty_table_ignores_fields(space) -> None:
    """Check that the fields are ignored."""
    assert (
        space.get_pretty_table(fields=["name"]).get_string()
        == space.get_pretty_table().get_string()
    )


def test_get_pretty_table_with_transformation(snapshot) -> None:
    """Check the columns added when a probability distribution has a transformation."""
    space = RandomSpace()
    space.add_variable("x", OT_NORMAL)
    space.add_variable("y", OTNormalDistribution_Settings(transformation="x+2"))
    assert space.get_pretty_table().get_string() == snapshot


def test_get_pretty_table_of_an_empty_space(snapshot) -> None:
    """Check the tabular view of a random space without random variable."""
    assert RandomSpace().get_pretty_table().get_string() == snapshot


@pytest.mark.parametrize(
    "name",
    [
        "distribution",
        "distributions",
        "get_range",
        "get_support",
        "to_design_space",
        "to_parameter_space",
        "add_random_variable",
        "add_random_vector",
        "get_tabular_view",
        "normalize_vect",
        "denormalize_vect",
        "get_current_value",
        "set_lower_bound",
    ],
)
def test_absent_members(space, name) -> None:
    """Check that the design-space and legacy members are absent."""
    assert not hasattr(space, name)


def test_transform_vect_round_trip(space) -> None:
    """Check that the iso-probabilistic transformation is invertible."""
    unit_value = array([0.1, 0.5, 0.9])
    value = space.untransform_vect(unit_value)
    assert_allclose(space.transform_vect(value), unit_value)


def test_transform_vect_2d(space) -> None:
    """Check that the transformation handles 2D arrays."""
    unit_values = array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.6]])
    values = space.untransform_vect(unit_values)
    assert values.shape == (2, 3)
    assert_allclose(space.transform_vect(values), unit_values)


def test_untransform_vect_check(space, snapshot) -> None:
    """Check the error raised when the components are not in [0,1]."""
    with assert_exception(ValueError, snapshot):
        space.untransform_vect(array([0.1, 0.5, 1.2]))


def test_untransform_vect_no_check(space) -> None:
    """Check that skipping the [0,1] check does not change the result."""
    unit_value = array([0.1, 0.5, 0.9])
    assert_allclose(
        space.untransform_vect(unit_value, no_check=True),
        space.untransform_vect(unit_value),
    )


def test_untransform_vect_wrong_size(space, snapshot) -> None:
    """Check the error raised when the point does not have the right dimension."""
    with assert_exception(ValueError, snapshot):
        space.untransform_vect(array([0.1, 0.5]))


def test_transform_vect_3d(space) -> None:
    """Check that the transformation handles the dimensions preceding the components."""
    unit_values = array([
        [[0.1, 0.5, 0.9], [0.2, 0.4, 0.6]],
        [[0.3, 0.7, 0.8], [0.15, 0.45, 0.65]],
    ])
    values = space.untransform_vect(unit_values)
    assert values.shape == (2, 2, 3)
    # The points are mapped one by one, whatever the dimensions preceding them.
    assert_allclose(values[1, 0], space.untransform_vect(unit_values[1, 0]))
    assert_allclose(space.transform_vect(values), unit_values)


def test_untransform_vect_0d(space, snapshot) -> None:
    """Check the error raised when the array to be untransformed has no dimension."""
    with assert_exception(ValueError, snapshot):
        space.untransform_vect(array(0.5))


def test_transform_vect_with_copula(space) -> None:
    """Check that the transformation takes the dependency into account."""
    unit_value = array([0.1, 0.5, 0.9])
    without_copula = space.untransform_vect(unit_value)
    space.add_copula(("x", "y"), _create_correlated_copula())
    with_copula = space.untransform_vect(unit_value)
    assert not (without_copula == with_copula).all()
    assert_allclose(space.transform_vect(with_copula), unit_value)


def test_untransform_vect_with_block_independent_copulas() -> None:
    """Check the transform with several copulas (block-independent dependence).

    The Rosenblatt transform must be applied block-wise so that it remains fast;
    relying on the conditional distributions of the block-independent copula
    would make sampling prohibitively slow.
    """
    space = RandomSpace()
    for name in ["a", "b", "c", "d", "e"]:
        space.add_variable(name, OT_NORMAL)

    correlation = CorrelationMatrix(2)
    correlation[0, 1] = 0.8
    space.add_copula(("a", "b"), NormalCopula(correlation))
    correlation = CorrelationMatrix(2)
    correlation[0, 1] = 0.5
    space.add_copula(("c", "d"), NormalCopula(correlation))
    # "e" is independent of the other variables.

    rng = default_rng(0)
    unit_samples = rng.random((3000, 5))
    samples = space.untransform_vect(unit_samples, no_check=True)
    assert samples.shape == (3000, 5)

    correlations = corrcoef(samples, rowvar=False)
    # The dependence is captured within each block ...
    assert_allclose(correlations[0, 1], 0.791, atol=1e-3)
    assert_allclose(correlations[2, 3], 0.474, atol=1e-3)
    # ... while the components of different blocks remain independent.
    assert_allclose(correlations[0, 2], 0.0, atol=0.05)
    assert_allclose(correlations[0, 4], 0.0, atol=0.05)
    assert_allclose(correlations[1, 4], 0.0, atol=0.05)
    assert_allclose(correlations[2, 4], 0.0, atol=0.05)
    assert_allclose(correlations[3, 4], 0.0, atol=0.05)

    # The transform is invertible.
    assert_allclose(space.transform_vect(samples), unit_samples)


def test_pickle(space) -> None:
    """Check that a random space survives a pickle round-trip."""
    restored = pickle.loads(pickle.dumps(space))
    assert restored == space
    assert restored.variables.distribution.dimension == 3
    assert restored.compute_samples(2).shape == (2, 3)


def test_pickle_with_copula(space) -> None:
    """Check that the copulas survive a pickle round-trip."""
    space.add_copula(("x", "y"), _create_correlated_copula())
    restored = pickle.loads(pickle.dumps(space))
    assert restored.variables.copulas[0][0] == ("x", "y")


def test_unpickle_parameter_space(monkeypatch, snapshot) -> None:
    """Check the error raised when unpickling a space pickled as a ParameterSpace.

    The name ParameterSpace is redirected onto RandomSpace by gemseo._deprecation,
    so such a pickle reaches RandomSpace with a state it cannot read.
    """
    module_name = "gemseo.algos.parameter_space"
    module = ModuleType(module_name)

    class ParameterSpace:
        """A stand-in for the removed class, to pickle its qualified name."""

    ParameterSpace.__module__ = module_name
    ParameterSpace.__qualname__ = ParameterSpace.__name__
    module.ParameterSpace = ParameterSpace
    parameter_space = ParameterSpace()
    parameter_space.__dict__.update({"_variables": {}, "distributions": {}})
    monkeypatch.setitem(sys.modules, module_name, module)
    pickled_parameter_space = pickle.dumps(parameter_space)
    monkeypatch.delitem(sys.modules, module_name)

    with assert_exception(TypeError, snapshot):
        pickle.loads(pickled_parameter_space)


def test_sample_space_with_unit_samples(space) -> None:
    """Check that a random space can be sampled in the unit hypercube."""
    samples = doe_library_factory.create("OT_MONTE_CARLO").sample_space(
        space,
        settings=OT_MONTE_CARLO_Settings(n_samples=4, seed=1),
        use_unit_samples=True,
    )
    assert samples.shape == (4, 3)
    assert ((samples >= 0.0) & (samples <= 1.0)).all()


def test_evaluation_scenario(scalar_space) -> None:
    """Check that a random space can be sampled by an evaluation scenario."""
    scenario = EvaluationScenario([AnalyticDiscipline({"f": "x+y"})], scalar_space)
    assert scenario.input_space is scalar_space
    scenario.add_observable("f")
    scenario.execute(OT_MONTE_CARLO_Settings(n_samples=3, seed=1))
    assert len(scenario.to_dataset()) == 3


def test_evaluation_scenario_with_discrete_marginal() -> None:
    """Check that an evaluation scenario samples a discrete marginal correctly."""
    space = RandomSpace()
    space.add_variable(
        "x", OTFiniteDiscreteDistribution_Settings(value_to_weight={0: 1, 1: 2})
    )
    space.add_variable("y", OT_UNIFORM)
    scenario = EvaluationScenario([AnalyticDiscipline({"z": "x+y"})], space)
    scenario.add_observable("z")
    scenario.execute(OT_MONTE_CARLO_Settings(n_samples=10, seed=1))
    data = scenario.to_dataset().to_numpy()
    assert not isnan(data).any()
    assert_allclose(
        data,
        array([
            [1.0, 0.10, 1.10],
            [1.0, 0.23, 1.23],
            [0.0, 0.13, 0.13],
            [1.0, 0.66, 1.66],
            [0.0, 0.04, 0.04],
            [1.0, 0.75, 1.75],
            [1.0, 0.18, 1.18],
            [0.0, 0.13, 0.13],
            [1.0, 0.79, 1.79],
            [1.0, 0.77, 1.77],
        ]),
        atol=6e-3,
    )


def test_sensitivity_analysis(scalar_space) -> None:
    """Check that a sensitivity analysis samples a random space."""
    discipline = AnalyticDiscipline({"f": "x+2*y"})
    settings = OT_MONTE_CARLO_Settings(n_samples=10, seed=1)
    analysis = CorrelationAnalysis()
    dataset = analysis.compute_samples(
        [discipline], scalar_space, 10, algo_settings=settings
    )
    assert len(dataset) == 10
    assert analysis._input_names == ["x", "y"]


def test_empty_space() -> None:
    """Check the container behavior of an empty random space."""
    space = RandomSpace()
    assert len(space) == 0
    assert list(space) == []
    assert space.dimension == 0
    assert space.variables.distribution is None
    assert "Random space" in str(space)


@pytest.mark.parametrize(
    "method",
    ["compute_samples", "transform_vect", "untransform_vect"],
)
def test_empty_space_error(method, snapshot) -> None:
    """Check the error raised when the random space is empty."""
    space = RandomSpace()
    arguments = () if method == "compute_samples" else (array([0.5]),)
    with assert_exception(ValueError, snapshot):
        getattr(space, method)(*arguments)
