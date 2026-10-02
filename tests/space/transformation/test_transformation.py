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

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING
from typing import ClassVar

import pytest
from numpy import array
from numpy import broadcast_to
from numpy import concatenate
from numpy import inf
from numpy import sqrt
from numpy import zeros
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from gemseo.core.function.array_function import ArrayFunction
from gemseo.space.design import DesignSpace
from gemseo.space.transformation._working import create_working_transformation
from gemseo.space.transformation.base import BaseSpaceTransformation
from gemseo.space.transformation.composition import SpaceComposition
from gemseo.space.transformation.identity import SpaceIdentity
from gemseo.space.transformation.normalization import SpaceNormalization
from gemseo.space.transformation.relaxation import SpaceRelaxation
from gemseo.space.variable import DataType
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from gemseo.util.typing import NumberArray


@pytest.fixture
def design_space() -> DesignSpace:
    """A design space with a float variable and an integer one."""
    space = DesignSpace()
    space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=2.0)
    space.add_integer_variable("i", lower_bound=0, upper_bound=10, value=4)
    return space


@pytest.fixture
def mixed_space(design_space) -> DesignSpace:
    """The design space above, with a discrete variable added."""
    design_space.add_discrete_variable("d", [1, 3, 8], value=3)
    return design_space


def test_normalization_round_trip(design_space) -> None:
    """Check that the backward map inverts the forward one."""
    transformation = SpaceNormalization(design_space)
    original_value = array([2.0, 4.0])

    working_value = transformation.transform_value(original_value)
    # The integer component is not normalized, see the test below.
    assert_allclose(working_value, array([0.2, 4.0]))
    assert_allclose(
        transformation.inverse_transform_value(working_value), original_value
    )


def test_normalization_delegates_the_rounding_to_denormalize_vect(
    design_space,
) -> None:
    """Check what `denormalize_vect` already does to the integer components.

    `DesignSpace.denormalize_vect` rounds them itself,
    whatever a driver asked for:
    the rounder sits inside the normalizer.
    A driver that wants them explored continuously relaxes them first,
    so that the normalization is built on a space without integer variables.
    """
    transformation = SpaceNormalization(design_space)

    untransformed = transformation.inverse_transform_value(array([0.25, 4.5]))
    assert_allclose(untransformed, array([2.5, 4.0]))
    # This transformation rounds nothing of its own, so projecting changes nothing.
    assert_allclose(transformation.project(untransformed), untransformed)


def test_relaxation_leaves_the_backward_map_alone(design_space) -> None:
    """Check that a relaxed value reaches the functions, and the database, as it is.

    The maps are the identity:
    a value of the working space is a value of the original coordinates,
    relaxed,
    and only `project` takes it back to the domain the user declared.
    """
    transformation = SpaceRelaxation(design_space)
    value = array([2.5, 4.5])

    assert_allclose(transformation.transform_value(value), value)
    assert_allclose(transformation.inverse_transform_value(value), value)
    assert_allclose(transformation.project(value), array([2.5, 4.0]))
    assert transformation.working_space is not design_space
    assert transformation.is_affine


def test_relaxation_working_space(mixed_space) -> None:
    """Check that the working space relaxes the integer and discrete variables.

    Every variable becomes a float one with the same bounds,
    the bounds of a discrete variable being its smallest and largest choices,
    and the current value is carried over as a float.
    The normalization policy follows the kind of a variable,
    so a relaxed variable is normalizable where the one it replaces was not.
    Two variables with the same bounds share their relaxed variable.
    The original space is left untouched.
    """
    mixed_space.add_integer_variable("j", lower_bound=0, upper_bound=10, value=5)
    working_space = SpaceRelaxation(mixed_space).working_space

    assert list(working_space) == ["x", "i", "d", "j"]
    assert [variable.type for variable in working_space.variables.values()] == [
        DataType.REAL
    ] * 4
    assert working_space.variables["j"] is working_space.variables["i"]
    assert_allclose(working_space.get_lower_bounds(), array([0.0, 0.0, 1.0, 0.0]))
    assert_allclose(working_space.get_upper_bounds(), array([10.0, 10.0, 8.0, 10.0]))
    current_value = working_space.get_current_value()
    assert current_value.dtype.kind == "f"
    assert_allclose(current_value, array([2.0, 4.0, 3.0, 5.0]))
    assert_equal(
        dict(working_space.name_to_normalization_mask),
        {
            "x": array([True]),
            "i": array([True]),
            "d": array([True]),
            "j": array([True]),
        },
    )

    assert [variable.type for variable in mixed_space.variables.values()] == [
        DataType.REAL,
        DataType.INTEGER,
        DataType.DISCRETE,
        DataType.INTEGER,
    ]
    assert_equal(
        dict(mixed_space.name_to_normalization_mask),
        {
            "x": array([True]),
            "i": array([False]),
            "d": array([False]),
            "j": array([False]),
        },
    )


def test_relaxation_working_space_current_value_mutation_is_isolated(
    design_space,
) -> None:
    """Check that mutating the working space's current value spares the original.

    A driver mutates the working space's current value while it runs,
    e.g. through `set_current_variable` or by casting it to complex for a
    complex-step derivative,
    and the working space only shares what is immutable with the space
    it was built from,
    so none of that reaches the current value of the latter.
    """
    original_value = design_space.get_current_value(["i"]).copy()
    original_dtype = original_value.dtype
    working_space = SpaceRelaxation(design_space).working_space

    working_space.set_current_variable("i", array([7.5]))
    assert_allclose(design_space.get_current_value(["i"]), original_value)

    working_space.to_complex()
    assert working_space.get_current_value(["i"]).dtype.kind == "c"
    assert design_space.get_current_value(["i"]).dtype == original_dtype


def test_relaxation_names(mixed_space) -> None:
    """Check the names of the variables a relaxation relaxes.

    `relaxed_names` agrees with `working_space`: the names it lists are the
    ones the working space carries over as float variables that were not
    float to start with.
    """
    relaxation = SpaceRelaxation(mixed_space)
    assert relaxation.relaxed_names == frozenset({"i", "d"})
    relaxation = SpaceRelaxation(mixed_space, relax_discrete=False)
    assert relaxation.relaxed_names == frozenset({"i"})
    relaxation = SpaceRelaxation(mixed_space, relax_integer=False)
    assert relaxation.relaxed_names == frozenset({"d"})
    assert (
        SpaceRelaxation(
            mixed_space, relax_integer=False, relax_discrete=False
        ).relaxed_names
        == frozenset()
    )


def test_relaxation_keeps_a_variable_without_a_value(design_space) -> None:
    """Check that a relaxed variable without a value keeps none."""
    design_space.add_integer_variable("j", lower_bound=0, upper_bound=3)

    working_space = SpaceRelaxation(design_space).working_space

    assert not working_space.has_current_value
    assert_allclose(working_space.get_current_value(["x", "i"]), array([2.0, 4.0]))


def test_relaxation_projects_onto_the_declared_domain(mixed_space) -> None:
    """Check that `project` rounds an integer and snaps a discrete one to a choice.

    A discrete component goes to its nearest choice,
    the lower one when two are as near,
    and a matrix of samples is projected row by row.
    """
    transformation = SpaceRelaxation(mixed_space)

    assert_allclose(
        transformation.project(array([2.5, 4.4, 5.4])), array([2.5, 4.0, 3.0])
    )
    assert_allclose(
        transformation.project(array([2.5, 4.6, 2.0])), array([2.5, 5.0, 1.0])
    )
    assert_allclose(
        transformation.project(array([[2.5, 4.6, 2.0], [0.0, 0.4, 7.0]])),
        array([[2.5, 5.0, 1.0], [0.0, 0.0, 8.0]]),
    )


def test_relaxation_projects_without_mutating_the_input() -> None:
    """Check that `project` does not modify its input in place.

    A space without an integer variable takes `round_vect` down the branch
    that returns its input untouched, so the discrete components used to be
    snapped into the caller's own array.
    """
    space = DesignSpace()
    space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0)
    space.add_discrete_variable("d", [1, 3, 8])
    transformation = SpaceRelaxation(space, relax_discrete=True)
    value = array([2.5, 5.4])

    projected_value = transformation.project(value)

    assert projected_value is not value
    assert_allclose(value, array([2.5, 5.4]))
    assert_allclose(projected_value, array([2.5, 3.0]))


def test_relaxation_keeps_the_discrete_variables_when_asked(mixed_space) -> None:
    """Check that not relaxing the discrete variables leaves them as they are.

    The integer variable is still relaxed,
    and `project` snaps no component that is not relaxed.
    """
    transformation = SpaceRelaxation(mixed_space, relax_discrete=False)
    working_space = transformation.working_space

    assert [variable.type for variable in working_space.variables.values()] == [
        DataType.REAL,
        DataType.REAL,
        DataType.DISCRETE,
    ]
    # The discrete component, `5.4`, is left as it is: not snapped to a choice.
    assert_allclose(
        transformation.project(array([2.5, 4.4, 5.4])), array([2.5, 4.0, 5.4])
    )


def test_relaxation_keeps_the_integer_variables_when_asked(mixed_space) -> None:
    """Check the symmetrical case: not relaxing the integer variables.

    The discrete variable is still relaxed,
    and `project` rounds the integer component regardless,
    which is harmless since it is already integral,
    the working value of a variable that is not relaxed.
    """
    transformation = SpaceRelaxation(mixed_space, relax_integer=False)
    working_space = transformation.working_space

    assert [variable.type for variable in working_space.variables.values()] == [
        DataType.REAL,
        DataType.INTEGER,
        DataType.REAL,
    ]
    assert_allclose(
        transformation.project(array([2.5, 4.0, 5.4])), array([2.5, 4.0, 3.0])
    )


def test_relaxation_of_a_space_with_nothing_to_relax() -> None:
    """Check that a space without integer or discrete variables is kept as it is."""
    space = DesignSpace()
    space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0)

    assert SpaceRelaxation(space).working_space is space


def test_relaxation_leaves_a_jacobian_unchanged(design_space) -> None:
    """Check that the relaxation scales no derivative.

    The maps are the identity,
    so the relaxation contributes no factor to the chain rule, either way.
    """
    transformation = SpaceRelaxation(design_space)
    jacobian = array([[1.0, 2.0]])

    assert_allclose(transformation.transform_jacobian(jacobian), jacobian)
    assert_allclose(transformation.inverse_transform_jacobian(jacobian), jacobian)


def test_create_for_relaxes_then_normalizes(design_space) -> None:
    """Check the composition a driver asking for both transformations gets.

    The normalization is built on the relaxed space,
    so the integer variable is normalized too,
    and the backward map denormalizes without rounding.
    """
    composition = create_working_transformation(
        design_space, normalize=True, relax_integer=True, relax_discrete=True
    )

    assert [
        type(transformation).__name__ for transformation in composition.transformations
    ] == [
        "SpaceRelaxation",
        "SpaceNormalization",
    ]
    relaxation, normalization = composition.transformations
    assert normalization.original_space is relaxation.working_space
    assert_allclose(
        composition.inverse_transform_value(array([0.25, 0.45])), array([2.5, 4.5])
    )


def test_create_for_each_combination(design_space) -> None:
    """Check that each transformation is applied only when the driver asks for it."""
    assert len(create_working_transformation(design_space)) == 0
    assert len(create_working_transformation(design_space, normalize=True)) == 1
    assert (
        len(
            create_working_transformation(
                design_space, relax_integer=True, relax_discrete=True
            )
        )
        == 1
    )


def test_create_for_skips_relaxation_with_nothing_to_relax() -> None:
    """Check that a space without integer or discrete variables takes no relaxation."""
    space = DesignSpace()
    space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0)

    composition = create_working_transformation(
        space, normalize=True, relax_integer=True, relax_discrete=True
    )

    assert [
        type(transformation).__name__ for transformation in composition.transformations
    ] == ["SpaceNormalization"]


def test_create_for_relaxes_a_discrete_variable() -> None:
    """Check that a discrete variable alone is enough for the relaxation."""
    space = DesignSpace()
    space.add_discrete_variable("d", [1, 3, 8])

    composition = create_working_transformation(space, relax_discrete=True)

    assert [
        type(transformation).__name__ for transformation in composition.transformations
    ] == ["SpaceRelaxation"]


def test_create_for_relax_discrete_on_a_space_without_discrete_variables(
    design_space,
) -> None:
    """Check that asking to relax the discrete variables alone applies nothing.

    `design_space` has an integer variable and no discrete one,
    so a request to relax the discrete variables alone finds nothing to relax.
    """
    assert len(create_working_transformation(design_space, relax_discrete=True)) == 0


def test_create_for_relax_integer_on_a_space_without_integer_variables() -> None:
    """Check the symmetrical case: relaxing the integer variables alone.

    The space has a discrete variable and no integer one,
    so a request to relax the integer variables alone finds nothing to relax.
    """
    space = DesignSpace()
    space.add_discrete_variable("d", [1, 3, 8])

    assert len(create_working_transformation(space, relax_integer=True)) == 0


def test_create_for_a_non_design_space() -> None:
    """Check that a non-normalizable space applies no coordinate transformation.

    This is the shape a problem over a random space takes.
    """
    from gemseo.space.random import RandomSpace

    assert len(create_working_transformation(RandomSpace())) == 0


def test_create_for_a_non_design_space_with_normalization(snapshot) -> None:
    """Check that normalizing a space that has no bounds to normalize is refused."""
    from gemseo.space.random import RandomSpace

    with assert_exception(ValueError, snapshot):
        create_working_transformation(RandomSpace(), normalize=True)


def test_normalization_jacobian(design_space) -> None:
    """Check that the Jacobian maps both ways."""
    transformation = SpaceNormalization(design_space)
    jacobian = array([[1.0, 1.0]])

    working_jacobian = transformation.transform_jacobian(jacobian)
    # Only the float component is scaled, the integer one is not normalized.
    assert_allclose(working_jacobian, array([[10.0, 1.0]]))
    assert_allclose(
        transformation.inverse_transform_jacobian(working_jacobian), jacobian
    )


def test_normalization_transform_space(design_space) -> None:
    """Check the working space built from a design space."""
    working_space = SpaceNormalization(design_space).working_space

    assert list(working_space) == ["x", "i"]
    # The integer variable keeps its own bounds,
    # since it is not normalized,
    # and it is continuous in the working space:
    # the algorithm explores it continuously
    # and the backward map rounds.
    assert_allclose(working_space.get_lower_bounds(), array([0.0, 0.0]))
    assert_allclose(working_space.get_upper_bounds(), array([1.0, 10.0]))
    assert_allclose(working_space.get_current_value(), array([0.2, 4.0]))


def test_normalization_keeps_unbounded_variables() -> None:
    """Check that a component without finite bounds is left alone."""
    space = DesignSpace()
    space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0)
    space.add_real_variable("y")
    transformation = SpaceNormalization(space)

    assert not transformation.requires_finite_bounds
    assert_allclose(
        transformation.transform_value(array([5.0, 3.0])), array([0.5, 3.0])
    )

    working_space = transformation.working_space
    assert_allclose(working_space.get_lower_bounds(), array([0.0, -inf]))
    assert_allclose(working_space.get_upper_bounds(), array([1.0, inf]))


def test_identity(design_space) -> None:
    """Check that the identity leaves the space and the values unchanged."""
    transformation = SpaceIdentity(design_space)
    value = array([1.0, 2.0])

    assert transformation.working_space is design_space
    assert_equal(transformation.transform_value(value), value)
    assert_equal(transformation.inverse_transform_value(value), value)
    assert_equal(transformation.transform_jacobian(value), value)
    assert_equal(transformation.inverse_transform_jacobian(value), value)
    assert transformation.is_affine


def test_composition_is_the_identity_when_empty(design_space) -> None:
    """Check that an empty composition changes nothing."""
    composition = SpaceComposition(design_space)
    value = array([1.0, 2.0])

    assert len(composition) == 0
    assert composition.is_affine
    assert composition.working_space is design_space
    assert_equal(composition.transform_value(value), value)
    assert_equal(composition.inverse_transform_value(value), value)
    assert_equal(composition.transform_jacobian(value), value)
    assert_equal(composition.inverse_transform_jacobian(value), value)
    assert_equal(composition.project(value), value)
    assert composition.create_constraints(design_space) == ()


def test_composition_drops_the_identity(design_space) -> None:
    """Check that the identity is dropped at construction."""
    composition = SpaceComposition(
        design_space,
        SpaceIdentity,
        SpaceNormalization,
        SpaceIdentity,
    )

    (transformation,) = composition.transformations
    assert type(transformation) is SpaceNormalization


def test_composition_applies_the_transformations_in_order(design_space) -> None:
    """Check that the backward map applies the transformations reversed."""
    composition = SpaceComposition(design_space, SpaceNormalization)
    (transformation,) = composition.transformations
    original_value = array([2.0, 4.0])

    working_value = composition.transform_value(original_value)
    assert_allclose(working_value, transformation.transform_value(original_value))
    assert_allclose(composition.inverse_transform_value(working_value), original_value)


def test_composition_iterates_over_its_transformations(design_space) -> None:
    """Check that a composition iterates its transformations, original to working."""
    composition = SpaceComposition(design_space, SpaceRelaxation, SpaceNormalization)
    relaxation, normalization = composition.transformations

    assert list(composition) == [relaxation, normalization]


def test_composition_collects_the_constraints_of_its_transformations(
    design_space,
) -> None:
    """Check that a composition gathers the constraints of its transformations, ordered.

    Every transformation is handed the working space of the composition,
    since that is the space the constraints of the composition are expressed on.
    """
    working_spaces = []

    class _ConstrainedTransformation(SpaceIdentity):
        """A transformation imposing a constraint of its own."""

        def __init__(self, space: DesignSpace, name: str) -> None:
            """
            Args:
                space: The original space.
                name: The name of the constraint.
            """  # noqa: D205, D212
            super().__init__(space)
            self.__name = name

        def create_constraints(self, working_space) -> tuple[ArrayFunction, ...]:  # noqa: D102
            working_spaces.append(working_space)
            return (ArrayFunction(lambda value: value, name=self.__name),)

    composition = SpaceComposition(
        design_space,
        partial(_ConstrainedTransformation, name="c1"),
        partial(_ConstrainedTransformation, name="c2"),
    )

    constraints = composition.create_constraints(design_space)

    assert [constraint.name for constraint in constraints] == ["c1", "c2"]
    assert [space is design_space for space in working_spaces] == [True, True]


def test_composition_composes_chains(design_space) -> None:
    """Check that a composition is itself a transformation."""
    composition = SpaceComposition(
        design_space,
        lambda space: SpaceComposition(space, SpaceNormalization),
    )

    assert_allclose(composition.transform_value(array([2.0, 4.0])), array([0.2, 4.0]))


class _ShiftTransformation(SpaceIdentity):
    """A transformation shifting a value by one half.

    Being a translation, it leaves a Jacobian unchanged, as the identity does.
    """

    def transform_value(self, value: NumberArray) -> NumberArray:  # noqa: D102
        return value + 0.5

    def inverse_transform_value(  # noqa: D102
        self, value: NumberArray, no_check: bool = False
    ) -> NumberArray:
        return value - 0.5


class _SquareTransformation(SpaceIdentity):
    """A non-affine transformation squaring a value of its own positive space."""

    is_affine = False

    def transform_value(self, value: NumberArray) -> NumberArray:  # noqa: D102
        return value**2

    def inverse_transform_value(  # noqa: D102
        self, value: NumberArray, no_check: bool = False
    ) -> NumberArray:
        return sqrt(value)

    def transform_jacobian(  # noqa: D102
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        # dw/du = 1 / (2 w) at the value w of the original space of this transformation.
        return jacobian / (2 * value)

    def inverse_transform_jacobian(  # noqa: D102
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        return jacobian * 2 * value


class _IntegerOnWorkingTransformation(SpaceIdentity):
    """A transformation whose original domain is its own original space's integers."""

    def project(self, value: NumberArray) -> NumberArray:  # noqa: D102
        return value.round()


def test_composition_transform_jacobian_moves_the_point_along_the_transformations(
    design_space,
) -> None:
    """Check that a transformation takes its Jacobian at a point of its own space.

    The point where the second transformation is taken
    is the image of the point under the first one,
    so the chain rule is checked against a finite difference.
    """
    composition = SpaceComposition(
        design_space, _ShiftTransformation, _SquareTransformation
    )
    original_value = array([1.0])
    working_value = composition.transform_value(original_value)

    perturbation = 1e-7
    finite_difference = (
        composition.inverse_transform_value(working_value + perturbation)
        - composition.inverse_transform_value(working_value)
    ) / perturbation

    jacobian = composition.transform_jacobian(array([[1.0]]), original_value)

    # dv/du = 1 / (2 (v + 1/2)) = 1/3 at v = 1,
    # and not 1 / (2 v) = 1/2,
    # which is what handing the second transformation the value of the
    # original space of the composition would give.
    assert_allclose(jacobian, array([[1 / 3]]))
    assert_allclose(jacobian, array([finite_difference]), atol=1e-7)


def test_composition_inverse_transform_jacobian_moves_the_point_along_transformations(
    design_space,
) -> None:
    """Check that the backward map reads the point in the coordinates of each
    transformation.

    The working Jacobian is the one the test above checks against a finite difference,
    so the backward map must give the original Jacobian back.
    """  # noqa: D205, D212
    composition = SpaceComposition(
        design_space, _ShiftTransformation, _SquareTransformation
    )

    jacobian = composition.inverse_transform_jacobian(array([[1 / 3]]), array([1.0]))

    # du/dv = 2 (v + 1/2) = 3 at v = 1,
    # and not 2 v = 2,
    # which is what handing the second transformation the value of the
    # original space of the composition would give.
    assert_allclose(jacobian, array([[1.0]]))


def test_composition_jacobian_without_a_value(design_space) -> None:
    """Check that an affine composition maps a Jacobian without being given a point."""
    composition = SpaceComposition(design_space, SpaceNormalization)
    jacobian = array([[1.0, 1.0]])

    working_jacobian = composition.transform_jacobian(jacobian)

    assert_allclose(working_jacobian, array([[10.0, 1.0]]))
    assert_allclose(composition.inverse_transform_jacobian(working_jacobian), jacobian)


def test_composition_project_uses_the_coordinates_of_each_transformation(
    design_space,
) -> None:
    """Check that a transformation projects a value of its own original space.

    The second transformation accepts the integers of *its* original space,
    which is the working space of the first one.
    Projecting the value of the original space of the composition, in either order,
    leaves it outside the domain of the second transformation.
    """
    composition = SpaceComposition(
        design_space, _ShiftTransformation, _IntegerOnWorkingTransformation
    )

    projected_value = composition.project(array([0.4]))

    # Rounding 0.4 gives 0.0,
    # which is 0.5 in the coordinates of the second transformation
    # and therefore not an integer.
    assert_allclose(projected_value, array([0.5]))
    assert_allclose(composition.transform_value(projected_value), array([1.0]))


class FreezingTransformation(BaseSpaceTransformation[DesignSpace]):
    """A transformation freezing the last component of its space at its current value.

    The working space has one component fewer than the original one,
    which is what a transformation encoding, splitting or freezing variables does:
    the contract leaves the dimension of the working space to the transformation.
    """

    is_affine: ClassVar[bool] = True

    __frozen_value: NumberArray
    """The value of the frozen component, of shape `(1,)`."""

    def __init__(self, space: DesignSpace) -> None:
        """
        Args:
            space: The original space.
        """  # noqa: D205, D212
        super().__init__(space)
        self.__frozen_value = space.get_current_value()[-1:]

    def _create_working_space(self) -> DesignSpace:  # noqa: D102
        space = self._original_space
        names = list(space.variables)
        size = space.variables[names[-1]].size
        if size == 1:
            return space.filter(names[:-1], copy=True)

        return space.filter(names, copy=True).filter_dimensions(
            names[-1], list(range(size - 1))
        )

    def transform_value(self, value: NumberArray) -> NumberArray:  # noqa: D102
        return value[..., :-1]

    def inverse_transform_value(  # noqa: D102
        self, value: NumberArray, no_check: bool = False
    ) -> NumberArray:
        frozen_value = broadcast_to(self.__frozen_value, (*value.shape[:-1], 1))
        return concatenate((value, frozen_value), axis=-1)

    def transform_jacobian(  # noqa: D102
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        return jacobian[..., :-1]

    def inverse_transform_jacobian(  # noqa: D102
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        return concatenate((jacobian, zeros((*jacobian.shape[:-1], 1))), axis=-1)


def test_composition_accepts_a_dimension_change(design_space) -> None:
    """Check that a transformation is free to give the working space its own dimension.

    The composition carries whole values and Jacobians from one transformation to
    the next, so nothing in it ties the dimension of the working space to the
    original one.
    The frozen component is `4` in the coordinates of the freezing transformation,
    which is `3.5` in those of the composition,
    and only a value freezing that component makes the round trip.
    """
    composition = SpaceComposition(
        design_space, _ShiftTransformation, FreezingTransformation
    )
    original_value = array([2.0, 3.5])

    assert composition.working_space.dimension == 1
    assert list(composition.working_space.variables) == ["x"]

    working_value = composition.transform_value(original_value)
    assert_allclose(working_value, array([2.5]))
    assert_allclose(composition.inverse_transform_value(working_value), original_value)

    working_jacobian = composition.transform_jacobian(
        array([[1.0, 2.0]]), original_value
    )
    assert_allclose(working_jacobian, array([[1.0]]))
    assert_allclose(
        composition.inverse_transform_jacobian(working_jacobian, original_value),
        array([[1.0, 0.0]]),
    )


class _BoundsToConstraintsTransformation(BaseSpaceTransformation[DesignSpace]):
    """A transformation freeing the bounds of its space and imposing them as
    constraints.

    This is what an algorithm without bound support asks for:
    the working space has the variables of the original one, unbounded,
    and the bounds come back as two inequality constraints,
    `lower_bounds - value <= 0` and `value - upper_bounds <= 0`.
    The maps are the identity.
    """

    is_affine: ClassVar[bool] = True

    def _create_working_space(self) -> DesignSpace:  # noqa: D102
        space = self._original_space.filter(list(self._original_space), copy=True)
        for name in space:
            space.set_lower_bound(name, -inf)
            space.set_upper_bound(name, inf)

        return space

    def transform_value(self, value: NumberArray) -> NumberArray:  # noqa: D102
        return value

    def inverse_transform_value(  # noqa: D102
        self, value: NumberArray, no_check: bool = False
    ) -> NumberArray:
        return value

    def transform_jacobian(  # noqa: D102
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        return jacobian

    def inverse_transform_jacobian(  # noqa: D102
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        return jacobian

    def create_constraints(  # noqa: D102
        self, working_space: DesignSpace
    ) -> tuple[ArrayFunction, ...]:
        lower_bounds = self._original_space.get_lower_bounds()
        upper_bounds = self._original_space.get_upper_bounds()
        return (
            ArrayFunction(
                lambda value: lower_bounds - value,
                name="lower_bounds",
                f_type=ArrayFunction.FunctionType.INEQ,
                input_names=list(working_space.variables),
            ),
            ArrayFunction(
                lambda value: value - upper_bounds,
                name="upper_bounds",
                f_type=ArrayFunction.FunctionType.INEQ,
                input_names=list(working_space.variables),
            ),
        )


def test_create_constraints_returns_the_constraints_a_transformation_imposes(
    design_space,
) -> None:
    """Check that a transformation hands back the constraints it imposes.

    The transformation frees the bounds of the space
    and returns them as inequality constraints,
    which read negative where a bound holds and positive where it is violated.
    """
    transformation = _BoundsToConstraintsTransformation(design_space)
    working_space = transformation.working_space

    assert list(working_space) == ["x", "i"]
    assert_equal(working_space.get_lower_bounds(), array([-inf, -inf]))
    assert_equal(working_space.get_upper_bounds(), array([inf, inf]))

    constraints = transformation.create_constraints(working_space)

    assert len(constraints) == 2
    assert all(isinstance(constraint, ArrayFunction) for constraint in constraints)
    assert [constraint.name for constraint in constraints] == [
        "lower_bounds",
        "upper_bounds",
    ]
    assert all(
        constraint.f_type == ArrayFunction.FunctionType.INEQ
        for constraint in constraints
    )
    assert all(constraint.input_names == ["x", "i"] for constraint in constraints)

    lower, upper = constraints
    inside = array([2.0, 4.0])
    assert_equal(lower.evaluate(inside), array([-2.0, -4.0]))
    assert_equal(upper.evaluate(inside), array([-8.0, -6.0]))
    outside = array([-1.0, 12.0])
    assert_equal(lower.evaluate(outside), array([1.0, -12.0]))
    assert_equal(upper.evaluate(outside), array([-11.0, 2.0]))

    # A composition collects them from the transformation as they are.
    composition = SpaceComposition(design_space, _BoundsToConstraintsTransformation)
    assert [
        constraint.name
        for constraint in composition.create_constraints(composition.working_space)
    ] == ["lower_bounds", "upper_bounds"]


def test_composition_reports_requiring_finite_bounds(design_space) -> None:
    """Check that a composition requires finite bounds when one transformation does."""

    class _BoundedTransformation(SpaceIdentity):
        requires_finite_bounds = True

    assert not SpaceComposition(design_space).requires_finite_bounds
    assert SpaceComposition(design_space, _BoundedTransformation).requires_finite_bounds


def test_base_defaults(design_space) -> None:
    """Check the defaults carried by the contract."""
    assert not BaseSpaceTransformation.is_affine
    assert not BaseSpaceTransformation.requires_finite_bounds

    transformation = SpaceIdentity(design_space)
    value = array([1.0, 2.0])
    assert_equal(transformation.project(value), value)
    assert transformation.create_constraints(design_space) == ()


def test_working_space_is_built_once(design_space) -> None:
    """Check that the working space is built on the first access and then kept.

    A transformation built on it can only be checked against one object,
    so a second, equal, working space would defeat that check.
    """
    transformation = SpaceNormalization(design_space)

    assert transformation.working_space is transformation.working_space


def test_original_space(design_space) -> None:
    """Check that the original space is the one the transformation is built for.

    An empty composition has no transformation of its own, so its working space is
    the original space of the composition, and a composition of one transformation
    shares the working space of that transformation.
    """
    normalization = SpaceNormalization(design_space)
    assert normalization.original_space is design_space

    composition = SpaceComposition(design_space, SpaceNormalization)
    assert composition.original_space is design_space
    assert composition.working_space is composition.transformations[0].working_space

    empty_chain = SpaceComposition(design_space)
    assert empty_chain.working_space is design_space


def test_composition_of_one_transformation_builds_its_working_space_lazily(
    design_space, monkeypatch
) -> None:
    """Check that a composition of one transformation reads no working space eagerly.

    `SpaceRelaxation._create_working_space` copies the space,
    which is costly and unneeded to build the composition:
    nothing is built after the last transformation,
    so its working space is left to be read, and cached, on the first
    access to the composition's own `working_space`, not at construction.
    """
    calls = []
    original_create_working_space = SpaceRelaxation._create_working_space

    def _counting_create_working_space(self):
        calls.append(self)
        return original_create_working_space(self)

    monkeypatch.setattr(
        SpaceRelaxation, "_create_working_space", _counting_create_working_space
    )

    composition = SpaceComposition(design_space, SpaceRelaxation)
    assert calls == []

    working_space = composition.working_space

    assert calls == [composition.transformations[0]]
    assert working_space is composition.transformations[0].working_space


def test_composition_builds_each_transformation_on_the_working_space_of_the_one_before(
    design_space,
) -> None:
    """Check that a transformation is built on the working space of the one before it.

    The normalization is built on the original space of the composition,
    and the relaxation is built on the working space of the normalization,
    so the working space of the composition is the one of the last transformation.
    """
    composition = SpaceComposition(design_space, SpaceNormalization, SpaceRelaxation)

    transformations = composition.transformations
    assert transformations[0].original_space is design_space
    assert transformations[1].original_space is transformations[0].working_space
    assert composition.working_space is transformations[1].working_space


def test_composition_skips_a_dropped_identity_when_building_the_next_transformation(
    design_space,
) -> None:
    """Check that an identity factory between two transformations is dropped.

    The identity is not kept as a transformation,
    and the transformation after it is built on the working space of the last kept
    transformation, not on the one the dropped identity would have produced.
    """
    composition = SpaceComposition(
        design_space,
        SpaceNormalization,
        SpaceIdentity,
        SpaceRelaxation,
    )

    transformations = composition.transformations
    assert len(transformations) == 2
    assert transformations[1].original_space is transformations[0].working_space


def test_requires_finite_bounds_is_checked_at_construction(snapshot) -> None:
    """Check the error when a variable of the space has no finite bounds.

    A fully bounded space is accepted, since every variable has finite bounds.
    """

    class _BoundedTransformation(SpaceIdentity):
        requires_finite_bounds = True

    space = DesignSpace()
    space.add_real_variable("x", lower_bound=0.0, upper_bound=1.0)
    space.add_real_variable("y")

    with assert_exception(ValueError, snapshot):
        _BoundedTransformation(space)

    bounded_space = DesignSpace()
    bounded_space.add_real_variable("x", lower_bound=0.0, upper_bound=1.0)
    _BoundedTransformation(bounded_space)
