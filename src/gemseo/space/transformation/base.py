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

"""The contract of a space transformation."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Generic
from typing import TypeVar

from numpy import isfinite

from gemseo.space.variable.numeric import BaseNumericVariable
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.space.base import BaseVariableSpace
    from gemseo.util.typing import NumberArray

_SpaceT = TypeVar("_SpaceT", bound="BaseVariableSpace")


class BaseSpaceTransformation(
    Generic[_SpaceT], metaclass=ABCGoogleDocstringInheritanceMeta
):
    """A map from an original space to a working space.

    A transformation is built for an *original* space,
    and derives a *working* one from it.

    Each space is associated with a domain
    defining the values that its variables can take.
    For example,
    an integer variable can only take integer values.

    Two coordinate systems are involved and must not be conflated:

    - the *original* coordinates,
      at the scale the user declared,
      where the disciplines are evaluated and where the database is keyed,
    - the *working* coordinates,
      the image of the original ones under this transformation,
      e.g. normalized.

    A transformation relaxing the original domain
    may let a value expressed in the original coordinates fall outside it.
    For example,
    an integer variable may be explored continuously between its bounds
    and in that case,
    all the original coordinates lie between the bounds,
    but only a finite number of them lie within the original domain.

    Two maps go backwards:
    [inverse_transform_value][gemseo.space.transformation.base.BaseSpaceTransformation.inverse_transform_value],
    run at every evaluation,
    and [project][gemseo.space.transformation.base.BaseSpaceTransformation.project],
    meant to be run once,
    on the result.
    These maps must not be merged
    because projecting at every evaluation would
    turn a relaxed function into a step function and zero its gradient.

    The working space is built once,
    from the original space as it is at that moment,
    and kept:
    a transformation describes a space as it was when the transformation was built,
    so a space that changes afterwards calls for a new transformation.
    A driver builds one transformation per run,
    which is that lifetime.

    The working space need not have the variables of the original one,
    nor its dimension:
    the maps carry whole values and Jacobians from one space to the other,
    and whoever consumes a transformation reads the working space from it.
    A transformation splitting a vector into scalars,
    freezing a component
    or encoding a variable
    fits this contract as it is.
    """  # noqa: E501

    is_affine: ClassVar[bool] = False
    """Whether the map is affine.

    This states a property of the map,
    for a consumer able to exploit it,
    such as one folding it into the coefficients of a linear function.
    The evaluation of a problem does not:
    the database is keyed on the untransformed point,
    which the backward map has to produce at every evaluation whatever the map is.
    """

    requires_finite_bounds: ClassVar[bool] = False
    """Whether the map is defined only where the bounds are finite.

    A transformation declaring this refuses,
    at construction,
    a space where a component has an infinite bound or no bound at all.
    """

    _original_space: _SpaceT
    """The space this transformation is built for."""

    __working_space: _SpaceT | None
    """The working space, once built."""

    def __init__(self, space: _SpaceT) -> None:
        """
        Args:
            space: The original space.

        Raises:
            ValueError: When the map requires finite bounds
                and a component of the space has none.
        """  # noqa: D205, D212
        if self.requires_finite_bounds:
            unbounded = [
                name
                for name, variable in space.variables.items()
                if not isinstance(variable, BaseNumericVariable)
                or not isfinite(variable.lower_bound).all()
                or not isfinite(variable.upper_bound).all()
            ]
            if unbounded:
                msg = (
                    f"{type(self).__name__} requires finite bounds, "
                    "which these variables do not have: "
                    f"{', '.join(unbounded)}."
                )
                raise ValueError(msg)

        self._original_space = space
        self.__working_space = None

    @property
    def original_space(self) -> _SpaceT:
        """The space this transformation is built for."""
        return self._original_space

    @property
    def working_space(self) -> _SpaceT:
        """The image of the original space under this transformation."""
        # It is built on the first access and kept,
        # so that it is one object,
        # which a transformation built on it can be checked against.
        if self.__working_space is None:
            self.__working_space = self._create_working_space()

        return self.__working_space

    @abstractmethod
    def _create_working_space(self) -> _SpaceT:
        """Build the working space from the original one.

        Returns:
            The working space.
        """

    @abstractmethod
    def transform_value(self, value: NumberArray) -> NumberArray:
        """Map a value of the original space to the working space.

        Args:
            value: A value of the original space.

        Returns:
            The corresponding value of the working space.
        """

    @abstractmethod
    def inverse_transform_value(
        self, value: NumberArray, no_check: bool = False
    ) -> NumberArray:
        """Map a value of the working space back to the original space.

        Args:
            value: A value of the working space.
            no_check: Whether to skip the checks on the value.

        Returns:
            The corresponding value of the original space.
        """  # noqa: E501

    @abstractmethod
    def transform_jacobian(
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        """Map a Jacobian of the original space to the working space.

        Args:
            jacobian: A Jacobian with respect to the original space.
            value: The value of the original space where the Jacobian is taken,
                which a non-affine map needs.

        Returns:
            The Jacobian with respect to the working space.
        """

    @abstractmethod
    def inverse_transform_jacobian(
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        """Map a Jacobian of the working space back to the original space.

        Args:
            jacobian: A Jacobian with respect to the working space.
            value: The value of the original space where the Jacobian is taken,
                which a non-affine map needs.

        Returns:
            The Jacobian with respect to the original space.
        """

    def project(self, value: NumberArray) -> NumberArray:
        """Project a value in the original coordinates onto the original domain.

        This map is meant to be applied once,
        on the result,
        and never at every evaluation.
        A driver applies it to the optimum it writes back into the design space,
        and a composition applies it to each of its transformations.

        It is the identity by default,
        since only a transformation relaxing the domain has anything to restore.

        Args:
            value: A value expressed in the original coordinates.

        Returns:
            The projected value, in the original domain.
        """
        return value

    def create_constraints(
        self, working_space: BaseVariableSpace
    ) -> tuple[ArrayFunction, ...]:
        """Create the constraints that the transformation imposes.

        This states a property of the transformation,
        for a consumer able to exploit it,
        such as one adding these constraints to a problem once and for all.
        The evaluation of a problem does not read it yet:
        it is part of the contract for the story that will.

        The space is an argument,
        unlike the working space of this transformation,
        because the constraints are wanted in the coordinates of whoever consumes them:
        a composition asks each of its transformations for constraints on the
        working space of the composition,
        which is not the working space of the transformation.
        Carrying the constraints of a transformation to those coordinates
        is the job of the story that consumes them.

        Args:
            working_space: The space the constraints are expressed on.

        Returns:
            The constraints, empty by default.
        """
        return ()
