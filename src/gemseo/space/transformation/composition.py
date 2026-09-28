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

"""The ordered composition of space transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.space.transformation.base import BaseSpaceTransformation
from gemseo.space.transformation.base import _SpaceT
from gemseo.space.transformation.identity import SpaceIdentity

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.space.base import BaseVariableSpace
    from gemseo.util.typing import NumberArray


class SpaceComposition(BaseSpaceTransformation[_SpaceT]):
    """An ordered composition of transformations, from original to working.

    The forward maps apply the transformations in order
    and the backward ones apply them reversed,
    so that the composition is itself a transformation
    and a composition composes compositions.
    An empty composition is the identity.

    A transformation reads and writes values of *its own* original space,
    which is the working space of the transformation before it.
    The composition is therefore given the factories of its transformations,
    such as their classes,
    and builds each transformation on the working space of the one before it,
    starting from the original space of the composition.
    So a value the composition passes to a transformation,
    the point where a Jacobian is taken or the point to project,
    is carried to the coordinates of that transformation first,
    and never handed over as it is.
    """

    __transformations: tuple[BaseSpaceTransformation[_SpaceT], ...]
    """The transformations, from the original space to the working one."""

    def __init__(
        self,
        space: _SpaceT,
        *transformation_factories: Callable[
            [_SpaceT], BaseSpaceTransformation[_SpaceT]
        ],
    ) -> None:
        """
        Args:
            space: The original space of the composition.
            *transformation_factories: The factories of the transformations,
                from the original space to the working one,
                each called on the space its transformation is built on,
                e.g. a transformation class
                or a `functools.partial` of it setting its other arguments.
                The identity transformations are dropped.
        """  # noqa: D205, D212
        # `type(...) is` rather than `isinstance`:
        # a subclass of the identity is not necessarily the identity,
        # and dropping it would silently ignore a transformation.
        # An identity transformation has its original space as working space,
        # so dropping it leaves the next transformation built on the same space.
        # The transformations are set before the base class is initialized,
        # since the flags it reads are the ones of the transformations.
        transformations: list[BaseSpaceTransformation[_SpaceT]] = []
        transformation_space = space
        last_factory_index = len(transformation_factories) - 1
        for index, create_transformation in enumerate(transformation_factories):
            transformation = create_transformation(transformation_space)
            if type(transformation) is not SpaceIdentity:
                transformations.append(transformation)
                # Built only when a next transformation is built on it,
                # since it can be costly, e.g. a copy of the space;
                # the last one is built on demand by `_create_working_space`.
                if index < last_factory_index:
                    transformation_space = transformation.working_space

        self.__transformations = tuple(transformations)
        super().__init__(space)

    @property
    def transformations(self) -> tuple[BaseSpaceTransformation[_SpaceT], ...]:
        """The transformations, from the original space to the working one."""
        return self.__transformations

    @property
    def is_affine(self) -> bool:  # type: ignore[override]
        """Whether every transformation is affine, and so the composition is."""
        return all(
            transformation.is_affine for transformation in self.__transformations
        )

    @property
    def requires_finite_bounds(self) -> bool:  # type: ignore[override]
        """Whether any transformation requires finite bounds."""
        return any(
            transformation.requires_finite_bounds
            for transformation in self.__transformations
        )

    def __len__(self) -> int:
        return len(self.__transformations)

    def __iter__(self) -> Iterator[BaseSpaceTransformation[_SpaceT]]:
        return iter(self.__transformations)

    def _create_working_space(self) -> _SpaceT:  # noqa: D102
        # The transformations are built one on the working space of the other,
        # so the working space of the composition is the one of its last
        # transformation, and the original space when there is none.
        if not self.__transformations:
            return self._original_space

        return self.__transformations[-1].working_space

    def transform_value(self, value: NumberArray) -> NumberArray:  # noqa: D102
        for transformation in self.__transformations:
            value = transformation.transform_value(value)

        return value

    def inverse_transform_value(  # noqa: D102
        self, value: NumberArray, no_check: bool = False
    ) -> NumberArray:
        for transformation in reversed(self.__transformations):
            value = transformation.inverse_transform_value(value, no_check=no_check)

        return value

    def __compute_transformation_values(
        self, value: NumberArray | None
    ) -> tuple[NumberArray | None, ...]:
        """Compute the value at which each transformation is taken.

        The original space of a transformation is the working space of the
        transformation before it, so a transformation reads the image of the
        value under the transformations preceding it, not the value of the
        original space of the composition.

        Args:
            value: The value of the original space of the composition, if any.

        Returns:
            The value of the original space of each transformation, in the
            order of the transformations.
        """
        if not self.__transformations:
            return ()

        if value is None:
            return (None,) * len(self.__transformations)

        values = [value]
        for transformation in self.__transformations[:-1]:
            values.append(transformation.transform_value(values[-1]))

        return tuple(values)

    def transform_jacobian(  # noqa: D102
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        values = self.__compute_transformation_values(value)
        for transformation, transformation_value in zip(
            self.__transformations, values, strict=True
        ):
            jacobian = transformation.transform_jacobian(jacobian, transformation_value)

        return jacobian

    def inverse_transform_jacobian(  # noqa: D102
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        values = self.__compute_transformation_values(value)
        for transformation, transformation_value in zip(
            reversed(self.__transformations), reversed(values), strict=True
        ):
            jacobian = transformation.inverse_transform_jacobian(
                jacobian, transformation_value
            )

        return jacobian

    def project(self, value: NumberArray) -> NumberArray:  # noqa: D102
        if not self.__transformations:
            return value

        # A transformation projects onto the domain of its own original space,
        # so the value is carried forward to the coordinates of each
        # transformation and the last projection is carried back
        # to the original space of the composition.
        transformations = self.__transformations[:-1]
        for transformation in transformations:
            value = transformation.transform_value(transformation.project(value))

        value = self.__transformations[-1].project(value)
        for transformation in reversed(transformations):
            value = transformation.inverse_transform_value(value)

        return value

    def create_constraints(  # noqa: D102
        self, working_space: BaseVariableSpace
    ) -> tuple[ArrayFunction, ...]:
        constraints: list[ArrayFunction] = []
        for transformation in self.__transformations:
            constraints.extend(transformation.create_constraints(working_space))

        return tuple(constraints)
