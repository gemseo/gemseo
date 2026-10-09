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
"""The creation of the working transformation a driver asks for."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from gemseo.space.transformation.composition import SpaceComposition

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo.space.base import BaseVariableSpace
    from gemseo.space.transformation.base import BaseSpaceTransformation
    from gemseo.util.typing import NumberArray


def project_onto_declared_domain(
    space: BaseVariableSpace, value: NumberArray
) -> NumberArray:
    """Project a value onto the domain the user declared.

    A value may fall outside the domain a space declares,
    e.g. `3.4` for an integer variable an algorithm explores continuously,
    see
    [SpaceRelaxation][gemseo.space.transformation.relaxation.SpaceRelaxation].
    This projects it back,
    without building the working space of a
    [working transformation][gemseo.space.transformation._working.create_working_transformation],
    a copy of the space that the projection never reads.

    Args:
        space: The space the value is expressed on.
        value: The value, in the coordinates of `space`.

    Returns:
        The value projected onto the domain `space` declares,
        `value` itself when `space` has nothing to relax.
    """  # noqa: E501
    from gemseo.enum._variable_type import VariableType
    from gemseo.space.design import DesignSpace
    from gemseo.space.transformation.relaxation import SpaceRelaxation

    if not isinstance(space, DesignSpace):
        return value

    variables = space.variables
    if not (
        variables.has_variables_of_type(VariableType.INTEGER)
        or variables.has_variables_of_type(VariableType.DISCRETE)
    ):
        return value

    return SpaceRelaxation(space, relax_integer=True, relax_discrete=True).project(
        value
    )


def create_working_transformation(
    space: BaseVariableSpace,
    normalize: bool = False,
    relax_integer: bool = False,
    relax_discrete: bool = False,
) -> SpaceComposition:
    """Create the composition of coordinate transformations a driver asks for.

    The transformations are ordered from the original space to the working one:
    the relaxation first,
    then the normalization,
    built on the relaxed space,
    so that the backward map denormalizes
    and hands the relaxed value over as it is.
    A space that is not a design space applies neither transformation,
    which leaves an empty composition.

    Args:
        space: The space the problem is defined on.
        normalize: Whether the working values are normalized.
        relax_integer: Whether the integer variables are relaxed
            to float ones.
        relax_discrete: Whether the discrete variables are relaxed
            to float ones.

    Returns:
        The composition, possibly empty.

    Raises:
        ValueError: When normalization is asked of a space that has no
            bounds to normalize against.
    """
    from gemseo.enum._variable_type import VariableType
    from gemseo.space.design import DesignSpace
    from gemseo.space.transformation.normalization import SpaceNormalization
    from gemseo.space.transformation.relaxation import SpaceRelaxation

    if not isinstance(space, DesignSpace):
        if normalize:
            # The bounds of a random variable
            # are the limits of the support of its distribution;
            # they are descriptive only,
            # so there is nothing to normalize against.
            msg = (
                "The functions cannot take normalized inputs because "
                f"a {type(space).__name__} cannot be normalized."
            )
            raise ValueError(msg)

        return SpaceComposition(space)

    transformation_factories: list[
        Callable[[DesignSpace], BaseSpaceTransformation[DesignSpace]]
    ] = []
    variables = space.variables
    if (relax_integer and variables.has_variables_of_type(VariableType.INTEGER)) or (
        relax_discrete and variables.has_variables_of_type(VariableType.DISCRETE)
    ):
        transformation_factories.append(
            partial(
                SpaceRelaxation,
                relax_integer=relax_integer,
                relax_discrete=relax_discrete,
            )
        )

    if normalize:
        # The composition builds a transformation on the working space of the
        # one before it, so the normalization normalizes the relaxed space
        # when there is one.
        transformation_factories.append(SpaceNormalization)

    return SpaceComposition(space, *transformation_factories)
