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

"""The relaxation of the integer and discrete variables of a design space."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import abs as np_abs
from numpy import asarray
from numpy import float64

from gemseo.space.transformation.base import BaseSpaceTransformation
from gemseo.space.variable import VariableType
from gemseo.space.variable.factory import deterministic_variable_factory

if TYPE_CHECKING:
    from gemseo.space.design import DesignSpace
    from gemseo.util.typing import NumberArray
    from gemseo.util.typing import RealArray


class SpaceRelaxation(BaseSpaceTransformation["DesignSpace"]):
    """The map relaxing the integer and discrete variables of a design space.

    The working space is a copy of the original one
    where every relaxed integer or discrete variable is replaced by a float one
    with the same bounds,
    so that an algorithm explores it continuously between its bounds;
    a variable of a kind that is not relaxed is carried over as it is.
    The maps are the identity:
    a value of the working space is a value of the original coordinates,
    and the functions and the database see it as it is,
    relaxed,
    e.g. `3.4` for an integer variable.

    [project][gemseo.space.transformation.relaxation.SpaceRelaxation.project]
    is the map back to the domain the user declared:
    it rounds a relaxed integer component
    and snaps a relaxed discrete one to its nearest choice.
    It is meant to be applied once,
    to a result,
    and never at every evaluation:
    projecting at every evaluation would turn a relaxed function into a step function
    and zero its gradient.
    """  # noqa: E501

    is_affine: ClassVar[bool] = True

    __relax_integer: bool
    """Whether the integer variables are relaxed."""

    __relax_discrete: bool
    """Whether the discrete variables are relaxed."""

    __discrete_components: tuple[tuple[range, RealArray], ...]
    """The indices and the choices of each relaxed discrete variable."""

    __relaxed_names: frozenset[str]
    """The names of the variables this transformation relaxes to float ones."""

    def __init__(
        self,
        space: DesignSpace,
        relax_integer: bool = True,
        relax_discrete: bool = True,
    ) -> None:
        """
        Args:
            relax_integer: Whether the integer variables are relaxed.
            relax_discrete: Whether the discrete variables are relaxed.
        """  # noqa: D205, D212
        self.__relax_integer = relax_integer
        self.__relax_discrete = relax_discrete
        super().__init__(space)
        # Computed once, here, and read from both `_create_working_space` and
        # `relaxed_names`, so the two agree on what is relaxed.
        self.__relaxed_names = frozenset(
            name
            for name, variable in space.variables.items()
            if (variable.type == VariableType.INTEGER and relax_integer)
            or (variable.type == VariableType.DISCRETE and relax_discrete)
        )
        name_to_indices = space._variables.name_to_indices
        self.__discrete_components = tuple(
            (name_to_indices[name], asarray(variable.choices, dtype=float64))
            for name, variable in space.variables.items()
            if variable.type == VariableType.DISCRETE and relax_discrete
        )

    @property
    def relaxed_names(self) -> frozenset[str]:
        """The names of the variables this transformation relaxes to float ones."""
        return self.__relaxed_names

    def _create_working_space(self) -> DesignSpace:  # noqa: D102
        space = self._original_space
        name_to_value = space._current_value
        name_to_variable = {}
        name_to_working_value = {}
        # The relaxed variables of a space repeat themselves,
        # every integer variable with the same bounds mapping to the same float one,
        # and a variable is immutable,
        # so one object per pair of bounds is built and shared.
        bounds_to_variable = {}
        for name, variable in space.variables.items():
            value = name_to_value.get(name)
            if name not in self.__relaxed_names:
                # A float variable, or one of a kind that is not relaxed,
                # is carried over as it is, with its own bounds.
                name_to_working_value[name] = value
                continue

            # The bounds of a discrete variable are its smallest and largest choices,
            # so relaxing it is relaxing an integer one: same bounds, float type.
            lower_bound = asarray(variable.lower_bound, dtype=float64)
            upper_bound = asarray(variable.upper_bound, dtype=float64)
            key = (lower_bound.tobytes(), upper_bound.tobytes())
            relaxed_variable = bounds_to_variable.get(key)
            if relaxed_variable is None:
                relaxed_variable = bounds_to_variable[key] = (
                    deterministic_variable_factory.create(
                        VariableType.REAL,
                        size=variable.size,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    )
                )

            name_to_variable[name] = relaxed_variable
            name_to_working_value[name] = (
                None if value is None else asarray(value, dtype=float64)
            )

        if not name_to_variable:
            # Nothing to relax: the working space is the original one.
            return space

        # The working space has the variables of the original one,
        # in the same order and with the same sizes,
        # so it is built as a copy with the relaxed variables substituted,
        # rather than one variable at a time;
        # see SpaceNormalization for the cost this avoids.
        return space._copy_with_variables(name_to_variable, name_to_working_value)

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

    def project(self, value: NumberArray) -> NumberArray:  # noqa: D102
        # `round_vect` rounds every integer component of the original space,
        # relaxed or not;
        # an unrelaxed one is already integral by the time it reaches `project`,
        # so rounding it again changes nothing.
        # When the space has no integer variable,
        # `round_vect` returns `value` itself, untouched,
        # so the copy is made explicit here,
        # and the discrete components are then snapped in place into that copy.
        projected_value = self._original_space.round_vect(value.copy(), copy=False)
        # `__discrete_components` only holds the relaxed discrete variables,
        # see `__init__`.
        for indices, choices in self.__discrete_components:
            component = projected_value[..., indices]
            # A tie is settled by `argmin`, which keeps the first,
            # and so the lower, of the two nearest choices.
            projected_value[..., indices] = choices[
                np_abs(choices - component[..., None]).argmin(axis=-1)
            ]

        return projected_value
