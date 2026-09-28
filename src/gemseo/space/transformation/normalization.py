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

"""The normalization of a design space."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import where
from numpy import zeros

from gemseo.space.transformation.base import BaseSpaceTransformation
from gemseo.space.variable import DataType
from gemseo.space.variable.factory import deterministic_variable_factory

if TYPE_CHECKING:
    from gemseo.space.design import DesignSpace
    from gemseo.util.typing import NumberArray


class SpaceNormalization(BaseSpaceTransformation["DesignSpace"]):
    """The map from a design space to its normalized image.

    A component with finite bounds is mapped to $[0,1]$.
    A component without them is left unchanged,
    as `DesignSpace.normalize_vect` does,
    so this transformation does not require finite bounds.
    An integer component is left unchanged too,
    since `DesignSpace` does not normalize integer variables by default;
    it is explored continuously over its own bounds
    and the backward map rounds it.

    Normalization only.
    Relaxing the integer and discrete variables is a separate concern,
    and a separate transformation,
    since a driver can ask for one without the other; see
    [SpaceRelaxation][gemseo.space.transformation.relaxation.SpaceRelaxation].
    """  # noqa: E501

    is_affine: ClassVar[bool] = True

    def _create_working_space(self) -> DesignSpace:  # noqa: D102
        space = self._original_space
        lower_bounds = space.get_lower_bounds()
        upper_bounds = space.get_upper_bounds()
        name_to_value = space._current_value
        # The normalization reads the whole vector,
        # which a space where a single variable has no value cannot provide.
        # The missing components are filled with zeros
        # and the normalized vector is read back only for the variables
        # that do have a value,
        # so that one variable left unset does not cost the others
        # the point the user declared.
        current_value = (
            space.normalize_vect(
                space.convert_dict_to_array({
                    name: zeros(space.variables[name].size) if value is None else value
                    for name, value in name_to_value.items()
                })
            )
            if any(value is not None for value in name_to_value.values())
            else None
        )
        name_to_mask = space.name_to_normalization_mask
        name_to_variable = {}
        name_to_working_value = {}
        # The variables of a normalized space repeat themselves,
        # a whole space of scalar variables with finite bounds
        # mapping to the very same variable,
        # and a variable is immutable,
        # so one object per pair of bounds is built and shared.
        # Building one per name would pay a validation per variable.
        bounds_to_variable = {}
        start = 0
        for name in space:
            stop = start + space.variables[name].size
            mask = name_to_mask.get(name)
            if mask is None or not mask.any():
                # A variable this map leaves alone is carried over as it is,
                # with its own type and bounds.
                # An integer one is in that case,
                # since a design space does not normalize integer variables,
                # and a consumer handling integrality itself reads that type back.
                name_to_working_value[name] = name_to_value.get(name)
            else:
                # A normalized component lands in the unit interval by construction,
                # and is given that interval
                # rather than the image of its bounds:
                # dividing a range by itself is off by an epsilon,
                # which would leave the bound of the working space
                # just below the value it starts from.
                lower_bound = where(mask, 0.0, lower_bounds[start:stop])
                upper_bound = where(mask, 1.0, upper_bounds[start:stop])
                key = (lower_bound.tobytes(), upper_bound.tobytes())
                variable = bounds_to_variable.get(key)
                if variable is None:
                    variable = bounds_to_variable[key] = (
                        deterministic_variable_factory.create(
                            DataType.REAL,
                            size=stop - start,
                            lower_bound=lower_bound,
                            upper_bound=upper_bound,
                        )
                    )

                name_to_variable[name] = variable
                name_to_working_value[name] = (
                    None
                    if current_value is None or name_to_value.get(name) is None
                    else current_value[start:stop]
                )

            start = stop

        # The working space has the variables of the space it normalizes,
        # in the same order and with the same sizes,
        # so it is built as a copy with the normalized variables substituted,
        # rather than one variable at a time:
        # a design space reindexes itself
        # and refreshes its current value on every variable added,
        # which a wide space pays as a quadratic cost.
        return space._copy_with_variables(name_to_variable, name_to_working_value)

    def transform_value(self, value: NumberArray) -> NumberArray:  # noqa: D102
        return self._original_space.normalize_vect(value)

    def inverse_transform_value(  # noqa: D102
        self, value: NumberArray, no_check: bool = False
    ) -> NumberArray:
        return self._original_space.denormalize_vect(value, no_check=no_check)

    def transform_jacobian(  # noqa: D102
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        return self._original_space.normalize_grad(jacobian)

    def inverse_transform_jacobian(  # noqa: D102
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        return self._original_space.denormalize_grad(jacobian)
