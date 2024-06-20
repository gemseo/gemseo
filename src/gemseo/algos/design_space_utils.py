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
"""Design space helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal
from typing import overload

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.algos.design_space import DesignSpace


@overload
def get_value_and_bounds(
    design_space: DesignSpace, normalize_ds: bool, as_dict: Literal[False] = False
) -> tuple[ndarray, ndarray, ndarray]: ...


@overload
def get_value_and_bounds(
    design_space: DesignSpace, normalize_ds: bool, as_dict: Literal[True] = True
) -> tuple[dict[str, ndarray], dict[str, ndarray], dict[str, ndarray]]: ...


# TODO: return the design space to be used by the solver instead of a tuple
def get_value_and_bounds(
    design_space: DesignSpace, normalize_ds: bool, as_dict: bool = False
) -> (
    tuple[ndarray, ndarray, ndarray]
    | tuple[dict[str, ndarray], dict[str, ndarray], dict[str, ndarray]]
):
    """Return the design variable values and their lower and upper bounds.

    Args:
        design_space: The design space.
        normalize_ds: Whether to normalize the design variables.
        as_dict: Whether to return dictionaries instead of NumPy arrays.

    Returns:
        The values of the design variables,
        their lower bounds,
        and their upper bounds.
    """
    if not normalize_ds:
        return (
            design_space.get_current_value(complex_to_real=True, as_dict=as_dict),
            design_space.get_lower_bounds(as_dict=as_dict),
            design_space.get_upper_bounds(as_dict=as_dict),
        )

    current_value = design_space.get_current_value(
        complex_to_real=True, as_dict=as_dict, normalize=True
    )
    lower_bounds = design_space.normalize_vect(design_space.get_lower_bounds())
    upper_bounds = design_space.normalize_vect(design_space.get_upper_bounds())
    if not as_dict:
        return current_value, lower_bounds, upper_bounds

    return (
        current_value,
        design_space.array_to_dict(lower_bounds),
        design_space.array_to_dict(upper_bounds),
    )
