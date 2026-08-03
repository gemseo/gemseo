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
"""Normalizer for versioned variables."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from numpy import concatenate
from numpy import isin
from numpy import where
from numpy import zeros

from gemseo.space.design._checking import check_out_array
from gemseo.space.design._constants import BOUND_ATOL
from gemseo.space.design._registry_derived_data import RegistryDerivedData
from gemseo.util._compatibility.scipy import sparse_classes
from gemseo.util._numpy import FLOAT64_DTYPE
from gemseo.util._numpy import INT64_DTYPE
from gemseo.util._numpy import convert_array_type

if TYPE_CHECKING:
    from numpy import dtype
    from numpy import ndarray

    from gemseo.space.design._bounds import Bounds
    from gemseo.space.design._integer_rounder import IntegerRounder
    from gemseo.space.design._variables import Variables
    from gemseo.util.typing import RealOrComplexArrayT


LOGGER = logging.getLogger(__name__)


class Normalizer(RegistryDerivedData):
    """Forward/inverse normalization keyed by `Variables.version`."""

    __bounds: Bounds
    """The variable bounds."""

    __integer_rounder: IntegerRounder
    """The rounder of the integer components of the design vector."""

    __normalization_factor: ndarray | None
    """The normalization factor `upper - lower`."""

    __normalization_factor_inv: ndarray | None
    """The inverse of the normalization factor."""

    __normalization_indices: ndarray | None
    """The indices of the normalizable components of the design vector."""

    def __init__(
        self,
        variables: Variables,
        bounds: Bounds,
        integer_rounder: IntegerRounder,
    ) -> None:
        """
        Args:
            variables: The variables.
            bounds: The bounds.
            integer_rounder: The rounder of the integer components.
        """  # noqa: D205, D212
        super().__init__(variables)
        self._register_guard(self._rebuild)
        self.__bounds = bounds
        self.__integer_rounder = integer_rounder
        self.__normalization_factor = None
        self.__normalization_factor_inv = None
        self.__normalization_indices = None

    def _rebuild(self) -> None:
        """Rebuild the normalization data."""
        lower = self.__bounds.full_lower_bound
        upper = self.__bounds.full_upper_bound
        self.__normalization_factor = upper - lower
        name_to_normalization_mask = self._variables.name_to_normalization_mask
        normalization_mask = (
            concatenate([name_to_normalization_mask[name] for name in self._variables])
            if name_to_normalization_mask
            else zeros(0, dtype=bool)
        )
        self.__normalization_indices = normalization_mask.nonzero()[0]
        # Avoid divide-by-zero when lb == ub.
        is_zero = self.__normalization_factor == 0.0
        self.__normalization_factor_inv = 1.0 / where(
            is_zero, 1, self.__normalization_factor
        )

    def normalize(
        self,
        full_value: RealOrComplexArrayT,
        common_dtype: dtype,
        subtract_lower_bound: bool = True,
        out: RealOrComplexArrayT | None = None,
    ) -> RealOrComplexArrayT:
        """Normalize a full value.

        Args:
            full_value: The full value.
            common_dtype: The common dtype of the values of the variables
                (typically derived from the current values).
            subtract_lower_bound: Whether to subtract the lower bound
                before normalizing.
            out: The array to store the normalized full value.
                Its dtype and shape must be those of the normalized full value.
                If `None`, allocate a new array.

        Returns:
            The normalized full value.

        Raises:
            ValueError: When `out` cannot store the normalized full value.
        """
        self._refresh()
        normalization_indices = self.__normalization_indices
        if normalization_indices is None or normalization_indices.size == 0:
            # Without any component to normalize,
            # the full value is merely copied and so keeps its dtype.
            if out is None:
                return full_value.copy()

            check_out_array(out, full_value.dtype, full_value.shape)
            out[...] = full_value
            return out

        current_x_dtype = common_dtype
        if current_x_dtype.kind == "i":
            current_x_dtype = FLOAT64_DTYPE

        if out is None:
            out = full_value.astype(current_x_dtype)
        else:
            check_out_array(out, current_x_dtype, full_value.shape)
            out[...] = full_value

        if subtract_lower_bound:
            out[..., normalization_indices] -= self.__bounds.full_lower_bound[
                normalization_indices
            ]

        if isinstance(out, sparse_classes):
            column_mask = isin(out.indices, normalization_indices)
            out.data[column_mask] *= self.__normalization_factor_inv[out.indices][
                column_mask
            ]  # type: ignore[index]
        else:
            out[..., normalization_indices] *= self.__normalization_factor_inv[
                normalization_indices
            ]  # type: ignore[index]

        return out

    def denormalize(
        self,
        full_value: RealOrComplexArrayT,
        common_dtype: dtype,
        add_lower_bound: bool = True,
        no_check: bool = False,
        out: ndarray | None = None,
    ) -> RealOrComplexArrayT:
        """Denormalize a normalized full value.

        Args:
            full_value: The normalized full value.
            common_dtype: The common dtype of the values of the variables
                (typically derived from the current values).
            add_lower_bound: Whether to add the lower bound back after denormalizing.
            no_check: Whether to skip the `[0,1]` membership check.
            out: The array to store the denormalized full value.
                Its dtype and shape must be those of the denormalized full value.
                If `None`, allocate a new array.

        Returns:
            The denormalized full value.

        Raises:
            ValueError: When `out` cannot store the denormalized full value.
        """
        self._refresh()
        normalization_indices = self.__normalization_indices
        lower_bounds = self.__bounds.full_lower_bound

        if not no_check and normalization_indices is not None:
            value_ = full_value[..., normalization_indices]
            lower_bounds_violated = value_ < -BOUND_ATOL
            upper_bounds_violated = value_ > 1 + BOUND_ATOL
            any_lower = lower_bounds_violated.any()
            any_upper = upper_bounds_violated.any()
            msg = "All components of the normalized vector should be between 0 and 1; "
            if any_lower:
                msg += f"lower bounds violated: {value_[lower_bounds_violated]}; "

            if any_upper:
                msg += f"upper bounds violated: {value_[upper_bounds_violated]}; "

            if any_lower or any_upper:
                msg = msg[:-2] + "."
                LOGGER.warning(msg)

        current_dtype = common_dtype
        recast_to_int = current_dtype.kind == "i"
        if recast_to_int:
            current_dtype = FLOAT64_DTYPE

        has_integer = self.__integer_rounder.has_integer
        # The integer recast only occurs when there are integer components to round.
        recast_to_int = recast_to_int and has_integer
        result_dtype = INT64_DTYPE if recast_to_int else current_dtype

        if out is not None:
            check_out_array(out, result_dtype, full_value.shape)

        if out is not None and out.dtype == current_dtype:
            # Fill and scale the array of the caller in place.
            # An integer array is excluded here
            # because it cannot store the intermediate float values.
            value = out
            value[...] = full_value
        elif full_value.dtype == current_dtype:
            value = full_value.copy()
        else:
            # convert_array_type takes the real part when the target dtype is complex,
            # hence it must only be called when a conversion is actually needed,
            # otherwise the imaginary part of a complex full value would be lost.
            value = convert_array_type(full_value, current_dtype)

        if normalization_indices is not None and normalization_indices.size:
            if isinstance(value, sparse_classes):
                column_mask = isin(value.indices, normalization_indices)
                value.data[column_mask] *= self.__normalization_factor[value.indices][
                    column_mask
                ]  # type: ignore[index]
            else:
                value[..., normalization_indices] *= self.__normalization_factor[
                    normalization_indices
                ]  # type: ignore[index]

            if add_lower_bound:
                value[..., normalization_indices] += lower_bounds[normalization_indices]

        if has_integer:
            value = self.__integer_rounder.round(value, copy=False)

        if out is None:
            return convert_array_type(value, INT64_DTYPE) if recast_to_int else value

        if value is not out:
            # The values are already rounded, hence the integer assignment is exact.
            out[...] = value

        return out
