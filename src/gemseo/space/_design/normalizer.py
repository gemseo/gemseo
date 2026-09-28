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

from numpy import asarray
from numpy import concatenate
from numpy import isin
from numpy import where
from numpy import zeros

from gemseo.space._core.registry_derived_data import RegistryDerivedData
from gemseo.space._design.constants import bound_atol
from gemseo.util._compatibility.scipy import sparse_classes
from gemseo.util._numpy import convert_array_type
from gemseo.util._numpy import float64_dtype
from gemseo.util._numpy import int64_dtype

if TYPE_CHECKING:
    from numpy import dtype

    from gemseo.space._design.bounds import Bounds
    from gemseo.space._design.integer_rounder import IntegerRounder
    from gemseo.space._design.variables import DesignVariables
    from gemseo.util.typing import IntegerArray
    from gemseo.util.typing import NumberArray
    from gemseo.util.typing import RealOrComplexArrayT
logger = logging.getLogger(__name__)


class Normalizer(RegistryDerivedData):
    """Forward/inverse normalization keyed by `Variables.version`."""

    __bounds: Bounds
    """The variable bounds."""

    __integer_rounder: IntegerRounder
    """The rounder of the integer components of the design vector."""

    __normalization_factor: NumberArray | None
    """The normalization factor `upper - lower`. `None` when never been used."""

    __normalization_factor_inv: NumberArray | None
    """The inverse of the normalization factor. `None` when never been used."""

    __normalization_indices: IntegerArray | None
    """The indices of the normalizable components of the design vector.
    `None` when never been used."""

    def __init__(
        self,
        variables: DesignVariables,
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
    ) -> RealOrComplexArrayT:
        """Normalize a full value.

        Args:
            full_value: The full value.
            common_dtype: The common dtype of the values of the variables
                (typically derived from the current values).
            subtract_lower_bound: Whether to subtract the lower bound
                before normalizing.

        Returns:
            The normalized full value.
        """
        self._refresh()
        normalization_indices = self.__normalization_indices
        if normalization_indices is None or normalization_indices.size == 0:
            # Without any component to normalize,
            # the full value is merely copied and so keeps its dtype.
            return full_value.copy()

        current_x_dtype = common_dtype
        if current_x_dtype.kind == "i":
            current_x_dtype = float64_dtype

        value = full_value.astype(current_x_dtype)
        is_sparse = isinstance(value, sparse_classes)
        # A dense value is read through a base array view of itself,
        # which shares its memory:
        # the augmented assignments below are elementwise on an array
        # and matrix products on a `numpy.matrix`,
        # which is what a densified legacy sparse matrix is,
        # and a matrix of more than one column made them raise.
        # The view costs nothing and the value keeps the type it came with.
        components = value if is_sparse else asarray(value)
        if subtract_lower_bound:
            components[..., normalization_indices] -= self.__bounds.full_lower_bound[
                normalization_indices
            ]

        if is_sparse:
            column_mask = isin(value.indices, normalization_indices)
            value.data[column_mask] *= self.__normalization_factor_inv[value.indices][
                column_mask
            ]  # type: ignore[index]
        else:
            components[..., normalization_indices] *= self.__normalization_factor_inv[
                normalization_indices
            ]  # type: ignore[index]

        return value

    def denormalize(
        self,
        full_value: RealOrComplexArrayT,
        common_dtype: dtype,
        add_lower_bound: bool = True,
        no_check: bool = False,
    ) -> RealOrComplexArrayT:
        """Denormalize a normalized full value.

        Args:
            full_value: The normalized full value.
            common_dtype: The common dtype of the values of the variables
                (typically derived from the current values).
            add_lower_bound: Whether to add the lower bound back after denormalizing.
            no_check: Whether to skip the `[0,1]` membership check.

        Returns:
            The denormalized full value.
        """
        self._refresh()
        normalization_indices = self.__normalization_indices
        lower_bounds = self.__bounds.full_lower_bound

        if not no_check and normalization_indices is not None:
            value_ = full_value[..., normalization_indices]
            lower_bounds_violated = value_ < -bound_atol
            upper_bounds_violated = value_ > 1 + bound_atol
            any_lower = lower_bounds_violated.any()
            any_upper = upper_bounds_violated.any()
            msg = "All components of the normalized vector should be between 0 and 1; "
            if any_lower:
                msg += f"lower bounds violated: {value_[lower_bounds_violated]}; "

            if any_upper:
                msg += f"upper bounds violated: {value_[upper_bounds_violated]}; "

            if any_lower or any_upper:
                msg = msg[:-2] + "."
                logger.warning(msg)

        current_dtype = common_dtype
        if full_value.dtype.kind == "c":
            # A complex full value carries a complex-step perturbation
            # in its imaginary part,
            # which recasting it to the (real) common dtype would silently drop.
            # Kept at its own dtype instead,
            # so it already matches `current_dtype` below
            # and neither the conversion nor the integer recast touches it.
            current_dtype = full_value.dtype

        recast_to_int = current_dtype.kind == "i"
        if recast_to_int:
            current_dtype = float64_dtype

        # Adding the lower bound back is what tells a point of the space
        # from a direction in it:
        # a gradient is denormalized without it,
        # and rounding a gradient would destroy its integer components
        # instead of snapping a point to the grid.
        round_integers = self.__integer_rounder.has_integer and add_lower_bound
        # The integer recast only occurs when there are integer components to round.
        recast_to_int = recast_to_int and round_integers

        if full_value.dtype == current_dtype:
            value = full_value.copy()
        else:
            # convert_array_type drops the imaginary part of a complex array
            # whatever the target dtype is:
            # explicitly, through its own real part, when the target is complex,
            # and implicitly, through the cast itself, when it is not.
            # This call is skipped whenever the dtypes already match,
            # which the branch above guarantees for a complex full value,
            # so it only ever converts a real one here.
            value = convert_array_type(full_value, current_dtype)

        if normalization_indices is not None and normalization_indices.size:
            # A dense value is scaled through a base array view of itself;
            # see `normalize`.
            is_sparse = isinstance(value, sparse_classes)
            components = value if is_sparse else asarray(value)
            if is_sparse:
                column_mask = isin(value.indices, normalization_indices)
                value.data[column_mask] *= self.__normalization_factor[value.indices][
                    column_mask
                ]  # type: ignore[index]
            else:
                components[..., normalization_indices] *= self.__normalization_factor[
                    normalization_indices
                ]  # type: ignore[index]

            if add_lower_bound:
                components[..., normalization_indices] += lower_bounds[
                    normalization_indices
                ]

        if round_integers:
            value = self.__integer_rounder.round(value, copy=False)

        return convert_array_type(value, int64_dtype) if recast_to_int else value
