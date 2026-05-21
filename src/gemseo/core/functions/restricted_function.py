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
"""Function restricted to a subset of input components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import arange
from numpy import argsort
from numpy import delete
from numpy import insert

from gemseo.core.functions.array_function import ArrayFunction

if TYPE_CHECKING:
    from gemseo.typing import IntegerArray
    from gemseo.typing import NumberArray


class RestrictedFunction(ArrayFunction):
    """An ArrayFunction restricted to a subset of its input components.

    The remaining input components are frozen at fixed values.
    """

    __frozen_indexes: IntegerArray
    """The indices of the frozen input components, sorted in ascending order."""

    __frozen_values: NumberArray
    """The values of the frozen input components, ordered consistently with
    `__frozen_indexes`."""

    __function: ArrayFunction
    """The original function."""

    __insert_obj: IntegerArray
    """The insertion positions in the sub-vector for `numpy.insert`.

    `numpy.insert` expects positions relative to the sub-vector (active inputs),
    not the full vector. For sorted `frozen_indexes`, the j-th insertion position
    is `frozen_indexes[j] - j`, since j elements have already been inserted before
    the j-th frozen component.
    """

    def __init__(
        self,
        function: ArrayFunction,
        frozen_indexes: IntegerArray,
        frozen_values: NumberArray,
        name: str = "",
    ) -> None:
        """
        Args:
            function: The original function.
            frozen_indexes: The indices of the input components to freeze.
            frozen_values: The values of the frozen input components.
            name: The name of the restricted function.
                If empty, defaults to `"{function.name}_restriction"`.

        Raises:
            ValueError: If `frozen_indexes` and `frozen_values`
                do not have the same shape,
                or `frozen_indexes` contains duplicates.
        """  # noqa: D205, D212, D415
        if frozen_indexes.shape != frozen_values.shape:
            msg = "Arrays of frozen indexes and values must have the same shape."
            raise ValueError(msg)

        if len(set(frozen_indexes)) != len(frozen_indexes):
            msg = "frozen_indexes must contain unique indices."
            raise ValueError(msg)

        sort_order = argsort(frozen_indexes)
        self.__frozen_indexes = frozen_indexes[sort_order]
        self.__frozen_values = frozen_values[sort_order]
        self.__insert_obj = self.__frozen_indexes - arange(
            len(self.__frozen_indexes), dtype=self.__frozen_indexes.dtype
        )
        self.__function = function

        frozen_index_set = set(self.__frozen_indexes.tolist())
        active_input_names = [
            n for i, n in enumerate(function.input_names) if i not in frozen_index_set
        ]

        super().__init__(
            self._func_to_wrap,
            name or f"{function.name}_restriction",
            jac=self._jac_to_wrap if function.has_jac else None,
            f_type=function.f_type,
            expr=function.expr,
            input_names=active_input_names,
            dim=function.dim,
            output_names=function.output_names,
            force_real=function.force_real,
            original_name=function.original_name,
        )

    def _func_to_wrap(self, x_sub: NumberArray) -> NumberArray:
        """Evaluate the original function after reinserting the frozen components.

        Args:
            x_sub: The values of the active (non-frozen) input components.

        Returns:
            The value of the original function at the reconstructed full input vector.
        """
        x_full = insert(x_sub, self.__insert_obj, self.__frozen_values)
        return self.__function.func(x_full)

    def _jac_to_wrap(self, x_sub: NumberArray) -> NumberArray:
        """Compute the Jacobian restricted to the active input components.

        Args:
            x_sub: The values of the active (non-frozen) input components.

        Returns:
            The Jacobian of the original function
            evaluated at the reconstructed full input vector,
            with the columns corresponding to frozen components removed.
        """
        x_full = insert(x_sub, self.__insert_obj, self.__frozen_values)
        return delete(self.__function.jac(x_full), self.__frozen_indexes, axis=-1)
