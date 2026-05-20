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
"""A function evaluating another one with an offset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.functions.array_function import ArrayFunction

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class Offset(ArrayFunction):
    """Wrap an ArrayFunction plus an offset value."""

    def __init__(self, value: NumberArray | complex, function: ArrayFunction) -> None:
        """
        Args:
            value: The offset value.
            function: The original function.
        """  # noqa: D205, D212, D415
        offset_function = function.offset(value)
        super().__init__(
            offset_function.func,
            offset_function.name,
            f_type=offset_function.f_type,
            jac=offset_function.jac,
            expr=offset_function.expr,
            input_names=offset_function.input_names,
            dim=offset_function.dim,
            output_names=offset_function.output_names,
            force_real=offset_function.force_real,
            special_repr=offset_function.special_repr,
        )
