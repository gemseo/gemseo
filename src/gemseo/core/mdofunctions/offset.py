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

from numbers import Number

from gemseo.core.mdofunctions.mdo_function import ArrayType
from gemseo.core.mdofunctions.mdo_function import MDOFunction


class Offset(MDOFunction):
    """Wrap an MDOFunction plus an offset value."""

    def __init__(self, value: ArrayType | Number, mdo_function: MDOFunction) -> None:
        """
        Args:
            value: The offset value.
            mdo_function: The original MDOFunction object.
        """  # noqa: D205, D212, D415
        function = mdo_function.offset(value)
        super().__init__(
            function.func,
            function.name,
            f_type=function.f_type,
            jac=function.jac,
            expr=function.expr,
            args=function.args,
            dim=function.dim,
            outvars=function.outvars,
            force_real=function.force_real,
            special_repr=function.special_repr,
        )
