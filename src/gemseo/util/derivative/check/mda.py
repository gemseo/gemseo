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

"""Jacobian checker for MDAs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.util.derivative.check.discipline import DisciplineJacobianChecker

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.mda.core.base import BaseMDA


class MDAJacobianChecker(DisciplineJacobianChecker):
    """Checks the Jacobian of a `BaseMDA` by numerical approximation."""

    def __init__(self, mda: BaseMDA) -> None:
        """
        Args:
            mda: The MDA whose Jacobian is to be checked.
        """  # noqa: D205, D212
        super().__init__(mda)

    def _prepare_io(
        self,
        input_names: Iterable[str],
        output_names: Iterable[str],
    ) -> tuple[list[str], list[str]]:
        input_names, output_names = super()._prepare_io(input_names, output_names)
        mda = self._discipline
        couplings = set(mda.coupling_structure.all_couplings)
        input_names = [n for n in input_names if n not in couplings]
        output_names = [n for n in output_names if n not in couplings]
        if mda.NORMALIZED_RESIDUAL_NORM in output_names:
            output_names.remove(mda.NORMALIZED_RESIDUAL_NORM)
        input_names = [
            n for n in input_names if mda.io.input_grammar.data_converter.is_numeric(n)
        ]
        output_names = [
            n
            for n in output_names
            if mda.io.output_grammar.data_converter.is_numeric(n)
        ]
        return input_names, output_names
