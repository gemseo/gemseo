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
"""Warm-started disciplines chain."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.chains.chain import MDOChain

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline


class MDOWarmStartedChain(MDOChain):
    """Chain capable of warm starting a given list of variables.

    The values of the variables to warm start are stored after each run and used to
    initialize the next one.

    This chain cannot be linearized.
    """

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        variable_names_to_warm_start: Sequence[str],
        name: str = "",
    ) -> None:
        """
        Args:
            disciplines: The disciplines.
            variable_names_to_warm_start: The names of the variables to be warm started.
                These names must be outputs of the disciplines in the chain.
                If the list is empty, no variables are warm started.
            name: The name of the discipline.
                If ``None``, use the class name.

        Raises:
            ValueError: If the variable names to warm start are not outputs of the
                chain.
        """  # noqa: D205, D212, D415
        super().__init__(disciplines=disciplines, name=name)
        self._variable_names_to_warm_start = variable_names_to_warm_start
        self._warm_start_variable_names_to_values = {}
        if variable_names_to_warm_start and not self.io.output_grammar.has_names(
            variable_names_to_warm_start
        ):
            all_output_names = self.io.output_grammar
            missing_output_names = set(variable_names_to_warm_start).difference(
                all_output_names
            )
            msg = (
                "The following variable names are not "
                f"outputs of the chain: {missing_output_names}."
                f" Available outputs are: {list(all_output_names)}."
            )
            raise ValueError(msg)

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        msg = f"{self.__class__.__name__} cannot be linearized."
        raise NotImplementedError(msg)

    def _execute(self) -> None:
        if self._warm_start_variable_names_to_values:
            self.io.data.update(self._warm_start_variable_names_to_values)
        super()._execute()
        if self._variable_names_to_warm_start:
            self._warm_start_variable_names_to_values = {
                name: self.io.data[name] for name in self._variable_names_to_warm_start
            }
