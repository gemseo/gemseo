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
"""Additive discipline chain."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.chains.parallel_chain import MDOParallelChain

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline


class MDOAdditiveChain(MDOParallelChain):
    """Execute disciplines in parallel and sum specified outputs across disciplines."""

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        outputs_to_sum: Iterable[str],
        name: str = "",
        use_threading: bool = True,
        n_processes: int | None = None,
    ) -> None:
        """
        Args:
            disciplines: The disciplines.
            outputs_to_sum: The names of the outputs to sum.
            name: The name of the discipline.
                If ``None``, use the class name.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory.
                This is important to note
                if you want to execute the same discipline multiple times,
                you shall use multiprocessing.
            n_processes: The maximum simultaneous number of threads,
                if ``use_threading`` is True, or processes otherwise,
                used to parallelize the execution.
                If ``None``, uses the number of disciplines.

        Notes:
            The actual number of processes could be lower than ``n_processes``
            if there are less than ``n_processes`` disciplines.
            ``n_processes`` can be lower than the total number of CPUs on the machine.
            Each discipline may itself run on several CPUs.
        """  # noqa: D205, D212, D415
        super().__init__(disciplines, name, use_threading, n_processes)
        self._outputs_to_sum = outputs_to_sum

    def _execute(self) -> None:
        # Run the disciplines in parallel
        super()._execute()

        # Sum the required outputs across disciplines
        for output_name in self._outputs_to_sum:
            disciplinary_outputs = [
                discipline.io.data[output_name]
                for discipline in self.disciplines
                if output_name in discipline.io.data
            ]
            self.io.data[output_name] = (
                sum(disciplinary_outputs) if disciplinary_outputs else None
            )

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        # Differentiate the disciplines in parallel
        super()._compute_jacobian(input_names, output_names)

        # Sum the Jacobians of the required outputs across disciplines
        for output_name in self._outputs_to_sum:
            self.jac[output_name] = {}
            for input_name in input_names:
                disciplinary_jacobians = [
                    discipline.jac[output_name][input_name]
                    for discipline in self.disciplines
                    if input_name in discipline.jac[output_name]
                ]

                assert disciplinary_jacobians
                self.jac[output_name][input_name] = sum(disciplinary_jacobians)
