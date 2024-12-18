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
"""A parallel discipline chain."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import ndarray

from gemseo.core._process_flow.base_process_flow import BaseProcessFlow
from gemseo.core.discipline.discipline_data import DisciplineData
from gemseo.core.parallel_execution.disc_parallel_execution import DiscParallelExecution
from gemseo.core.parallel_execution.disc_parallel_linearization import (
    DiscParallelLinearization,
)
from gemseo.core.process_discipline import ProcessDiscipline
from gemseo.utils.data_conversion import deepcopy_dict_of_arrays

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline


class _ProcessFlow(BaseProcessFlow):
    """The process flow."""

    is_parallel = True


class MDOParallelChain(ProcessDiscipline):
    """Chain of processes that executes disciplines in parallel."""

    _process_flow_class: ClassVar[type[BaseProcessFlow]] = _ProcessFlow

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        name: str = "",
        use_threading: bool = True,
        n_processes: int | None = None,
        use_deep_copy: bool = False,
    ) -> None:
        """
        Args:
            disciplines: The disciplines.
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
            use_deep_copy: Whether to deepcopy the discipline input data.

        Notes:
            The actual number of processes could be lower than ``n_processes``
            if there are less than ``n_processes`` disciplines.
            ``n_processes`` can be lower than the total number of CPUs on the machine.
            Each discipline may itself run on several CPUs.
        """  # noqa: D205, D212, D415
        super().__init__(disciplines, name=name)
        self._use_deep_copy = use_deep_copy
        self._initialize_grammars()
        if n_processes is None:
            n_processes = len(self._disciplines)

        self.parallel_execution = DiscParallelExecution(
            self._disciplines, n_processes, use_threading=use_threading
        )
        self.parallel_lin = DiscParallelLinearization(
            self._disciplines, n_processes, use_threading=use_threading
        )

    def _initialize_grammars(self) -> None:
        """Define the input and output grammars from the disciplines' ones."""
        self.io.input_grammar.clear()
        self.io.output_grammar.clear()
        for discipline in self._disciplines:
            self.io.input_grammar.update(discipline.io.input_grammar)
            self.io.output_grammar.update(discipline.io.output_grammar)

    def _get_input_data_copies(self) -> list[DisciplineData]:
        """Return copies of the input data, one per discipline.

        Returns:
            One copy of the input data per discipline.
        """
        # Avoid overlaps with dicts in // by doing a deepcopy
        # The outputs of a discipline may be a coupling, and shall therefore
        # not be passed as input of another since the execution are assumed
        # to be independent here
        if self._use_deep_copy:
            return [
                DisciplineData(deepcopy_dict_of_arrays(self.io.data))
                for _ in range(len(self._disciplines))
            ]

        for value in self.io.data.values():
            if isinstance(value, ndarray):
                value.flags.writeable = False

        return [self.io.data] * len(self._disciplines)

    def _execute(self) -> None:
        self.parallel_execution.execute(self._get_input_data_copies())

        # Update data according to input order of priority
        for discipline in self._disciplines:
            self.io.data.update({
                output_name: discipline.io.data[output_name]
                for output_name in discipline.io.output_grammar
            })

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        self._set_disciplines_diff_outputs(output_names)
        self._set_disciplines_diff_inputs(input_names)
        jacobians = self.parallel_lin.execute(self._get_input_data_copies())
        self.jac = {}
        # Update jacobians according to input order of priority
        for discipline_jacobian in jacobians:
            for output_name, output_jacobian in discipline_jacobian.items():
                chain_jacobian = self.jac.get(output_name)
                if chain_jacobian is None:
                    chain_jacobian = {}
                    self.jac[output_name] = chain_jacobian
                chain_jacobian.update(output_jacobian)

        self._init_jacobian(
            input_names,
            output_names,
            fill_missing_keys=True,
            init_type=self.InitJacobianType.SPARSE,
        )

    def add_differentiated_inputs(  # noqa: D102
        self,
        input_names: Iterable[str] = (),
    ) -> None:
        super().add_differentiated_inputs(input_names)
        self._set_disciplines_diff_inputs(input_names)

    def _set_disciplines_diff_inputs(
        self,
        input_names: Iterable[str],
    ) -> None:
        """Add the inputs to the right sub discipline's differentiated inputs.

        Args:
            input_names: The names of the inputs to be added.
        """
        diff_inpts = set(input_names)
        for discipline in self._disciplines:
            inputs_set = set(discipline.io.input_grammar) & diff_inpts
            if inputs_set:
                discipline.add_differentiated_inputs(list(inputs_set))

    def add_differentiated_outputs(  # noqa: D102
        self,
        output_names: Iterable[str] = (),
    ) -> None:
        super().add_differentiated_outputs(output_names)
        self._set_disciplines_diff_outputs(output_names)

    def _set_disciplines_diff_outputs(self, output_names: Iterable[str]) -> None:
        """Add the outputs to the right-sub discipline's differentiated outputs.

        Args:
            output_names: The outputs to be added.
        """
        diff_outpts = set(output_names)
        for discipline in self._disciplines:
            outputs_set = set(discipline.io.output_grammar) & diff_outpts
            if outputs_set:
                discipline.add_differentiated_outputs(list(outputs_set))
