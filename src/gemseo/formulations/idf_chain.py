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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard, Matthias De Lozzo
"""Processing strongly coupled disciplines in parallel."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.core.chains.chain import MDOChain
from gemseo.core.chains.chain import _ProcessFlow as _MDOChainProcessFlow
from gemseo.core.chains.parallel_chain import MDOParallelChain
from gemseo.core.coupling_structure import CouplingStructure

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.core._process_flow.base_process_flow import BaseProcessFlow
    from gemseo.core.dependency_graph import ExecutionSequence
    from gemseo.core.discipline import Discipline


class _ProcessFlow(_MDOChainProcessFlow):
    """The process data and execution flow."""

    def _get_disciplines_couplings(
        self, disciplines: Sequence[Discipline]
    ) -> list[tuple[Discipline, Discipline, list[str]]]:
        coupling_structure = CouplingStructure(disciplines)
        strong_couplings = set(coupling_structure.strong_couplings)
        disciplines_couplings = coupling_structure.graph.get_disciplines_couplings()
        new_disciplines_couplings = []
        for source, target, couplings in disciplines_couplings:
            new_couplings = set(couplings) - strong_couplings
            if new_couplings:
                new_disciplines_couplings.append((
                    source,
                    target,
                    sorted(new_couplings),
                ))

        return new_disciplines_couplings


class IDFChain(MDOChain):
    """A discipline for processing strongly coupled disciplines in parallel.

    Given a collection of disciplines,
    a sequence of execution of disciplines and process disciplines is deduced
    from its coupling structure
    under the constraint of running the strongly coupled disciplines in parallel.

    Specifically, the coupling structure provides sequential tasks, where each task is
    a set of uncoupled groups of strongly coupled disciplines.
    The tasks are then performed sequentially,
    by performing their uncoupled groups either in parallel or in serial,
    and the strongly coupled disciplines of these groups in parallel.
    """

    _process_flow_class: ClassVar[type[BaseProcessFlow]] = _ProcessFlow

    def __init__(
        self,
        execution_sequence: ExecutionSequence,
        n_processes: int,
        use_threading: bool,
    ) -> None:
        """
        Args:
            execution_sequence: The execution sequence.
            n_processes: The maximum simultaneous number of threads
                if ``use_threading`` is ``True``,
                or processes otherwise,
                used to parallelize the execution.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory.
                This is important to note
                if you want to execute the same discipline multiple times,
                you shall use multiprocessing.
        """  # noqa: D205 D212
        disciplines = [
            self.__create_discipline(
                uncoupled_groups_of_disciplines,
                n_processes,
                use_threading,
            )
            for uncoupled_groups_of_disciplines in execution_sequence
        ]
        super().__init__(disciplines)

    @staticmethod
    def __create_discipline(
        uncoupled_groups_of_coupled_disciplines: Iterable[tuple[Discipline, ...]],
        n_processes: int,
        use_threading: bool,
    ) -> Discipline:
        """Create a discipline to process uncoupled groups of coupled disciplines.

        In each group,
        the disciplines will be processed in parallel.

        Args:
            uncoupled_groups_of_coupled_disciplines: Uncoupled groups
                of coupled disciplines.
            n_processes: The maximum simultaneous number of threads
                to parallelize the execution.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory.
                This is important to note
                if you want to execute the same discipline multiple times,
                you shall use multiprocessing.

        Returns:
            A discipline to process uncoupled groups of strongly coupled disciplines.
        """
        uncoupled_disciplines = [
            MDOParallelChain(
                coupled_disciplines,
                n_processes=n_processes,
                use_threading=use_threading,
            )
            if len(coupled_disciplines) > 1
            else coupled_disciplines[0]
            for coupled_disciplines in uncoupled_groups_of_coupled_disciplines
        ]
        if len(uncoupled_disciplines) == 1:
            return uncoupled_disciplines[0]

        if n_processes > 1:
            return MDOParallelChain(
                uncoupled_disciplines,
                n_processes=n_processes,
                use_threading=use_threading,
            )

        return MDOChain(uncoupled_disciplines)
