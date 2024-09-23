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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A Jacobi algorithm for solving MDAs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.parallel_execution.disc_parallel_execution import DiscParallelExecution
from gemseo.mda.base_mda_solver import BaseMDASolver
from gemseo.utils.constants import N_CPUS
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.coupling_structure import CouplingStructure
    from gemseo.core.coupling_structure import DependencyGraph
    from gemseo.core.execution_sequence import LoopExecSequence
    from gemseo.typing import StrKeyMapping


class MDAJacobi(BaseMDASolver):
    r"""Perform an MDA using the Jacobi algorithm.

    This algorithm is a fixed point iteration method to solve systems of non-linear
    equations of the form,

    .. math::

        \left\{
            \begin{matrix}
                F_1(x_1, x_2, \dots, x_n) = 0 \\
                F_2(x_1, x_2, \dots, x_n) = 0 \\
                \vdots \\
                F_n(x_1, x_2, \dots, x_n) = 0
            \end{matrix}
        \right.

    Beginning with :math:`x_1^{(0)}, \dots, x_n^{(0)}`, the iterates are obtained as the
    solution of the following :math:`n` **independent** non-linear equations:

    .. math::

        \left\{
            \begin{matrix}
                r_1\left( x_1^{(i+1)} \right) =
                    F_1(x_1^{(i+1)}, x_2^{(i)}, \dots, x_n^{(i)}) = 0 \\
                r_2\left( x_2^{(i+1)} \right) =
                    F_2(x_1^{(i)}, x_2^{(i+1)}, \dots, x_n^{(i)}) = 0 \\
                \vdots \\
                r_n\left( x_n^{(i+1)} \right) =
                F_n(x_1^{(i)}, x_2^{(i)}, \dots, x_n^{(i+1)}) = 0
            \end{matrix}
        \right.
    """

    __n_processes: int
    """The maximum number of threads or processes for parallel execution."""

    parallel_execution: DiscParallelExecution | None
    """Either an executor of disciplines in parallel or ``None`` in serial mode."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        max_mda_iter: int = 10,
        name: str = "",
        n_processes: int = N_CPUS,
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        use_threading: bool = True,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        coupling_structure: CouplingStructure | None = None,
        log_convergence: bool = False,
        linear_solver: str = "DEFAULT",
        linear_solver_options: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        acceleration_method: AccelerationMethod = AccelerationMethod.ALTERNATE_2_DELTA,
        over_relaxation_factor: float = 1.0,
    ) -> None:
        """
        Args:
            n_processes: The maximum simultaneous number of threads if ``use_threading``
                is set to True, otherwise processes, used to parallelize the execution.
            use_threading: Whether to use threads instead of processes to parallelize
                the execution. Processes will copy (serialize) the disciplines,
                while threads will share the memory. If one wants to execute the
                same discipline multiple times then multiprocessing should be preferred.
        """  # noqa:D205 D212 D415
        self.__n_processes = n_processes
        super().__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            grammar_type=grammar_type,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            use_lu_fact=use_lu_fact,
            coupling_structure=coupling_structure,
            log_convergence=log_convergence,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
            acceleration_method=acceleration_method,
            over_relaxation_factor=over_relaxation_factor,
        )

        self._compute_input_coupling_names()
        self._set_resolved_variables(self._input_couplings)
        if n_processes == 1:
            self._execute_disciplines = self._execute_disciplines_sequentially
            self.parallel_execution = None
        else:
            self._execute_disciplines = self._execute_disciplines_in_parallel
            self.parallel_execution = DiscParallelExecution(
                disciplines,
                n_processes,
                use_threading,
                exceptions_to_re_raise=(ValueError,),
            )

    def _compute_input_coupling_names(self) -> None:
        """Compute the coupling variables that are inputs of the MDA.

        This must be overloaded here because the Jacobi algorithm induces a delay
        between the couplings, the strong couplings may be fully resolved but the weak
        ones may need one more iteration. The base MDA class uses strong couplings only,
        which is not satisfying here if some disciplines are not strongly coupled.
        """
        if len(self.coupling_structure.strongly_coupled_disciplines) == len(
            self.disciplines
        ):
            return super()._compute_input_coupling_names()

        self._input_couplings = sorted(
            set(self.coupling_structure.all_couplings).intersection(
                self.get_input_data_names()
            )
        )

        self._numeric_input_couplings = sorted(
            set(self._input_couplings).difference(self._non_numeric_array_variables)
        )

        return None

    def _execute_disciplines_in_parallel(self) -> None:
        """Execute the disciplines in parallel."""
        self.parallel_execution.execute([self.local_data] * len(self.disciplines))

    def _execute_disciplines_sequentially(self) -> None:
        """Execute the disciplines sequentially."""
        for discipline in self.disciplines:
            discipline.execute(self.local_data)

    def _execute_disciplines_and_update_local_data(
        self, input_data: StrKeyMapping = READ_ONLY_EMPTY_DICT
    ) -> None:
        self.reset_disciplines_statuses()
        self._execute_disciplines()
        for discipline in self.disciplines:
            self.local_data.update(discipline.get_output_data())

    def get_expected_workflow(self) -> LoopExecSequence:  # noqa:D102
        if self.parallel_execution is None:
            sub_workflow = ExecutionSequenceFactory.serial()
        else:
            sub_workflow = ExecutionSequenceFactory.parallel()

        for discipline in self.disciplines:
            sub_workflow.extend(discipline.get_expected_workflow())

        return ExecutionSequenceFactory.loop(self, sub_workflow)

    def _get_disciplines_couplings(
        self, graph: DependencyGraph
    ) -> list[tuple[MDODiscipline, MDAJacobi, list[str]]]:
        disciplines_couplings = []
        get_edge_data = graph.graph.get_edge_data
        for discipline in self.disciplines:
            for source, target in ((self, discipline), (discipline, self)):
                edge_data = get_edge_data(source, target)
                if edge_data:
                    disciplines_couplings.append((
                        source,
                        target,
                        sorted(edge_data.get(graph.IO)),
                    ))

        return disciplines_couplings

    def _run(self) -> None:
        super()._run()

        while True:
            local_data_before_execution = self.local_data.copy()
            self._execute_disciplines_and_update_local_data()
            self._compute_residuals(local_data_before_execution)

            if self._stop_criterion_is_reached:
                break

            updated_couplings = self._sequence_transformer.compute_transformed_iterate(
                self.get_current_resolved_variables_vector(),
                self.get_current_resolved_residual_vector(),
            )

            self._update_local_data_from_array(updated_couplings)
