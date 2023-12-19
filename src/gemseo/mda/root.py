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
#        :author: Charlie Vanaret, Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base module for Newton algorithm variants for solving MDAs."""

from __future__ import annotations

from multiprocessing import cpu_count
from typing import TYPE_CHECKING

from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.discipline import MDODiscipline
from gemseo.core.parallel_execution.disc_parallel_execution import DiscParallelExecution
from gemseo.core.parallel_execution.disc_parallel_linearization import (
    DiscParallelLinearization,
)
from gemseo.mda.base_mda_solver import BaseMDASolver

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any
    from typing import Final

    from numpy.typing import NDArray

    from gemseo.core.coupling_structure import MDOCouplingStructure

N_CPUS: Final[int] = cpu_count()


class MDARoot(BaseMDASolver):
    """Abstract class implementing MDAs based on (Quasi-)Newton methods."""

    n_processes: int
    """The maximum number of simultaneous threads,  if :attr:`.use_threading` is True,
    or processes otherwise, used to parallelize the execution."""

    use_threading: bool
    """Whether to use threads instead of processes to parallelize the execution."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        max_mda_iter: int = 10,
        name: str | None = None,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        coupling_structure: MDOCouplingStructure | None = None,
        log_convergence: bool = False,
        linear_solver: str = "DEFAULT",
        linear_solver_options: Mapping[str, Any] | None = None,
        parallel: bool = False,
        use_threading: bool = True,
        n_processes: int = N_CPUS,
        acceleration_method: AccelerationMethod = AccelerationMethod.NONE,
        over_relaxation_factor: float = 1.0,
    ) -> None:
        """
        Args:
            parallel: Whether to execute and linearize the disciplines in parallel.
            n_processes: The maximum simultaneous number of threads if ``use_threading``
                is set to True, otherwise processes, used to parallelize the execution.
            use_threading: Whether to use threads instead of processes to parallelize
                the execution. Processes will copy (serialize) all the disciplines,
                while threads will share all the memory. If one wants to execute the
                same discipline multiple times then multiprocessing should be preferred.
        """  # noqa:D205 D212 D415
        self.use_threading = use_threading
        self.n_processes = n_processes
        self.parallel = parallel

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

        self._compute_input_couplings()
        self._set_resolved_variables(self.strong_couplings)

    def linearize_all_disciplines(
        self, input_data: Mapping[str, NDArray], execute: bool = True
    ) -> None:
        """Linearize all disciplines.

        Args:
            input_data: The input data to be passed to the disciplines.
            execute: Whether to start by executing the discipline
                with the input data for which to compute the Jacobian;
                this allows to ensure that the discipline was executed
                with the right input data;
                it can be almost free if the corresponding output data
                have been stored in the :attr:`.cache`.
        """
        disciplines = self.coupling_structure.disciplines
        if self.parallel:
            parallel_linearization = DiscParallelLinearization(
                disciplines,
                self.n_processes,
                use_threading=self.use_threading,
                execute=execute,
            )
            parallel_linearization.execute([input_data] * len(disciplines))
        else:
            for disc in disciplines:
                disc.linearize(input_data, execute=execute)

    def execute_all_disciplines(
        self, input_local_data: Mapping[str, NDArray], update_local_data: bool = True
    ) -> None:
        """Execute all disciplines.

        Args:
            input_local_data: The input data of the disciplines.
            update_local_data: Whether to update the local data from the disciplines.
        """
        if self.parallel:
            parallel_execution = DiscParallelExecution(
                self.disciplines,
                self.n_processes,
                use_threading=self.use_threading,
            )
            parallel_execution.execute([input_local_data] * len(self.disciplines))
        else:
            for discipline in self.disciplines:
                discipline.execute(input_local_data)

        if update_local_data:
            for discipline in self.disciplines:
                self.local_data.update(discipline.get_output_data())
