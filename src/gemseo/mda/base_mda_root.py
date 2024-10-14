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

from typing import TYPE_CHECKING

from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.parallel_execution.disc_parallel_execution import DiscParallelExecution
from gemseo.core.parallel_execution.disc_parallel_linearization import (
    DiscParallelLinearization,
)
from gemseo.mda.base_mda_solver import BaseMDASolver
from gemseo.utils.constants import N_CPUS
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.coupling_structure import CouplingStructure
    from gemseo.core.discipline import Discipline
    from gemseo.typing import StrKeyMapping


class BaseMDARoot(BaseMDASolver):
    """Abstract class implementing MDAs based on (Quasi-)Newton methods."""

    __n_processes: int
    """The maximum number of threads or processes for parallel execution."""

    _parallel_execution: DiscParallelExecution | None
    """Either an executor of disciplines in parallel or ``None`` in serial mode."""

    _parallel_linearization: DiscParallelLinearization | None
    """Either an linearizor of disciplines in parallel or ``None`` in serial mode."""

    __use_threading: bool
    """Whether to use threads instead of processes to parallelize the execution."""

    _execute_before_linearizing: bool
    """Whether to start by executing the discipline before linearizing them."""

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        max_mda_iter: int = 10,
        name: str = "",
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        coupling_structure: CouplingStructure | None = None,
        log_convergence: bool = False,
        linear_solver: str = "DEFAULT",
        linear_solver_options: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        use_threading: bool = True,
        n_processes: int = N_CPUS,
        acceleration_method: AccelerationMethod = AccelerationMethod.NONE,
        over_relaxation_factor: float = 1.0,
        execute_before_linearizing: bool = False,
    ) -> None:
        """
        Args:
            n_processes: The maximum simultaneous number of threads if ``use_threading``
                is set to True, otherwise processes, used to parallelize the execution.
            use_threading: Whether to use threads instead of processes to parallelize
                the execution. Processes will copy (serialize) the disciplines,
                while threads will share the memory. If one wants to execute the
                same discipline multiple times then multiprocessing should be preferred.
            execute_before_linearizing: Whether to start by executing the disciplines
                with the input data for which to compute the Jacobian;
                this allows to ensure that the discipline were executed
                with the right input data;
                it can be almost free if the corresponding output data
                have been stored in the :attr:`.cache`.
        """  # noqa:D205 D212 D415
        self.__use_threading = use_threading
        self.__n_processes = n_processes
        self._execute_before_linearizing = execute_before_linearizing
        if n_processes > 1:
            self._execute_disciplines = self._execute_disciplines_in_parallel
            self._linearize_disciplines = self._linearize_disciplines_in_parallel
            self._parallel_execution = DiscParallelExecution(
                disciplines,
                n_processes,
                use_threading=use_threading,
            )
            self._parallel_linearization = DiscParallelLinearization(
                disciplines,
                n_processes,
                use_threading=use_threading,
                execute=execute_before_linearizing,
            )
        else:
            self._execute_disciplines = self._execute_disciplines_sequentially
            self._linearize_disciplines = self._linearize_disciplines_sequentially
            self._parallel_execution = None
            self._parallel_linearization = False

        super().__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
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
        self._set_resolved_variables(self.strong_couplings)

    def _linearize_disciplines_sequentially(self, input_data: StrKeyMapping) -> None:
        """Linearize the disciplines sequentially.

        Args:
            input_data: The input data to execute the disciplines.
        """
        for discipline in self.disciplines:
            discipline.linearize(input_data, execute=self._execute_before_linearizing)

    def _linearize_disciplines_in_parallel(
        self,
        input_data: StrKeyMapping,
    ) -> None:
        """Linearize the disciplines in parallel.

        Args:
            input_data: The input data to execute the disciplines.
        """
        self._parallel_linearization.execute([input_data] * len(self.disciplines))

    def _execute_disciplines_and_update_local_data(
        self, input_data: StrKeyMapping = READ_ONLY_EMPTY_DICT
    ) -> None:
        input_data = input_data or self.io.data
        self._execute_disciplines(input_data)
        for discipline in self.disciplines:
            self.io.data.update(discipline.get_output_data())

    def _execute_disciplines_in_parallel(self, input_data: StrKeyMapping) -> None:
        """Execute the discipline in parallel.

        Args:
            input_data: The input data of the disciplines.
        """
        self._parallel_execution.execute([input_data] * len(self.disciplines))

    def _execute_disciplines_sequentially(self, input_data: StrKeyMapping) -> None:
        """Execute the discipline sequentially.

        Args:
            input_data: The input data of the disciplines.
        """
        for discipline in self.disciplines:
            discipline.execute(input_data)
