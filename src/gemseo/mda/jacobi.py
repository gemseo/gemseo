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

from multiprocessing import cpu_count
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.parallel_execution.disc_parallel_execution import DiscParallelExecution
from gemseo.mda.base_mda_solver import BaseMDASolver

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import NDArray

    from gemseo.core.coupling_structure import MDOCouplingStructure
    from gemseo.core.execution_sequence import LoopExecSequence


N_CPUS: Final[int] = cpu_count()


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

    # TODO: API: Remove the class attributes.
    SECANT_ACCELERATION: ClassVar[str] = "secant"
    M2D_ACCELERATION: ClassVar[str] = "m2d"

    # TODO: API: Remove the compatibility mapping.
    __ACCELERATION_COMPATIBILITY: Final[dict[str, AccelerationMethod | None]] = {
        M2D_ACCELERATION: AccelerationMethod.ALTERNATE_2_DELTA,
        SECANT_ACCELERATION: AccelerationMethod.SECANT,
        "": None,
    }

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        max_mda_iter: int = 10,
        name: str | None = None,
        n_processes: int = N_CPUS,
        acceleration: str = "",  # TODO: API: Remove this argument.
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        use_threading: bool = True,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        coupling_structure: MDOCouplingStructure | None = None,
        log_convergence: bool = False,
        linear_solver: str = "DEFAULT",
        linear_solver_options: Mapping[str, Any] | None = None,
        acceleration_method: AccelerationMethod = AccelerationMethod.ALTERNATE_2_DELTA,
        over_relaxation_factor: float = 1.0,
    ) -> None:
        """
        Args:
            acceleration: Deprecated, please consider using the
                :attr:`MDA.acceleration_method` instead.
                The type of acceleration to be used to extrapolate the residuals and
                save CPU time by reusing the information from the last iterations,
                either ``None``, ``"m2d"``, or ``"secant"``, ``"m2d"`` is faster but
                uses the 2 last iterations.
            n_processes: The maximum simultaneous number of threads if ``use_threading``
                is set to True, otherwise processes, used to parallelize the execution.
            use_threading: Whether to use threads instead of processes to parallelize
                the execution. Processes will copy (serialize) all the disciplines,
                while threads will share all the memory. If one wants to execute the
                same discipline multiple times then multiprocessing should be prefered.
        """  # noqa:D205 D212 D415
        self.n_processes = n_processes

        # TODO: API: Remove the old names and attributes for acceleration.
        if self.__ACCELERATION_COMPATIBILITY[acceleration]:
            acceleration_method = self.__ACCELERATION_COMPATIBILITY[acceleration]

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
        self._set_resolved_variables(self._input_couplings)

        self.parallel_execution = DiscParallelExecution(
            disciplines,
            n_processes,
            use_threading,
            exceptions_to_re_raise=(ValueError,),
        )

    # TODO: API: Remove the property and its setter.
    @property
    def acceleration(self) -> AccelerationMethod:
        """The acceleration method."""
        return self.acceleration_method

    @acceleration.setter
    def acceleration(self, acceleration: str) -> None:
        self.acceleration_method = self.__ACCELERATION_COMPATIBILITY[acceleration]

    def _compute_input_couplings(self) -> None:
        """Compute all the coupling variables that are inputs of the MDA.

        This must be overloaded here because the Jacobi algorithm induces a delay
        between the couplings, the strong couplings may be fully resolved but the weak
        ones may need one more iteration. The base MDA class uses strong couplings only
        which is not satisfying here if all disciplines are not strongly coupled.
        """
        if len(self.coupling_structure.strongly_coupled_disciplines) == len(
            self.disciplines
        ):
            return super()._compute_input_couplings()

        self._input_couplings = sorted(
            set(self.coupling_structure.all_couplings).intersection(
                self.get_input_data_names()
            )
        )
        return None

    def execute_all_disciplines(self, input_local_data: Mapping[str, NDArray]) -> None:
        """Execute all the disciplines, possibly in parallel.

        Args:
            input_local_data: The input data of the disciplines.
        """
        self.reset_disciplines_statuses()

        if self.n_processes > 1:
            self.parallel_execution.execute([input_local_data] * len(self.disciplines))
        else:
            for discipline in self.disciplines:
                discipline.execute(input_local_data)

        for discipline in self.disciplines:
            self.local_data.update(discipline.get_output_data())

    def get_expected_workflow(self) -> LoopExecSequence:  # noqa:D102
        if self.n_processes > 1:
            sub_workflow = ExecutionSequenceFactory.parallel()
        else:
            sub_workflow = ExecutionSequenceFactory.serial()

        for discipline in self.disciplines:
            sub_workflow.extend(discipline.get_expected_workflow())

        return ExecutionSequenceFactory.loop(self, sub_workflow)

    def _run(self) -> None:
        super()._run()

        while True:
            input_data = self.local_data.copy()

            self.execute_all_disciplines(self.local_data)
            self._update_residuals(input_data)

            new_couplings = self._sequence_transformer.compute_transformed_iterate(
                self.get_current_resolved_variables_vector(),
                self.get_current_resolved_residual_vector(),
            )

            self._update_local_data(new_couplings)
            self._update_residuals(input_data)
            self._compute_residual(log_normed_residual=self._log_convergence)

            if self._stop_criterion_is_reached:
                break
