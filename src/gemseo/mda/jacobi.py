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

from copy import deepcopy
from multiprocessing import cpu_count
from typing import Any
from typing import Mapping
from typing import Sequence

from numpy import atleast_2d
from numpy import concatenate
from numpy import dot
from numpy import ndarray
from numpy.linalg import lstsq

from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.execution_sequence import LoopExecSequence
from gemseo.core.parallel_execution import DiscParallelExecution
from gemseo.mda.mda import MDA
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

N_CPUS = cpu_count()


class MDAJacobi(MDA):
    """Perform an MDA analysis using a Jacobi algorithm.

    This algorithm is an iterative technique to solve the linear system:

    .. math::

       Ax = b

    by decomposing the matrix :math:`A`
    into the sum of a diagonal matrix :math:`D`
    and the reminder :math:`R`.

    The new iterate is given by:

    .. math::

       x_{k+1} = D^{-1}(b-Rx_k)
    """

    SECANT_ACCELERATION = "secant"
    M2D_ACCELERATION = "m2d"

    _ATTR_TO_SERIALIZE = MDA._ATTR_TO_SERIALIZE + (
        "parallel_execution",
        "sizes",
        "acceleration",
        "n_processes",
    )

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        max_mda_iter: int = 10,
        name: str | None = None,
        n_processes: int = N_CPUS,
        acceleration: str = M2D_ACCELERATION,
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        use_threading: bool = True,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
        coupling_structure: MDOCouplingStructure | None = None,
        log_convergence: bool = False,
        linear_solver: str = "DEFAULT",
        linear_solver_options: Mapping[str, Any] = None,
    ) -> None:
        """
        Args:
            n_processes: The maximum simultaneous number of threads,
                if ``use_threading`` is True, or processes otherwise,
                used to parallelize the execution.
            acceleration: The type of acceleration
                to be used to extrapolate the residuals
                and save CPU time by reusing the information from the last iterations,
                either ``None``, ``"m2d"``, or ``"secant"``,
                ``"m2d"`` is faster but uses the 2 last iterations.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory.
                This is important to note
                if you want to execute the same discipline multiple times,
                you shall use multiprocessing.
        """
        self.n_processes = n_processes
        super().__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            use_lu_fact=use_lu_fact,
            grammar_type=grammar_type,
            coupling_structure=coupling_structure,
            log_convergence=log_convergence,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
        )
        self._set_default_inputs()
        self._compute_input_couplings()
        self.acceleration = acceleration
        self._dx_n = []
        self._g_x_n = []
        self.sizes = None
        self.parallel_execution = DiscParallelExecution(
            disciplines,
            n_processes,
            use_threading,
            exceptions_to_re_raise=(ValueError,),
        )

    def _compute_input_couplings(self) -> None:
        """Compute all the coupling variables that are inputs of the MDA.

        This must be overloaded here because the Jacobi algorithm induces a delay between
        the couplings, the strong couplings may be fully resolved but the weak ones may
        need one more iteration. The base MDA class uses strong couplings only which is
        not satisfying here if all disciplines are not strongly coupled.
        """
        if len(self.coupling_structure.strongly_coupled_disciplines) == len(
            self.disciplines
        ):
            return super()._compute_input_couplings()

        inputs = self.get_input_data_names()
        strong_cpl = self.coupling_structure.all_couplings
        self._input_couplings = set(strong_cpl) & set(inputs)

    def execute_all_disciplines(
        self,
        input_local_data: Mapping[str, ndarray],
    ) -> None:
        """Execute all the disciplines.

        Args:
            input_local_data: The input data of the disciplines.
        """
        self.reset_disciplines_statuses()
        if self.n_processes > 1:
            self.parallel_execution.execute(
                [deepcopy(input_local_data) for _ in range(len(self.disciplines))]
            )
        else:
            for discipline in self.disciplines:
                discipline.execute(deepcopy(input_local_data))

        for discipline in self.disciplines:
            self.local_data.update(discipline.get_output_data())

    def get_expected_workflow(self) -> LoopExecSequence:
        sub_workflow = ExecutionSequenceFactory.serial(self.disciplines)
        if self.n_processes > 1:
            sub_workflow = ExecutionSequenceFactory.parallel(self.disciplines)
        return ExecutionSequenceFactory.loop(self, sub_workflow)

    def _run(self) -> None:
        """Execute all disciplines in a loop until outputs converge.

        Stops when:

        .. math::

            ||outputs-previous output||/||first outputs|| < self.tolerance
        """
        if self.warm_start:
            self._couplings_warm_start()
        self._dx_n = []
        self._g_x_n = []
        # execute the disciplines
        current_couplings = self._current_input_couplings()
        self.execute_all_disciplines(deepcopy(self.local_data))
        new_couplings = self._current_input_couplings()
        self._dx_n.append(new_couplings - current_couplings)
        self._g_x_n.append(new_couplings)

        self._compute_residual(
            current_couplings,
            new_couplings,
            log_normed_residual=self._log_convergence,
        )
        current_couplings = new_couplings

        while not self._stop_criterion_is_reached:
            self.execute_all_disciplines(deepcopy(self.local_data))
            new_couplings = self._current_input_couplings()

            self._compute_residual(
                current_couplings,
                new_couplings,
                log_normed_residual=self._log_convergence,
            )
            x_np1 = self._compute_nex_iterate(current_couplings, new_couplings)
            current_couplings = x_np1

    def _compute_nex_iterate(
        self,
        current_couplings: ndarray,
        new_couplings: ndarray,
    ) -> dict[str, ndarray]:
        """Compute the next iterate given the evaluation of the couplings.

        Eventually compute the convergence acceleration term
        according to the secant or m2d methods.

        See:
        Iterative residual-based vector methods to accelerate
        fixed point iterations, Isabelle Ramiere, Thomas Helfer

        Args:
            current_couplings: The input couplings of the disciplines
                given for evaluation at the last iterations.
            current_couplings: The computed couplings of the disciplines
                at the last iterations.

        Returns:
            The next iterate.
        """

        self._dx_n.append(new_couplings - current_couplings)
        self._g_x_n.append(new_couplings)
        coupl_names = self._input_couplings
        if self.sizes is None:
            self.sizes = {key: self.local_data[key].size for key in coupl_names}
        dxn = self._dx_n[-1]
        dxn_1 = self._dx_n[-2]
        g_n = self._g_x_n[-1]
        gn_1 = self._g_x_n[-2]
        x_np1 = new_couplings
        if self.acceleration == self.SECANT_ACCELERATION:
            x_np1 = self._compute_secant_acc(dxn, dxn_1, g_n, gn_1)
        elif self.acceleration == self.M2D_ACCELERATION:
            if len(self._dx_n) >= 3:
                dxn_2 = self._dx_n[-3]
                dgn_2 = self._g_x_n[-3]
            else:
                dxn_2 = self._dx_n[-2]
                dgn_2 = self._g_x_n[-2]

            x_np1 = self._compute_m2d_acc(dxn, dxn_1, dxn_2, g_n, gn_1, dgn_2)

        if len(self._dx_n) > 3:  # Forget too old stuff
            self._dx_n = self._dx_n[-3:]
            self._g_x_n = self._g_x_n[-3:]
        new_c = split_array_to_dict_of_arrays(x_np1, self.sizes, coupl_names)
        self.local_data.update(new_c)
        return x_np1

    @staticmethod
    def _minimize_2md(
        dxn: ndarray,
        dxn_1: ndarray,
        dxn_2: ndarray,
    ) -> ndarray:
        """Compute the next iterate according to the m2d method.

        Minimize the sub-problem in the d-2 method.
        Use the least squares solver to find he minimizer of:

        .. math::

            dxn - x[0] * (dxn - dxn_1) - x[1] * (dxn_1 - dxn_2)

        Args:
            dxn: The delta couplings at last iteration.
            dxn_1: The delta couplings at last iteration-1.
            dxn_2: The delta couplings at last iteration-2.

        Returns:
            The extrapolation coefficients of the 2-delta method.
        """
        mat = concatenate((atleast_2d(dxn - dxn_1), atleast_2d(dxn_1 - dxn_2)))
        return lstsq(mat.T, dxn, rcond=None)[0]

    @staticmethod
    def _compute_secant_acc(
        dxn: ndarray,
        dxn_1: ndarray,
        cgn: ndarray,
        cgn_1: ndarray,
    ) -> ndarray:
        """Compute the next iterate according to the secant method.

        From the paper:
        "Iterative residual-based vector methods to accelerate
        fixed point iterations",  Isabelle Ramiere, Thomas Helfer
        (secant acceleration: page 15 equation (41)).

        Args:
            dxn: The delta couplings at last iteration.
            dxn_1: The delta couplings at last iteration-1.
            cgn: The computed couplings at last iteration.
            cgn_1: The computed couplings at last iteration-1.

        Returns:
            The next iterate.
        """
        d_dxn = dxn - dxn_1
        acc = (cgn - cgn_1) * dot(d_dxn, dxn) / dot(d_dxn, d_dxn)
        return cgn - acc

    def _compute_m2d_acc(
        self,
        dxn: ndarray,
        dxn_1: ndarray,
        dxn_2: ndarray,
        g_n: ndarray,
        gn_1: ndarray,
        gn_2: ndarray,
    ) -> ndarray:
        """Compute the 2-delta acceleration.

        From the paper:
        "Iterative residual-based vector methods to accelerate
        fixed point iterations",  Isabelle Ramiere, Thomas Helfer
        page 22 eq (50)

        Args:
            dxn: The delta couplings at last iteration.
            dxn_1: The delta couplings at last iteration-1.
            dxn_2: The delta couplings at last iteration-2.
            g_n: The computed couplings at last iteration.
            gn_1: The computed couplings at last iteration-1.
            gn_2: The computed couplings at last iteration-2.

        Returns:
            The next iterate.
        """
        lamba_min = self._minimize_2md(dxn, dxn_1, dxn_2)
        acc = lamba_min[0] * (g_n - gn_1) + lamba_min[1] * (gn_1 - gn_2)
        return g_n - acc
