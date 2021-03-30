# -*- coding: utf-8 -*-
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
"""
A Jacobi algorithm for solving MDAs
***********************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import range, super
from copy import deepcopy
from multiprocessing import cpu_count

from future import standard_library
from numpy import atleast_2d, concatenate, dot
from numpy.linalg import lstsq

from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.parallel_execution import DiscParallelExecution
from gemseo.mda.mda import MDA
from gemseo.utils.data_conversion import DataConversion

standard_library.install_aliases()

from gemseo import LOGGER

N_CPUS = cpu_count()


class MDAJacobi(MDA):
    """
    Perform a MDA analysis using a Jacobi algorithm,
    an iterative technique to solve the linear system:

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

    def __init__(
        self,
        disciplines,
        max_mda_iter=10,
        name=None,
        n_processes=N_CPUS,
        acceleration=M2D_ACCELERATION,
        tolerance=1e-6,
        linear_solver_tolerance=1e-12,
        use_threading=True,
        warm_start=False,
        use_lu_fact=False,
        norm0=None,
    ):
        """
        Constructor

        :param disciplines: the disciplines list
        :type disciplines: list(MDODiscipline)
        :param max_mda_iter: maximum number of iterations
        :type max_mda_iter: int
        :param name: the name of the chain
        :type name: str
        :param n_processes: maximum number of processors on which to run
        :type n_processes: int
        :param acceleration: type of acceleration to be used to extrapolate
            the residuals and save CPU time by reusing the information
            from the last iterations, either None, or m2d, or secant,
            m2d is faster but uses the 2 last iterations
        :type acceleration: str
        :param tolerance: tolerance of the iterative direct coupling solver,
            norm of the current residuals divided by initial residuals norm
            shall be lower than the tolerance to stop iterating
        :type tolerance: float
        :param linear_solver_tolerance: Tolerance of the linear solver
            in the adjoint equation
        :type linear_solver_tolerance: float
        :param use_threading: use multithreading for parallel executions
            otherwise use multiprocessing
        :type use_threading: bool
        :param warm_start: if True, the second iteration and ongoing
            start from the previous coupling solution
        :type warm_start: bool
        :param use_lu_fact: if True, when using adjoint/forward
            differenciation, store a LU factorization of the matrix
            to solve faster multiple RHS problem
        :type use_lu_fact: bool
        :param norm0: reference value of the norm of the residual to compute
            the decrease stop criteria.
            Iterations stops when norm(residual)/norm0<tolerance
        :type norm0: float
        """
        self.n_processes = n_processes
        super(MDAJacobi, self).__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            use_lu_fact=use_lu_fact,
        )
        self._initialize_grammars()
        self._set_default_inputs()
        self._compute_input_couplings()
        self.acceleration = acceleration
        self._dx_n = []
        self._g_x_n = []
        self.sizes = None
        self.parallel_execution = DiscParallelExecution(
            disciplines, n_processes, use_threading
        )

    def _initialize_grammars(self):
        """Defines all inputs and outputs of the chain"""
        for discipline in self.disciplines:
            self.input_grammar.update_from(discipline.input_grammar)
            self.output_grammar.update_from(discipline.output_grammar)

    def execute_all_disciplines(self, input_local_data):
        """Executes all self.disciplines

        :param input_local_data: the input data of the disciplines
        """
        self.reset_disciplines_statuses()

        if self.n_processes > 1:
            n_disc = len(self.disciplines)
            inputs_copy_list = [deepcopy(input_local_data) for _ in range(n_disc)]
            self.parallel_execution.execute(inputs_copy_list)
        else:
            for disc in self.disciplines:
                disc.execute(deepcopy(input_local_data))

        outputs = [discipline.get_output_data() for discipline in self.disciplines]
        for data in outputs:
            self.local_data.update(data)

    def get_expected_workflow(self):
        """
        See MDA.get_expected_workflow
        """
        sub_workflow = ExecutionSequenceFactory.serial(self.disciplines)
        if self.n_processes > 1:
            sub_workflow = ExecutionSequenceFactory.parallel(self.disciplines)
        return ExecutionSequenceFactory.loop(self, sub_workflow)

    def _run(self):
        """Run method of the chain:
        executes all disciplines in a loop until outputs converge.
        Stops when
        ||outputs-previous output||/||first outputs|| < self.tolerance

        :returns: the local data updated
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

        # store initial residual
        current_iter = 1
        self._compute_residual(
            current_couplings, new_couplings, current_iter, first=True
        )
        current_couplings = new_couplings

        while not self._termination(current_iter):
            self.execute_all_disciplines(deepcopy(self.local_data))
            new_couplings = self._current_input_couplings()

            # store current residual
            current_iter += 1
            self._compute_residual(current_couplings, new_couplings, current_iter)
            x_np1 = self._compute_nex_iterate(current_couplings, new_couplings)
            current_couplings = x_np1

    def _compute_nex_iterate(self, current_couplings, new_couplings):
        """
        Compute the next iterate given the evaluation of the couplings
        Eventually computes the secant method acceleration, see

        See :
        Iterative residual-based vector methods to accelerate
        fixed point iterations, Isabelle Ramiere, Thomas Helfer

        :param current_couplings: input couplings of the disciplines
            given for evaluation at the last iterations
        :param current_couplings: computed couplings of the disciplines
            at the last iterations
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
        new_c = DataConversion.array_to_dict(x_np1, coupl_names, self.sizes)
        self.local_data.update(new_c)
        return x_np1

    @staticmethod
    def _minimize_2md(dxn, dxn_1, dxn_2):
        """
        Compute the extrapolation coefficients of the
        2-delta method
        Minimizes the sub problem in the d-2 method
        Use a least squares solver to find he minimizer of
        dxn - x[0] * (dxn - dxn_1) - x[1] * (dxn_1 - dxn_2)

        :param dxn: delta couplings at last iteration
        :param dxn_1: delta couplings at last iteration-1
        :param dxn_2: delta couplings at last iteration-2
        :returns: lambda
        """
        mat = concatenate((atleast_2d(dxn - dxn_1), atleast_2d(dxn_1 - dxn_2)))
        return lstsq(mat.T, dxn, rcond=None)[0]

    @staticmethod
    def _compute_secant_acc(dxn, dxn_1, cgn, cgn_1):
        """
        secant acceleration

        from the paper:
        "Iterative residual-based vector methods to accelerate
        fixed point iterations",  Isabelle Ramiere, Thomas Helfer

        secant acceleration: page 15 equation (41)

        :param dxn: delta couplings at last iteration
        :param dxn_1: delta couplings at last iteration-1
        :param cgn: computed couplings at last iteration
        :param cgn_1: computed couplings at last iteration-1
        """
        d_dxn = dxn - dxn_1
        acc = (cgn - cgn_1) * dot(d_dxn, dxn) / dot(d_dxn, d_dxn)
        return cgn - acc

    def _compute_m2d_acc(self, dxn, dxn_1, dxn_2, g_n, gn_1, gn_2):
        """
        2-delta acceleration

        from the paper:
        "Iterative residual-based vector methods to accelerate
        fixed point iterations",  Isabelle Ramiere, Thomas Helfer
        page 22 eq (50)

        :param dxn: delta couplings at last iteration
        :param dxn_1: delta couplings at last iteration-1
        :param dxn_2: delta couplings at last iteration-2
        :param gn: computed couplings at last iteration
        :param gn_1: computed couplings at last iteration-1
        :param gn_2: computed couplings at last iteration-2
        """
        lamba_min = self._minimize_2md(dxn, dxn_1, dxn_2)
        acc = lamba_min[0] * (g_n - gn_1) + lamba_min[1] * (gn_1 - gn_2)
        return g_n - acc
