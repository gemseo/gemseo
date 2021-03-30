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
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Base class for all Multi-disciplinary Design Analysis
*****************************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import range, super
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
from future import standard_library
from matplotlib.ticker import MaxNLocator
from numpy import array, concatenate
from numpy.linalg import norm

from gemseo.core.coupling_structure import DependencyGraph, MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.jacobian_assembly import JacobianAssembly

standard_library.install_aliases()


class MDA(MDODiscipline):
    """Perform an MDA analysis.

    Base class.
    """

    FINITE_DIFFERENCES = "finite_differences"

    N_CPUS = cpu_count()

    def __init__(
        self,
        disciplines,
        max_mda_iter=10,
        name=None,
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,
        tolerance=1e-6,
        linear_solver_tolerance=1e-12,
        warm_start=False,
        use_lu_fact=False,
        norm0=None,
    ):
        """Constructor.

        :param disciplines: the disciplines list
        :param max_mda_iter: maximum iterations number for MDA
        :param tolerance: tolerance of the iterative direct coupling solver,
            norm of the current residuals divided by initial residuals norm
            shall be lower than the tolerance to stop iterating
        :param name: the name of the chain
        :param grammar_type: the type of grammar to use for IO declaration
            either JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE
        :param warm_start: if True, the second iteration and ongoing
            start from the previous coupling solution
        :param linear_solver_tolerance: Tolerance of the linear solver
            in the adjoint equation
        :param use_lu_fact: if True, when using adjoint/forward
            differenciation, store a LU factorization of the matrix
            to solve faster multiple RHS problem
        :param norm0: reference value of the norm of the residual to compute
            the decrease stop criteria.
            Iterations stops when norm(residual)/norm0<tolerance
        :type norm0: float
        """
        super(MDA, self).__init__(name, grammar_type=grammar_type)
        self.tolerance = tolerance
        self.linear_solver_tolerance = linear_solver_tolerance
        self.max_mda_iter = max_mda_iter
        self.disciplines = disciplines
        self.coupling_structure = MDOCouplingStructure(disciplines)
        self.assembly = JacobianAssembly(self.coupling_structure)
        self.residual_history = []
        self.reset_history_each_run = False
        self.warm_start = warm_start
        # Don't erase coupling values before calling _compute_jacobian
        self._linearize_on_last_state = True
        self.norm0 = norm0
        self.normed_residual = 1.0
        self.strong_couplings = self.coupling_structure.strong_couplings()
        self._input_couplings = []
        self.matrix_type = JacobianAssembly.SPARSE
        self.use_lu_fact = use_lu_fact
        # By default dont use an approximate cache for linearization
        self.lin_cache_tol_fact = 0.0

    def _run(self):
        """Run the MDA."""
        raise NotImplementedError()

    def _compute_input_couplings(self):
        """Compute the coupling variables that are inputs of the MDA."""
        inputs = self.get_input_data_names()
        input_couplings = set(self.strong_couplings) & set(inputs)
        self._input_couplings = list(input_couplings)

    def _current_input_couplings(self):
        """Compute the vector of the current input coupling values."""
        input_couplings = list(iter(self.get_outputs_by_name(self._input_couplings)))
        if not input_couplings:
            return array([])
        return concatenate(input_couplings)

    def _retreive_diff_inouts(self, force_all=False):
        """Get the list of outputs to be differentiated w.r.t. inputs.

        This method get the list of the outputs to be differentiated w.r.t. the
        inputs depending on the self._differentiated_inputs and
        self._differentiated_inputs attributes, and the force_all option
        """
        if force_all:
            strong_cpl = set(self.strong_couplings)
            inputs = set(self.get_input_data_names())
            outputs = self.get_output_data_names()
            # Dont linearize wrt
            inputs = inputs - (strong_cpl & inputs)
            # Don't do this with output couplings because
            # their derivatives wrt design variables may be needed
            # outputs = outputs - (strong_cpl & outputs)

            return inputs, outputs
        return MDODiscipline._retreive_diff_inouts(self, False)

    def _couplings_warm_start(self):
        """Load the previous couplings values to local data."""
        cached_outputs = self.cache.get_last_cached_outputs()
        if not cached_outputs:
            return
        for input_name in self._input_couplings:
            input_value = cached_outputs.get(input_name)
            if input_value is not None:
                self.local_data[input_name] = input_value

    def _set_default_inputs(self):
        """Compute the default default_inputs.

        This method computes the default default_inputs from the disciplines
        default default_inputs.
        """
        self.default_inputs = {}
        mda_input_names = self.get_input_data_names()
        for discipline in self.disciplines:
            for input_name in discipline.default_inputs:
                if input_name in mda_input_names:
                    self.default_inputs[input_name] = discipline.default_inputs[
                        input_name
                    ]

    def reset_disciplines_statuses(self):
        """Reset all the statuses of sub disciplines for run."""
        for discipline in self.disciplines:
            discipline.reset_statuses_for_run()

    def reset_statuses_for_run(self):
        """Reset the statuses."""
        MDODiscipline.reset_statuses_for_run(self)
        self.reset_disciplines_statuses()

    def get_expected_workflow(self):
        """Return the expected execution sequence.

        This method is used for xdsm representation
        See MDOFormulation.get_expected_workflow
        """
        disc_exec_seq = ExecutionSequenceFactory.serial(self.disciplines)
        return ExecutionSequenceFactory.loop(self, disc_exec_seq)

    def get_expected_dataflow(self):
        """Return the expected data exchange sequence.

        This method is used for xdsm representation
        See MDOFormulation.get_expected_dataflow
        """
        all_disc = [self] + self.disciplines
        graph = DependencyGraph(all_disc)
        res = graph.get_disciplines_couplings()
        return res

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Actual computation of the jacobians.

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization
            should be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be
            performed on all outputs (Default value = None)
        :param kwargs_lin: optional parameters for scipy sparse linear solver
        """
        # Do not re execute disciplines if inputs error is beyond self tol
        # Apply a safety factor on this (mda is a loop, inputs
        # of first discipline
        # have changed at convergence, therefore the cache is not exactly
        # the same as the current value
        exec_cache_tol = self.lin_cache_tol_fact * self.tolerance
        force_no_exec = exec_cache_tol != 0.0
        self.jac = self.assembly.total_derivatives(
            self.local_data,
            outputs,
            inputs,
            self.strong_couplings,
            tol=self.linear_solver_tolerance,
            mode=self.linearization_mode,
            matrix_type=self.matrix_type,
            use_lu_fact=self.use_lu_fact,
            exec_cache_tol=exec_cache_tol,
            force_no_exec=force_no_exec,
        )

    # fixed point methods
    def _compute_residual(
        self, current_couplings, new_couplings, current_iter, first=False, store_it=True
    ):
        """Compute the residual on the inputs of the MDA.

        :param current_couplings: the values of the couplings before
            the execution
        :param new_couplings: the values of the couplings after
            the execution
        :param current_iter: the current iteration of the fixed point
        :param first: if True, first residual of the fixed point
            (Default value = False)
        """
        if first and self.reset_history_each_run:
            self.residual_history = []

        normed_residual = norm((current_couplings - new_couplings).real)
        if self.norm0 is None:
            self.norm0 = normed_residual
        self.normed_residual = normed_residual / self.norm0

        if store_it:
            self.residual_history.append((self.normed_residual, current_iter))
        return self.normed_residual

    def check_jacobian(
        self,
        input_data=None,
        derr_approx=FINITE_DIFFERENCES,
        step=1e-7,
        threshold=1e-8,
        linearization_mode="auto",
        inputs=None,
        outputs=None,
        parallel=False,
        n_processes=N_CPUS,
        use_threading=False,
        wait_time_between_fork=0,
    ):
        """Check if the jacobian is correct.

        This method checks the jacobian computed by the linearize() method.

        :param input_data: input data dict (Default value = None)
        :param derr_approx: derivative approximation method: COMPLEX_STEP
            (Default value = COMPLEX_STEP)
        :param threshold: acceptance threshold for the jacobian error
            (Default value = 1e-8)
        :param linearization_mode: the mode of linearization: direct, adjoint
            or automated switch depending on dimensions
            of inputs and outputs (Default value = 'auto')
        :param inputs: list of inputs wrt which to differentiate
            (Default value = None)
        :param outputs: list of outputs to differentiate (Default value = None)
        :param step: the step for finite differences or complex step
        :param parallel: if True, executes in parallel
        :param n_processes: maximum number of processors on which to run
        :param use_threading: if True, use Threads instead of processes
            to parallelize the execution
            multiprocessing will copy (serialize) all the disciplines,
            while threading will share all the memory
            This is important to note if you want to execute the same
            discipline multiple times, you shall use multiprocessing
        :param wait_time_between_fork: time waited between two forks of the
            process /Thread
        :returns: True if the check is accepted, False otherwise
        """
        # Strong couplings are not linearized
        if inputs is None:
            inputs = self.get_input_data_names()
        inputs = list(iter(inputs))
        for str_cpl in self.strong_couplings:
            if str_cpl in inputs:
                inputs.remove(str_cpl)
        if outputs is None:
            outputs = self.get_output_data_names()
        outputs = list(iter(outputs))
        for str_cpl in self.strong_couplings:
            if str_cpl in outputs:
                outputs.remove(str_cpl)

        return super(MDA, self).check_jacobian(
            input_data,
            derr_approx,
            step,
            threshold,
            linearization_mode,
            inputs,
            outputs,
        )

    def _termination(self, current_iter):
        """Termination criterion.

        :param current_iter: current iteration of the fixed point method
        """
        stop = self.normed_residual <= self.tolerance
        stop = stop or self.max_mda_iter <= current_iter
        return stop

    def _set_cache_tol(self, cache_tol):
        """Set to the cache input tolerance.

        To be overloaded by subclasses

        :param cache_tol: float, cache tolerance
        """
        super(MDA, self)._set_cache_tol(cache_tol)
        for disc in self.disciplines:
            disc.cache_tol = cache_tol or 0.0

    def plot_residual_history(
        self,
        show=False,
        save=True,
        n_iterations=None,
        logscale=None,
        filename=None,
        figsize=(50, 10),
    ):
        """Generate a plot of the residual history.

        All residuals are stored in the history ; only the final
        residual of the converged MDA is plotted at each optimization
        iteration

        :param show: if True, displays the plot on screen
            (Default value = False)
        :param save: if True, saves the plot as a PDF file
            (Default value = True)
        :param n_iterations: if not None, fix the number of iterations in
            the x axis (Default value = None)
        :param logscale: if not None, fix the logscale in the y axis
            (Default value = None)
        :param filename: Default value = None)
        """
        fig = plt.figure(figsize=figsize)
        fig_ax = fig.add_subplot(1, 1, 1)

        # split list of couples
        residual = [res for (res, _) in self.residual_history]
        # red dot for first iteration
        colors = [
            "red" if current_iter == 1 else "black"
            for (_, current_iter) in self.residual_history
        ]

        fig_ax.scatter(
            list(range(len(residual))), residual, s=20, color=colors, zorder=2
        )
        fig_ax.plot(residual, linestyle="-", c="k", zorder=1)
        fig_ax.axhline(y=self.tolerance, c="blue", linewidth=0.5, zorder=0)
        fig_ax.set_title(self.name + ": residual plot")

        if n_iterations is None:
            n_iterations = len(self.residual_history)

        plt.yscale("log")
        plt.xlabel(r"iterations", fontsize=14)
        plt.xlim([-1, n_iterations])
        fig_ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        plt.ylabel(r"$\log(||residuals||/||y_0||)$", fontsize=14)
        if logscale is not None:
            plt.ylim(logscale)

        if save:
            if filename is None:
                filename = self.name + "_residual_history.pdf"
            plt.savefig(filename, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
