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
"""Base class for all Multi-disciplinary Design Analyses (MDA).."""
from __future__ import division, unicode_literals

import logging
from multiprocessing import cpu_count
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy import array, concatenate, ndarray
from numpy.linalg import norm

from gemseo.core.coupling_structure import DependencyGraph, MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequenceFactory, LoopExecSequence
from gemseo.core.jacobian_assembly import JacobianAssembly
from gemseo.utils.py23_compat import Path

LOGGER = logging.getLogger(__name__)


class MDA(MDODiscipline):
    """An MDA analysis."""

    FINITE_DIFFERENCES = "finite_differences"

    N_CPUS = cpu_count()
    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + (
        "disciplines",
        "warm_start",
        "_input_couplings",
        "reset_history_each_run",
        "norm0",
        "residual_history",
        "tolerance",
        "max_mda_iter",
        "_log_convergence",
        "lin_cache_tol_fact",
        "assembly",
        "coupling_structure",
        "max_mda_iter",
        "normed_residual",
        "strong_couplings",
        "matrix_type",
        "use_lu_fact",
        "linear_solver_tolerance",
    )

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        max_mda_iter=10,  # type: int
        name=None,  # type: Optional[str]
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        tolerance=1e-6,  # type: float
        linear_solver_tolerance=1e-12,  # type: float
        warm_start=False,  # type: bool
        use_lu_fact=False,  # type: bool
        coupling_structure=None,  # type: Optional[MDOCouplingStructure]
        log_convergence=False,  # type: bool
        linear_solver="DEFAULT",  # type: str
        linear_solver_options=None,  # type: Mapping[str,Any]
    ):  # type: (...) -> None
        """
        Args:
            disciplines: The disciplines from which to compute the MDA.
            max_mda_iter: The maximum iterations number for the MDA algorithm.
            name: The name to be given to the MDA.
                If None, use the name of the class.
            grammar_type: The type of the input and output grammars,
                either :attr:`JSON_GRAMMAR_TYPE` or :attr:`SIMPLE_GRAMMAR_TYPE`.
            tolerance: The tolerance of the iterative direct coupling solver;
                the norm of the current residuals divided by initial residuals norm
                shall be lower than the tolerance to stop iterating.
            linear_solver_tolerance: The tolerance of the linear solver
                in the adjoint equation.
            warm_start: Whether the second iteration and ongoing start
                from the previous coupling solution.
            use_lu_fact: Whether to store a LU factorization of the matrix
                when using adjoint/forward differentiation.
                to solve faster multiple RHS problem.
            coupling_structure: The coupling structure to be used by the MDA.
                If None, it is created from `disciplines`.
            log_convergence: Whether to log the MDA convergence,
                expressed in terms of normed residuals.
            linear_solver: The name of the linear solver.
            linear_solver_options: The options passed to the linear solver factory.
        """
        super(MDA, self).__init__(name, grammar_type=grammar_type)
        self.tolerance = tolerance
        self.linear_solver = linear_solver
        self.linear_solver_tolerance = linear_solver_tolerance
        self.linear_solver_options = linear_solver_options or {}
        self.max_mda_iter = max_mda_iter
        self.disciplines = disciplines
        if coupling_structure is None:
            self.coupling_structure = MDOCouplingStructure(disciplines)
        else:
            self.coupling_structure = coupling_structure
        self.assembly = JacobianAssembly(self.coupling_structure)
        self.residual_history = []
        self.reset_history_each_run = False
        self.warm_start = warm_start

        # Don't erase coupling values before calling _compute_jacobian

        self._linearize_on_last_state = True
        self.norm0 = None
        self.normed_residual = 1.0
        self.strong_couplings = self.coupling_structure.strong_couplings()
        self.all_couplings = self.coupling_structure.get_all_couplings()
        self._input_couplings = []
        self.matrix_type = JacobianAssembly.SPARSE
        self.use_lu_fact = use_lu_fact
        # By default dont use an approximate cache for linearization
        self.lin_cache_tol_fact = 0.0

        self._initialize_grammars()
        self._check_consistency()
        self.__check_linear_solver_options()
        self._check_couplings_types()
        self._log_convergence = log_convergence

    def _initialize_grammars(self):  # type: (...) -> None
        """Define all the inputs and outputs of the MDA.

        Add all the outputs of all the disciplines to the outputs.
        """
        for discipline in self.disciplines:
            self.input_grammar.update_from(discipline.input_grammar)
            self.output_grammar.update_from(discipline.output_grammar)

    @property
    def log_convergence(self):  # type: (...) -> bool
        """Whether to log the MDA convergence."""
        return self._log_convergence

    @log_convergence.setter
    def log_convergence(
        self,
        value,  # type: bool
    ):  # type: (...) -> None
        self._log_convergence = value

    def __check_linear_solver_options(self):  # type: (...) -> None
        """Check the linear solver options.

        The linear solver tolerance cannot be set
        using the linear solver option dictionary,
        as it is set using the linear_solver_tolerance keyword argument.

        Raises:
            ValueError: If the 'tol' keyword is in linear_solver_options.
        """
        if "tol" in self.linear_solver_options:
            msg = (
                "The linear solver tolerance shall be set"
                " using the linear_solver_tolerance argument."
            )
            raise ValueError(msg)

    def _check_consistency(self):  # type: (...) -> None
        """Check if there are not more than one equation per variable.

        For instance if a strong coupling is not also a self coupling.

        Raises:
            ValueError:
                * If there are too many coupling constraints.
                * If outputs are defined multiple times.
        """
        strong_c_disc = self.coupling_structure.strongly_coupled_disciplines(
            add_self_coupled=False
        )
        also_strong = [
            disc
            for disc in strong_c_disc
            if self.coupling_structure.is_self_coupled(disc)
        ]
        if also_strong:
            for disc in also_strong:
                in_outs = set(disc.get_input_data_names()) & set(
                    disc.get_output_data_names()
                )
                LOGGER.warning(
                    "Self coupling variables in discipline %s are: %s.",
                    disc.name,
                    in_outs,
                )

            also_strong_n = [disc.name for disc in also_strong]
            raise ValueError(
                "Too many coupling constraints; "
                "the following disciplines are self coupled "
                "and also strongly coupled with other disciplines: {}.".format(
                    also_strong_n
                )
            )

        all_outs = {}
        multiple_outs = []
        for disc in self.disciplines:
            for out in disc.get_output_data_names():
                if out in all_outs:
                    multiple_outs.append(out)
                all_outs[out] = disc

        if multiple_outs:
            raise ValueError(
                "Outputs are defined multiple times: {}.".format(multiple_outs)
            )

    def _compute_input_couplings(self):  # type: (...) -> None
        """Compute the strong couplings that are inputs of the MDA."""
        input_couplings = set(self.strong_couplings) & set(self.get_input_data_names())
        self._input_couplings = list(input_couplings)

    def _current_input_couplings(self):  # type: (...) -> ndarray
        """Return the current values of the input coupling variables."""
        input_couplings = list(iter(self.get_outputs_by_name(self._input_couplings)))
        if not input_couplings:
            return array([])
        return concatenate(input_couplings)

    def _current_strong_couplings(self):  # type: (...) -> ndarray
        """Return the current values of the strong coupling variables."""
        couplings = list(iter(self.get_outputs_by_name(self.strong_couplings)))
        if not couplings:
            return array([])
        return concatenate(couplings)

    def _retrieve_diff_inouts(
        self,
        force_all=False,  # type: bool
    ):  # type: (...) -> Tuple[Union[Set[str],List[str]],Union[Set[str],List[str]]]
        """Return the names of the inputs and outputs involved in the differentiation.

        Args:
            force_all: Whether to differentiate all outputs with respect to all inputs.
                If `False`,
                differentiate the :attr:`_differentiated_outputs`
                with respect to the :attr:`_differentiated_inputs`.

        Returns:
            The inputs according to which to differentiate,
            the outputs to be differentiated.
        """
        if force_all:
            strong_cpl = set(self.strong_couplings)
            inputs = set(self.get_input_data_names())
            outputs = self.get_output_data_names()
            # Don't linearize wrt
            inputs = inputs - (strong_cpl & inputs)
            # Don't do this with output couplings because
            # their derivatives wrt design variables may be needed
            # outputs = outputs - (strong_cpl & outputs)

            return inputs, outputs

        return MDODiscipline._retrieve_diff_inouts(self, False)

    def _couplings_warm_start(self):  # type: (...) -> None
        """Load the previous couplings values to local data."""
        cached_outputs = self.cache.get_last_cached_outputs()
        if not cached_outputs:
            return
        for input_name in self._input_couplings:
            input_value = cached_outputs.get(input_name)
            if input_value is not None:
                self.local_data[input_name] = input_value

    def _set_default_inputs(self):  # type: (...) -> None
        """Set the default input values of the MDA from the disciplines ones."""
        self.default_inputs = {}
        mda_input_names = self.get_input_data_names()
        for discipline in self.disciplines:
            for input_name in discipline.default_inputs:
                if input_name in mda_input_names:
                    self.default_inputs[input_name] = discipline.default_inputs[
                        input_name
                    ]

    def _check_couplings_types(self):  # type: (...) -> None
        """Check that the coupling variables are of type array in the grammars.

        Raises:
            ValueError: When at least one of the coupling variables is not an array.
        """
        not_arrays = []
        for discipline in self.disciplines:
            for grammar in (discipline.input_grammar, discipline.output_grammar):
                for coupling in self.all_couplings:
                    exists = grammar.is_data_name_existing(coupling)
                    if exists and not grammar.is_type_array(coupling):
                        not_arrays.append(coupling)

        not_arrays = sorted(set(not_arrays))
        if not_arrays:
            raise ValueError(
                "The coupling variables {} must be of type array.".format(not_arrays)
            )

    def reset_disciplines_statuses(self):  # type: (...) -> None
        """Reset all the statuses of the disciplines."""
        for discipline in self.disciplines:
            discipline.reset_statuses_for_run()

    def reset_statuses_for_run(self):  # type: (...) -> None
        MDODiscipline.reset_statuses_for_run(self)
        self.reset_disciplines_statuses()

    def get_expected_workflow(self):  # type: (...) ->LoopExecSequence
        disc_exec_seq = ExecutionSequenceFactory.serial(self.disciplines)
        return ExecutionSequenceFactory.loop(self, disc_exec_seq)

    def get_expected_dataflow(
        self,
    ):  # type: (...) -> List[Tuple[MDODiscipline,MDODiscipline,List[str]]]
        all_disc = [self]
        all_disc.extend(self.disciplines)
        graph = DependencyGraph(all_disc)
        res = graph.get_disciplines_couplings()
        return res

    def _compute_jacobian(
        self,
        inputs=None,  # type: Optional[Iterable[str]]
        outputs=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> None
        # Do not re execute disciplines if inputs error is beyond self tol
        # Apply a safety factor on this (mda is a loop, inputs
        # of first discipline
        # have changed at convergence, therefore the cache is not exactly
        # the same as the current value
        exec_cache_tol = self.lin_cache_tol_fact * self.tolerance
        force_no_exec = exec_cache_tol != 0.0
        self.__check_linear_solver_options()
        self.jac = self.assembly.total_derivatives(
            self.local_data,
            outputs,
            inputs,
            self.all_couplings,
            tol=self.linear_solver_tolerance,
            mode=self.linearization_mode,
            matrix_type=self.matrix_type,
            use_lu_fact=self.use_lu_fact,
            exec_cache_tol=exec_cache_tol,
            force_no_exec=force_no_exec,
            linear_solver=self.linear_solver,
            **self.linear_solver_options
        )

    # fixed point methods
    def _compute_residual(
        self,
        current_couplings,  # type: ndarray
        new_couplings,  # type: ndarray
        current_iter,  # type: int
        first=False,  # type: bool
        store_it=True,  # type: bool
        log_normed_residual=False,  # type: bool
    ):  # type: (...) -> ndarray
        """Compute the residual on the inputs of the MDA.

        Args:
            current_couplings: The values of the couplings before the execution.
            new_couplings: The values of the couplings after the execution.
            current_iter: The current iteration of the fixed-point method.
            first: Whether it is the first residual of the fixed-point method.
            store_it: Whether to store the normed residual.
            log_normed_residual: Whether to log the normed residual.

        Returns:
            The normed residual.
        """
        if first and self.reset_history_each_run:
            self.residual_history = []

        normed_residual = norm((current_couplings - new_couplings).real)
        if self.norm0 is None:
            self.norm0 = normed_residual
        if self.norm0 == 0:
            self.norm0 = 1
        self.normed_residual = normed_residual / self.norm0
        if log_normed_residual:
            LOGGER.info(
                "%s running... Normed residual = %s (iter. %s)",
                self.name,
                "{:.2e}".format(self.normed_residual),
                current_iter,
            )

        if store_it:
            self.residual_history.append((self.normed_residual, current_iter))
        return self.normed_residual

    def check_jacobian(
        self,
        input_data=None,  # type: Optional[Mapping[str,ndarray]]
        derr_approx=FINITE_DIFFERENCES,  # type: str
        step=1e-7,  # type: float
        threshold=1e-8,  # type: float
        linearization_mode="auto",  # type: str
        inputs=None,  # type: Optional[Iterable[str]]
        outputs=None,  # type: Optional[Iterable[str]]
        parallel=False,  # type: bool
        n_processes=N_CPUS,  # type: int
        use_threading=False,  # type: bool
        wait_time_between_fork=0,  # type: int
        auto_set_step=False,  # type: bool
        plot_result=False,  # type: bool
        file_path="jacobian_errors.pdf",  # type: Union[str,Path]
        show=False,  # type: bool
        figsize_x=10,  # type: float
        figsize_y=10,  # type: float
        reference_jacobian_path=None,
        save_reference_jacobian=False,
        indices=None,
    ):  # type: (...) -> bool
        """Check if the analytical Jacobian is correct with respect to a reference one.

        If `reference_jacobian_path` is not `None`
        and `save_reference_jacobian` is `True`,
        compute the reference Jacobian with the approximation method
        and save it in `reference_jacobian_path`.

        If `reference_jacobian_path` is not `None`
        and `save_reference_jacobian` is `False`,
        do not compute the reference Jacobian
        but read it from `reference_jacobian_path`.

        If `reference_jacobian_path` is `None`,
        compute the reference Jacobian without saving it.

        Args:
            input_data: The input values.
                If None, use the default input values.
            derr_approx: The derivative approximation method.
            threshold: The acceptance threshold for the Jacobian error.
            linearization_mode: The mode of linearization,
                either "direct", "adjoint" or "auto" switch
                depending on dimensions of inputs and outputs.
            inputs: The names of the inputs with respect to which to differentiate.
                If None, use the inputs of the MDA.
            outputs: The outputs to differentiate.
                If None, use all the outputs of the MDA.
            step: The step
                for finite differences or complex step differentiation methods.
            parallel: Whether to execute the MDA in parallel.
            n_processes: The maximum number of processors on which to run.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory.
                This is important to note
                if you want to execute the same discipline multiple times,
                you shall use multiprocessing.
            wait_time_between_fork: The time waited between two forks
                of the process / thread.
            auto_set_step: Whether to compute the optimal step
                for a forward first order finite differences gradient approximation.
            plot_result: Whether to plot the result of the validation
                comparing the exact and approximated Jacobians.
            file_path: The path to the output file if `plot_result` is `True`.
            show: Whether to open the figure.
            figsize_x: The *x* size of the figure in inches.
            figsize_y: The *y* size of the figure in inches.
            reference_jacobian_path: The path of the reference Jacobian file.
            save_reference_jacobian: Whether to save the reference Jacobian.
            indices: The indices of the inputs and outputs
                for the different sub-Jacobian matrices,
                formatted as ``{variable_name: variable_components}``
                where ``variable_components`` can be either
                an integer, e.g. `2`
                a sequence of integers, e.g. `[0, 3]`,
                a slice, e.g. `slice(0,3)`,
                the ellipsis symbol (`...`)
                or `None`, which is the same as ellipsis.
                If a variable name is missing, consider all its components.
                If None, consider all the components of all the ``inputs`` and ``outputs``.

        Return:
            Whether the passed Jacobian is correct.
        """
        # Strong couplings are not linearized
        if inputs is None:
            inputs = self.get_input_data_names()

        inputs = list(iter(inputs))
        for str_cpl in self.all_couplings:
            if str_cpl in inputs:
                inputs.remove(str_cpl)

        if outputs is None:
            outputs = self.get_output_data_names()

        outputs = list(iter(outputs))
        for str_cpl in self.all_couplings:
            if str_cpl in outputs:
                outputs.remove(str_cpl)

        return super(MDA, self).check_jacobian(
            input_data=input_data,
            derr_approx=derr_approx,
            step=step,
            threshold=threshold,
            linearization_mode=linearization_mode,
            inputs=inputs,
            outputs=outputs,
            parallel=parallel,
            n_processes=n_processes,
            use_threading=use_threading,
            wait_time_between_fork=wait_time_between_fork,
            auto_set_step=auto_set_step,
            plot_result=plot_result,
            file_path=file_path,
            show=show,
            figsize_x=figsize_x,
            figsize_y=figsize_y,
            reference_jacobian_path=reference_jacobian_path,
            save_reference_jacobian=save_reference_jacobian,
            indices=indices,
        )

    def _warn_convergence_criteria(
        self,
        current_iter,  # type: int
    ):  # type: (...) -> Tuple[bool,bool]
        """Log a warning if max_iter is reached and if max residuals is above tolerance.

        Args:
            current_iter: The current iteration of the MDA.

        Returns:
            * Whether the normed residual is lower than the tolerance.
            * Whether the maximum number of iterations is reached.
        """

        residual_is_small = self.normed_residual <= self.tolerance
        max_iter_is_reached = self.max_mda_iter <= current_iter
        if max_iter_is_reached and not residual_is_small:
            msg = (
                "%s has reached its maximum number of iterations "
                "but the normed residual %s is still above the tolerance %s."
            )
            LOGGER.warning(msg, self.name, self.normed_residual, self.tolerance)
        return residual_is_small, max_iter_is_reached

    def _termination(
        self,
        current_iter,  # type: int
    ):  # type: (...) -> bool
        """Termination criterion.

        Args:
            current_iter: The current iteration of the fixed point method.

        Returns:
            Whether to stop the MDA algorithm.
        """
        residual_is_small, max_iter_is_reached = self._warn_convergence_criteria(
            current_iter
        )
        return residual_is_small or max_iter_is_reached

    def _set_cache_tol(
        self,
        cache_tol,  # type: float
    ):  # type: (...) -> None
        """Set to the cache input tolerance.

        To be overloaded by subclasses.

        Args:
            cache_tol: The cache tolerance.
        """
        super(MDA, self)._set_cache_tol(cache_tol)
        for disc in self.disciplines:
            disc.cache_tol = cache_tol or 0.0

    def plot_residual_history(
        self,
        show=False,  # type: bool
        save=True,  # type: bool
        n_iterations=None,  # type: Optional[int]
        logscale=None,  # type: Optional[Tuple[int,int]]
        filename=None,  # type: Optional[str]
        figsize=(50, 10),  # type: Tuple[int,int]
    ):  # type: (...) -> None
        """Generate a plot of the residual history.

        All residuals are stored in the history;
        only the final residual of the converged MDA is plotted
        at each optimization iteration.

        Args:
            show: Whether to display the plot on screen.
            save: Whether to save the plot as a PDF file.
            n_iterations: The number of iterations on the *x* axis.
                If None, use all the iterations.
            logscale: The limits of the *y* axis.
                If None, do not change the limits of the *y* axis.
            filename: The name of the file to save the figure.
                If None, use "{mda.name}_residual_history.pdf".
            figsize: The *x* and *y* sizes of the figure in inches.
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
        fig_ax.set_title("{}: residual plot".format(self.name))

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
                filename = "{}_residual_history.pdf".format(self.name)
            plt.savefig(filename, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
