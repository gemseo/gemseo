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
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
An advanced MDA splitting algorithm based on graphs
***************************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from multiprocessing import cpu_count
from os.path import join, split

from future import standard_library

from gemseo.api import create_mda
from gemseo.core.chain import MDOChain
from gemseo.core.execution_sequence import SerialExecSequence
from gemseo.mda.mda import MDA

standard_library.install_aliases()


N_CPUS = cpu_count()


class MDAChain(MDA):
    """The **MDAChain** computes a chain of subMDAs and simple evaluations.

    The execution sequence is provided by the
    :class:`.DependencyGraph` class.
    """

    def __init__(
        self,
        disciplines,
        sub_mda_class="MDAJacobi",
        max_mda_iter=20,
        name=None,
        n_processes=N_CPUS,
        chain_linearize=False,
        tolerance=1e-6,
        use_lu_fact=False,
        norm0=None,
        **sub_mda_options
    ):
        """
        Constructor

        :param disciplines: the disciplines list
        :type disciplines: list(MDODiscipline)
        :param sub_mda_class: the class to instantiate for sub MDAs
        :type sub_mda_class: str
        :param max_mda_iter: maximum number of iterations for sub MDAs
        :type max_mda_iter: int
        :param name: name of self
        :type name: str
        :param n_processes: number of processes for parallel run
        :type n_processes: int
        :param chain_linearize: linearize the chain of execution, if True
            Otherwise, linearize the oveall MDA with base class method
            Last option is preferred to minimize computations in adjoint mode
            in direct mode, chain_linearize may be cheaper
        :type chain_linearize: bool
        :param tolerance: tolerance of the iterative direct coupling solver,
            norm of the current residuals divided by initial residuals norm
            shall be lower than the tolerance to stop iterating
        :type tolerance: float
        :param use_lu_fact: if True, when using adjoint/forward
            differenciation, store a LU factorization of the matrix
            to solve faster multiple RHS problem
        :type use_lu_fact: bool
        :param norm0: reference value of the norm of the residual to compute
            the decrease stop criteria.
            Iterations stops when norm(residual)/norm0<tolerance
        :type norm0: float
        :param sub_mda_options: options dict passed to the sub mda
        :type sub_mda_options: dict
        """
        self.n_processes = n_processes
        self.mdo_chain = None
        self.__chain_linearize = chain_linearize
        self.sub_mda_list = []

        # compute execution sequence of the disciplines
        super(MDAChain, self).__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            tolerance=tolerance,
            use_lu_fact=use_lu_fact,
        )
        sequence = self.coupling_structure.graph.execution_sequence
        self.execution_sequence = sequence
        self._create_mdo_chain(
            disciplines, sub_mda_class=sub_mda_class, **sub_mda_options
        )
        self._initialize_grammars()
        self._set_default_inputs()
        self._compute_input_couplings()
        # cascade the tolerance
        for sub_mda in self.sub_mda_list:
            sub_mda.tolerance = self.tolerance

    def _create_mdo_chain(
        self, disciplines, sub_mda_class="MDAJacobi", **sub_mda_options
    ):
        """Create an MDO chain from the execution sequence of the disciplines.

        :param sub_mda_class: class of sub-MDAs in
            {Jacobi, GS, Newton, Sequential}
        :param disciplines: list of disciplines
        :param sub_mda_options: options passed to the MDA at construction
        """
        chained_disciplines = []
        self.sub_mda_list = []
        for parallel_tasks in self.execution_sequence:
            # to parallelize, check if 1 < len(parallel_tasks)
            # for now, parallel tasks are run sequentially
            for coupled_disciplines in parallel_tasks:
                # several disciplines coupled
                if len(coupled_disciplines) > 1:
                    # order the MDA disciplines the same way as the
                    # original disciplines
                    sub_mda_disciplines = []
                    for (i, disc_i) in enumerate(disciplines):
                        if i in coupled_disciplines:
                            sub_mda_disciplines.append(disc_i)

                    # create a sub-MDA
                    sub_mda = create_mda(
                        sub_mda_class,
                        sub_mda_disciplines,
                        max_mda_iter=self.max_mda_iter,
                        tolerance=self.tolerance,
                        **sub_mda_options
                    )
                    sub_mda.n_processes = self.n_processes

                    chained_disciplines.append(sub_mda)
                    self.sub_mda_list.append(sub_mda)
                # single discipline
                else:
                    disc_index = coupled_disciplines[0]
                    single_discipline = disciplines[disc_index]
                    chained_disciplines.append(single_discipline)
        # create the MDO chain that sequentially evaluates the sub-MDAs and the
        # single disciplines
        self.mdo_chain = MDOChain(chained_disciplines, name="MDA chain")

    def _initialize_grammars(self):
        """Define all inputs and outputs of the chain."""
        self.input_grammar.update_from(self.mdo_chain.input_grammar)
        self.output_grammar.update_from(self.mdo_chain.output_grammar)

    def _run(self):
        """Execute the chained MDA."""
        if self.warm_start:
            self._couplings_warm_start()
        self.local_data = self.mdo_chain.execute(self.local_data)
        return self.local_data

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Actual computation of the jacobians.

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization
            should be performed wrt all inputs
            (Default value = None)
        :param outputs: linearization should be performed on
            outputs list.
            If None, linearization should be
            performed on all chain_outputs (Default value = None)
        """
        if self.__chain_linearize:
            self.mdo_chain.add_differentiated_inputs(inputs)
            self.mdo_chain.add_differentiated_outputs(outputs)
            # the Jacobian of the MDA chain is the Jacobian of the MDO chain
            last_cached = self.cache.get_last_cached_inputs()
            self.mdo_chain.linearize(last_cached)
            self.jac = self.mdo_chain.jac
        else:
            super(MDAChain, self)._compute_jacobian(inputs, outputs)

    def add_differentiated_inputs(self, inputs=None):
        """Add inputs to the differentiation list.

        Updates self._differentiated_inputs with inputs

        :param inputs: list of inputs variables to differentiate
            if None, all inputs of discipline are used (Default value = None)

        """
        MDA.add_differentiated_inputs(self, inputs)
        if self.__chain_linearize:
            self.mdo_chain.add_differentiated_inputs(inputs)

    def add_differentiated_outputs(self, outputs=None):
        """Add outputs to the differentiation list.

        Updates self._differentiated_inputs with inputs

        :param outputs: list of output variables to differentiate
            if None, all outputs of discipline are used
        """
        MDA.add_differentiated_outputs(self, outputs=outputs)
        if self.__chain_linearize:
            self.mdo_chain.add_differentiated_outputs(outputs)

    def get_expected_dataflow(self):
        """Get the expected dataflow.

        See MDOChain.get_expected_dataflow
        """
        return self.mdo_chain.get_expected_dataflow()

    def get_expected_workflow(self):
        """Get the expected workflow.

        See MDOChain.get_expected_workflow
        """
        exec_s = SerialExecSequence(self)
        workflow = self.mdo_chain.get_expected_workflow()
        exec_s.extend(workflow)
        return exec_s

    def reset_statuses_for_run(self):
        """Set all the statuses to PENDING."""
        super(MDAChain, self).reset_statuses_for_run()
        self.mdo_chain.reset_statuses_for_run()

    def plot_residual_history(
        self,
        show=False,
        save=True,
        n_iterations=None,
        logscale=None,
        filename=None,
        figsize=(50, 10),
    ):
        """Generate a plot of the residual history
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
        for sub_mda in self.sub_mda_list:
            if filename is not None:
                s_filename = split(filename)
                filename = join(
                    s_filename[0], sub_mda.__class__.__name__ + "_" + s_filename[1]
                )
            sub_mda.plot_residual_history(
                show, save, n_iterations, logscale, filename, figsize
            )
