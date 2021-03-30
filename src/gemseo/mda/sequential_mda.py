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
A chain of MDAs to build hybrids of MDA algorithms sequentially
***************************************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import super

from future import standard_library

from gemseo.core.discipline import MDODiscipline
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.mda import MDA
from gemseo.mda.newton import MDANewtonRaphson

standard_library.install_aliases()


class MDASequential(MDA):
    """
    Perform a MDA defined as a sequence of elementary MDAs.
    """

    def __init__(
        self,
        disciplines,
        mda_sequence,
        name=None,
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,
        max_mda_iter=10,
        tolerance=1e-6,
        linear_solver_tolerance=1e-12,
        warm_start=False,
        use_lu_fact=False,
        norm0=None,
    ):
        """
        Constructor

        :param disciplines: the disciplines list
        :type disciplines: list(MDODiscipline)
        :param mda_sequence: sequence of MDAs
        :type mda_sequence: list(MDA)
        :param max_mda_iter: maximum number of iterations
        :type max_mda_iter: int
        :param name: name
        :type name: str
        :param grammar_type: the type of grammar to use for IO declaration
            either JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE
        :type grammar_type: str
        :param tolerance: tolerance of the iterative direct coupling solver,
            norm of the current residuals divided by initial residuals norm
            shall be lower than the tolerance to stop iterating
        :type tolerance: float
        :param warm_start: if True, the second iteration and ongoing
            start from the previous coupling solution
        :type warm_start: bool
        :param linear_solver_tolerance: Tolerance of the linear solver
            in the adjoint equation
        :type linear_solver_tolerance: float
        :param use_lu_fact: use LU factorization
        :type use_lu_fact: bool
        :param norm0: reference value of the norm of the residual to compute
            the decrease stop criteria.
            Iterations stops when norm(residual)/norm0<tolerance
        :type norm0: float
        """
        super(MDASequential, self).__init__(
            disciplines,
            name=name,
            grammar_type=grammar_type,
            max_mda_iter=max_mda_iter,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            norm0=norm0,
        )
        self._initialize_grammars()
        self._set_default_inputs()
        self._compute_input_couplings()

        self.mda_sequence = mda_sequence
        for mda in self.mda_sequence:
            mda.reset_history_each_run = True

    def _initialize_grammars(self):
        """Defines all inputs and outputs"""
        for discipline in self.disciplines:
            self.input_grammar.update_from(discipline.input_grammar)
            self.output_grammar.update_from(discipline.output_grammar)

    def _run(self):
        """Runs the MDAs in a sequential way

        :returns: the local data
        """
        self._couplings_warm_start()
        # execute MDAs in sequence
        if self.reset_history_each_run:
            self.residual_history = []
        for mda_i in self.mda_sequence:
            mda_i.reset_statuses_for_run()
            self.local_data = mda_i.execute(self.local_data)
            self.residual_history += mda_i.residual_history
            if mda_i.normed_residual < self.tolerance:
                break


class GSNewtonMDA(MDASequential):
    """
    Perform some GaussSeidel iterations and then NewtonRaphson iterations.
    """

    def __init__(
        self,
        disciplines,
        name=None,
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,
        tolerance=1e-6,
        max_mda_iter=10,
        relax_factor=0.99,
        linear_solver="lgmres",
        max_mda_iter_gs=3,
        linear_solver_tolerance=1e-12,
        warm_start=False,
        use_lu_fact=False,
        norm0=None,
        **newton_mda_options
    ):
        """
        Constructor

        :param disciplines: the disciplines list
        :type disciplines: list(MDODiscipline)
        :param name: name
        :type name: str
        :param grammar_type: the type of grammar to use for IO declaration
            either JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE
        :type grammar_type: str
        :param tolerance: tolerance of the iterative direct coupling solver,
            norm of the current residuals divided by initial residuals norm
            shall be lower than the tolerance to stop iterating
        :type tolerance: float
        :param max_mda_iter: maximum number of iterations
        :type max_mda_iter: int
        :param relax_factor: relaxation factor
        :type relax_factor: float
        :param linear_solver: type of linear solver to be used to solve
            the Newton problem
        :type linear_solver: str
        :param max_mda_iter_gs: maximum number of iterations of the GaussSeidel
            solver
        :type max_mda_iter_gs: int
        :param warm_start: if True, the second iteration and ongoing
            start from the previous coupling solution
        :type warm_start: bool
        :param linear_solver_tolerance: Tolerance of the linear solver
            in the adjoint equation
        :type linear_solver_tolerance: float
        :param use_lu_fact: if True, when using adjoint/forward
            differenciation, store a LU factorization of the matrix
            to solve faster multiple RHS problem
        :type use_lu_fact: bool
        :param newton_mda_options: options passed to the MDANewtonRaphson
        :type newton_mda_options: dict
        :param norm0: reference value of the norm of the residual to compute
            the decrease stop criteria.
            Iterations stops when norm(residual)/norm0<tolerance
        :type norm0: float
        """
        mda_gs = MDAGaussSeidel(disciplines, max_mda_iter=max_mda_iter_gs, name=None)
        mda_gs.tolerance = tolerance
        mda_newton = MDANewtonRaphson(
            disciplines,
            max_mda_iter,
            relax_factor,
            name=None,
            grammar_type=grammar_type,
            linear_solver=linear_solver,
            use_lu_fact=use_lu_fact,
            norm0=norm0,
            **newton_mda_options
        )
        sequence = [mda_gs, mda_newton]
        super(GSNewtonMDA, self).__init__(
            disciplines,
            sequence,
            name=name,
            grammar_type=grammar_type,
            max_mda_iter=max_mda_iter,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
        )
