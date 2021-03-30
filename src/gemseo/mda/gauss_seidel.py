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
A Gauss Seidel algorithm for solving MDAs
*****************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import super

from future import standard_library

from gemseo.core.chain import MDOChain
from gemseo.core.discipline import MDODiscipline
from gemseo.mda.mda import MDA

standard_library.install_aliases()


class MDAGaussSeidel(MDA):
    """
    Perform a MDA analysis using a Gauss-Seidel algorithm,
    an iterative technique to solve the linear system:

    .. math::

       Ax = b

    by decomposing the matrix :math:`A`
    into the sum of a lower triangular matrix :math:`L_*`
    and a strictly upper triangular matrix :math:`U`.

    The new iterate is given by:

    .. math::

       x_{k+1} = L_*^{-1}(b-Ux_k)
    """

    def __init__(
        self,
        disciplines,
        name=None,
        max_mda_iter=10,
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,
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
        :param max_mda_iter: maximum number of iterations
        :type max_mda_iter: int
        :param name: the name of the chain
        :type name: str
        :param grammar_type: the type of grammar to use for IO declaration
            either JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE
        :type grammar_type: str
        :param tolerance: tolerance of the iterative direct coupling solver,
            norm of the current residuals divided by initial residuals norm
            shall be lower than the tolerance to stop iterating
        :type tolerance: float
        :param linear_solver_tolerance: Tolerance of the linear solver
            in the adjoint equation
        :type linear_solver_tolerance: float
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
        self.chain = MDOChain(disciplines, grammar_type=grammar_type)
        super(MDAGaussSeidel, self).__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            grammar_type=grammar_type,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            use_lu_fact=use_lu_fact,
        )
        self._initialize_grammars()
        self._set_default_inputs()
        self._compute_input_couplings()

    def reset_statuses_for_run(self):
        """Sets all the statuses to PENDING"""
        super(MDAGaussSeidel, self).reset_statuses_for_run()
        self.chain.reset_statuses_for_run()

    def _initialize_grammars(self):
        """Defines all inputs and outputs of the chain"""
        # self.chain.initialize_grammars()
        self.input_grammar.update_from(self.chain.input_grammar)
        self.output_grammar.update_from(self.chain.output_grammar)

    def _run(self):
        """Runs the disciplines in a sequential way until the difference
        between outputs is under tolerance

        :returns: the local data
        """
        if self.warm_start:
            self._couplings_warm_start()
        # execute the disciplines
        current_couplings = self._current_input_couplings()
        self.chain.reset_statuses_for_run()
        self.local_data.update(self.chain.execute(self.local_data))
        new_couplings = self._current_input_couplings()

        # store initial residual
        current_iter = 1
        self._compute_residual(
            current_couplings, new_couplings, current_iter, first=True
        )
        current_couplings = new_couplings

        while not self._termination(current_iter):
            self.chain.reset_statuses_for_run()
            self.local_data.update(self.chain.execute(self.local_data))
            new_couplings = self._current_input_couplings()

            # store current residual
            current_iter += 1
            self._compute_residual(current_couplings, new_couplings, current_iter)
            current_couplings = new_couplings
