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
"""A Gauss Seidel algorithm for solving MDAs."""
from __future__ import division, unicode_literals

from typing import Optional, Sequence

from gemseo.core.chain import MDOChain
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.mda.mda import MDA


class MDAGaussSeidel(MDA):
    """An MDA analysis based on the Gauss-Seidel algorithm.

    This algorithm is an iterative technique to solve the linear system:

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
        disciplines,  # type: Sequence[MDODiscipline]
        name=None,  # type: Optional[str]
        max_mda_iter=10,  # type: int
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        tolerance=1e-6,  # type: float
        linear_solver_tolerance=1e-12,  # type: float
        warm_start=False,  # type: bool
        use_lu_fact=False,  # type: bool
        over_relax_factor=1.0,  # type: float
        coupling_structure=None,  # type: Optional[MDOCouplingStructure]
        log_convergence=False #type: bool
    ):  # type: (...) -> None
        """
        Args:
            over_relax_factor: The relaxation coefficient,
                used to make the method more robust,
                if ``0<over_relax_factor<1`` or faster if ``1<over_relax_factor<=2``.
                If ``over_relax_factor =1.``, it is deactivated.
            log_convergence: Whether to log the MDA convergence,
                expressed in terms of normed residuals.
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
            coupling_structure=coupling_structure,
            log_convergence=log_convergence,
        )
        assert over_relax_factor > 0.0
        assert over_relax_factor <= 2.0
        self.over_relax_factor = over_relax_factor
        self._initialize_grammars()
        self._set_default_inputs()
        self._compute_input_couplings()

    def _initialize_grammars(self):
        self.input_grammar.update_from(self.chain.input_grammar)
        self.output_grammar.update_from(self.chain.output_grammar)

    def _run(self):
        # Run the disciplines in a sequential way
        # until the difference between outputs is under tolerance.
        if self.warm_start:
            self._couplings_warm_start()
        current_couplings = 0.0

        relax = self.over_relax_factor
        use_relax = relax != 1.0
        # store initial residual
        current_iter = 0
        while not self._termination(current_iter) or current_iter == 0:
            for discipline in self.disciplines:
                discipline.execute(self.local_data)
                outs = discipline.get_output_data()
                if use_relax:
                    # First time this output is computed, update directly local data
                    self.local_data.update(
                        {k: v for k, v in outs.items() if k not in self.local_data}
                    )
                    # The couplings already exist in the local data,
                    # so the over relaxation can be applied
                    self.local_data.update(
                        {
                            k: relax * v + (1.0 - relax) * self.local_data[k]
                            for k, v in outs.items()
                            if k in self.local_data
                        }
                    )
                else:
                    self.local_data.update(outs)

            new_couplings = self._current_strong_couplings()
            self._compute_residual(
                current_couplings,
                new_couplings,
                current_iter,
                first=current_iter == 0,
                log_normed_residual=self.log_convergence,
            )
            # store current residuals
            current_iter += 1
            current_couplings = new_couplings

        for discipline in self.disciplines:  # Update all outputs without relax
            self.local_data.update(discipline.get_output_data())
