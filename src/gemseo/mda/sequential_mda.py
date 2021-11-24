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
"""A chain of MDAs to build hybrids of MDA algorithms sequentially."""
from __future__ import division, unicode_literals

from typing import Any, Mapping, Optional, Sequence

from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.mda import MDA
from gemseo.mda.newton import MDANewtonRaphson


class MDASequential(MDA):
    """A sequence of elementary MDAs."""

    _ATTR_TO_SERIALIZE = MDA._ATTR_TO_SERIALIZE + ("mda_sequence",)

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        mda_sequence,  # type: Sequence[MDA]
        name=None,  # type: Optional[str]
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        max_mda_iter=10,  # type: int
        tolerance=1e-6,  # type: float
        linear_solver_tolerance=1e-12,  # type: float
        warm_start=False,  # type: bool
        use_lu_fact=False,  # type: bool
        coupling_structure=None,  # type: Optional[MDOCouplingStructure]
        linear_solver="DEFAULT",  # type: str
        linear_solver_options=None,  # type: Mapping[str,Any]
    ):  # type: (...) -> None
        """
        Args:
            mda_sequence: The sequence of MDAs.
        """
        super(MDASequential, self).__init__(
            disciplines,
            name=name,
            grammar_type=grammar_type,
            max_mda_iter=max_mda_iter,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            use_lu_fact=use_lu_fact,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
            coupling_structure=coupling_structure,
        )
        self._set_default_inputs()
        self._compute_input_couplings()

        self.mda_sequence = mda_sequence
        for mda in self.mda_sequence:
            mda.reset_history_each_run = True
            self._log_convergence = self._log_convergence or mda.log_convergence

    @MDA.log_convergence.setter
    def log_convergence(
        self,
        value,  # type: bool
    ):  # type: (...) -> None
        self._log_convergence = value
        for mda in self.mda_sequence:
            mda.log_convergence = value

    def _initialize_grammars(self):  # type: (...) -> None
        """Define all the inputs and outputs."""
        for discipline in self.disciplines:
            self.input_grammar.update_from(discipline.input_grammar)
            self.output_grammar.update_from(discipline.output_grammar)

    def _run(self):  # type: (...) -> None
        """Run the MDAs in a sequential way."""
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
    """Perform some Gauss-Seidel iterations and then Newton-Raphson iterations."""

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        name=None,  # type: Optional[str]
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        tolerance=1e-6,  # type: float
        max_mda_iter=10,  # type: int
        relax_factor=0.99,  # type: float
        linear_solver="DEFAULT",  # type: str
        max_mda_iter_gs=3,  # type: int
        linear_solver_tolerance=1e-12,  # type: float
        warm_start=False,  # type: bool
        use_lu_fact=False,  # type: bool
        coupling_structure=None,  # type: Optional[MDOCouplingStructure]
        linear_solver_options=None,  # type: Mapping[str,Any]
        log_convergence=False,  # type: bool
        **newton_mda_options  # type: float
    ):
        """
        Args:
            relax_factor: The relaxation factor.
            linear_solver: The type of linear solver
                to be used to solve the Newton problem.
            max_mda_iter_gs: The maximum number of iterations
                of the Gauss-Seidel solver.
            log_convergence: Whether to log the MDA convergence,
                expressed in terms of normed residuals.
            **newton_mda_options: The options passed to :class:`.MDANewtonRaphson`.
        """
        mda_gs = MDAGaussSeidel(
            disciplines,
            max_mda_iter=max_mda_iter_gs,
            name=None,
            log_convergence=log_convergence,
        )
        mda_gs.tolerance = tolerance
        mda_newton = MDANewtonRaphson(
            disciplines,
            max_mda_iter,
            relax_factor,
            name=None,
            grammar_type=grammar_type,
            linear_solver=linear_solver,
            use_lu_fact=use_lu_fact,
            coupling_structure=coupling_structure,
            log_convergence=log_convergence,
            linear_solver_options=linear_solver_options,
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
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
            coupling_structure=coupling_structure,
        )
