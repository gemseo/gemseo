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
from __future__ import annotations

from typing import Any
from typing import Mapping
from typing import Sequence

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
        disciplines: Sequence[MDODiscipline],
        mda_sequence: Sequence[MDA],
        name: str | None = None,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
        max_mda_iter: int = 10,
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        coupling_structure: MDOCouplingStructure | None = None,
        linear_solver: str = "DEFAULT",
        linear_solver_options: Mapping[str, Any] = None,
    ) -> None:
        """
        Args:
            mda_sequence: The sequence of MDAs.
        """
        super().__init__(
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
        value: bool,
    ) -> None:
        self._log_convergence = value
        for mda in self.mda_sequence:
            mda.log_convergence = value

    def _initialize_grammars(self) -> None:
        """Define all the inputs and outputs."""
        for discipline in self.disciplines:
            self.input_grammar.update(discipline.input_grammar)
            self.output_grammar.update(discipline.output_grammar)
        self._add_residuals_norm_to_output_grammar()

    def _run(self) -> None:
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
        disciplines: Sequence[MDODiscipline],
        name: str | None = None,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
        tolerance: float = 1e-6,
        max_mda_iter: int = 10,
        relax_factor: float = 0.99,
        linear_solver: str = "DEFAULT",
        max_mda_iter_gs: int = 3,
        linear_solver_tolerance: float = 1e-12,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        coupling_structure: MDOCouplingStructure | None = None,
        linear_solver_options: Mapping[str, Any] = None,
        log_convergence: bool = False,
        **newton_mda_options: float,
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
            disciplines, max_mda_iter=max_mda_iter_gs, log_convergence=log_convergence
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
            **newton_mda_options,
        )
        sequence = [mda_gs, mda_newton]
        super().__init__(
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
