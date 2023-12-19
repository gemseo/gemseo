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

from typing import TYPE_CHECKING

from gemseo.core.discipline import MDODiscipline
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.mda import MDA
from gemseo.mda.newton_raphson import MDANewtonRaphson

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any

    from gemseo.core.coupling_structure import MDOCouplingStructure


class MDASequential(MDA):
    """A sequence of elementary MDAs."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        mda_sequence: Sequence[MDA],
        name: str | None = None,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        max_mda_iter: int = 10,
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        coupling_structure: MDOCouplingStructure | None = None,
        linear_solver: str = "DEFAULT",
        linear_solver_options: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Args:
            mda_sequence: The sequence of MDAs.
        """  # noqa:D205 D212 D415
        super().__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            grammar_type=grammar_type,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            use_lu_fact=use_lu_fact,
            coupling_structure=coupling_structure,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
        )
        self._compute_input_couplings()

        self.mda_sequence = mda_sequence
        for mda in self.mda_sequence:
            mda.reset_history_each_run = True
            self._log_convergence = self._log_convergence or mda.log_convergence

    @MDA.log_convergence.setter
    def log_convergence(self, value: bool) -> None:  # noqa: D102
        self._log_convergence = value
        for mda in self.mda_sequence:
            mda.log_convergence = value

    def _run(self) -> None:
        super()._run()

        if self.reset_history_each_run:
            self.residual_history = []

        # Execute the MDAs in sequence
        for mda in self.mda_sequence:
            mda.reset_statuses_for_run()

            # Execute the i-th MDA
            self.local_data = mda.execute(self.local_data)

            # Extend the residual history
            self.residual_history += mda.residual_history

            if mda.normed_residual < self.tolerance:
                break


class MDAGSNewton(MDASequential):
    """Perform some Gauss-Seidel iterations and then Newton-Raphson iterations."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        name: str | None = None,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        tolerance: float = 1e-6,
        max_mda_iter: int = 10,
        relax_factor: float = 0.99,
        linear_solver: str = "DEFAULT",
        max_mda_iter_gs: int = 3,
        linear_solver_tolerance: float = 1e-12,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        coupling_structure: MDOCouplingStructure | None = None,
        linear_solver_options: Mapping[str, Any] | None = None,
        log_convergence: bool = False,
        **newton_mda_options: float | str | None,
    ) -> None:
        """
        Args:
            relax_factor: The relaxation factor.
            max_mda_iter_gs: The maximum number of iterations of the Gauss-Seidel MDA.
            newton_linear_solver: The name of the linear solver for the Newton method.
            newton_linear_solver_options: The options for the Newton linear solver.
            log_convergence: Whether to log the MDA convergence,
                expressed in terms of normed residuals.
            **newton_mda_options: The options for the Newton MDA.
        """  # noqa:D205 D212 D415
        mda_gauss_seidel = MDAGaussSeidel(
            disciplines,
            max_mda_iter=max_mda_iter_gs,
            tolerance=tolerance,
            log_convergence=log_convergence,
        )

        mda_newton = MDANewtonRaphson(
            disciplines,
            max_mda_iter=max_mda_iter,
            grammar_type=grammar_type,
            use_lu_fact=use_lu_fact,
            coupling_structure=coupling_structure,
            log_convergence=log_convergence,
            linear_solver_options=linear_solver_options,
            **newton_mda_options,
        )

        super().__init__(
            disciplines,
            [mda_gauss_seidel, mda_newton],
            max_mda_iter=max_mda_iter,
            name=name,
            grammar_type=grammar_type,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
            coupling_structure=coupling_structure,
        )
