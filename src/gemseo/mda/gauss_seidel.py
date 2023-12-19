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

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.discipline import MDODiscipline
from gemseo.mda.base_mda_solver import BaseMDASolver

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any

    from gemseo.core.coupling_structure import MDOCouplingStructure


class MDAGaussSeidel(BaseMDASolver):
    r"""Perform an MDA using the Gauss-Seidel algorithm.

    This algorithm is a fixed point iteration method to solve systems of non-linear
    equations of the form,

    .. math::

        \left\{
            \begin{matrix}
                F_1(x_1, x_2, \dots, x_n) = 0 \\
                F_2(x_1, x_2, \dots, x_n) = 0 \\
                \vdots \\
                F_n(x_1, x_2, \dots, x_n) = 0
            \end{matrix}
        \right.

    Beginning with :math:`x_1^{(0)}, \dots, x_n^{(0)}`, the iterates are obtained by
    performing **sequentially** the following :math:`n` steps.

    **Step 1:** knowing :math:`x_2^{(i)}, \dots, x_n^{(i)}`, compute :math:`x_1^{(i+1)}`
    by solving,

    .. math::

        r_1\left( x_1^{(i+1)} \right) =
            F_1(x_1^{(i+1)}, x_2^{(i)}, \dots, x_n^{(i)}) = 0.

    **Step** :math:`k \leq n`: knowing :math:`x_1^{(i+1)}, \dots, x_{k-1}^{(i+1)}` on
    one hand, and :math:`x_{k+1}^{(i)}, \dots, x_n^{(i)}` on the other hand, compute
    :math:`x_1^{(i+1)}` by solving,

    .. math::

        r_k\left( x_k^{(i+1)} \right) = F_1(x_1^{(i+1)}, \dots, x_{k-1}^{(i+1)},
        x_k^{(i+1)}, x_{k+1}^{(i)}, \dots, x_n^{(i)}) = 0.

    These :math:`n` steps account for one iteration of the Gauss-Seidel method.
    """

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[MDODiscipline],
        name: str | None = None,
        max_mda_iter: int = 10,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        over_relax_factor: float | None = None,  # TODO: API: Remove the argument.
        coupling_structure: MDOCouplingStructure | None = None,
        log_convergence: bool = False,
        linear_solver: str = "DEFAULT",
        linear_solver_options: Mapping[str, Any] | None = None,
        acceleration_method: AccelerationMethod = AccelerationMethod.NONE,
        over_relaxation_factor: float = 1.0,
    ) -> None:
        """
        Args:
            over_relax_factor: Deprecated, please consider using
                :attr:`MDA.over_relaxation_factor` instead.
                The relaxation coefficient, used to make the method more robust, if
                ``0<over_relax_factor<1`` or faster if ``1<over_relax_factor<=2``. If
                ``over_relax_factor =1.``, it is deactivated.
        """  # noqa:D205 D212 D415
        # TODO: API: Remove the old name and attributes for over-relaxation factor.
        if over_relax_factor is not None:
            over_relaxation_factor = over_relax_factor

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
            log_convergence=log_convergence,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
            acceleration_method=acceleration_method,
            over_relaxation_factor=over_relaxation_factor,
        )

        self._compute_input_couplings()
        self._set_resolved_variables(self.strong_couplings)

    # TODO: API: Remove the property and its setter.
    @property
    def over_relax_factor(self) -> float:
        """The over-relaxation factor."""
        return self.over_relaxation_factor

    @over_relax_factor.setter
    def over_relax_factor(self, over_relaxation_factor: float) -> None:
        self.over_relaxation_factor = over_relaxation_factor

    def _initialize_grammars(self) -> None:
        """Define the input and output grammars from the disciplines' ones."""
        for discipline in self.disciplines:
            self.input_grammar.update(
                discipline.input_grammar, exclude_names=self.output_grammar.keys()
            )
            self.output_grammar.update(discipline.output_grammar)

    def execute_all_disciplines(self) -> None:
        """Execute all the disciplines in sequence."""
        for discipline in self.disciplines:
            discipline.execute(self.local_data)
            self.local_data.update(discipline.get_output_data())

    def _run(self) -> None:
        super()._run()
        self.execute_all_disciplines()

        while True:
            input_data = self.local_data.copy()

            self.execute_all_disciplines()
            self._update_residuals(input_data)

            new_couplings = self._sequence_transformer.compute_transformed_iterate(
                self.get_current_resolved_variables_vector(),
                self.get_current_resolved_residual_vector(),
            )

            self._update_local_data(new_couplings)
            self._update_residuals(input_data)
            self._compute_residual(log_normed_residual=self._log_convergence)

            if self._stop_criterion_is_reached:
                break
