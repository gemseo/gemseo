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
#        :author: Charlie Vanaret, Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The Newton-Raphson algorithm for solving MDAs.

`Newton-Raphson <https://en.wikipedia.org/wiki/Newton%27s_method>`__
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.discipline import MDODiscipline
from gemseo.mda.root import MDARoot

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import NDArray

    from gemseo.core.coupling_structure import MDOCouplingStructure
    from gemseo.core.discipline_data import DisciplineData


LOGGER = logging.getLogger(__name__)


class MDANewtonRaphson(MDARoot):
    r"""Newton solver for MDA.

    The `Newton-Raphson method <https://en.wikipedia.org/wiki/Newton%27s_method>`__ is
    an iterative method to solve general equations of the form,

    .. math::

        F(x) = 0, \quad \text{where} \quad F: \mathbb{R}^n \rightarrow \mathbb{R}^n.

    Beginning with :math:`x_0 \in \mathbb{R}^n` the successive iterates are given by:

    .. math::

       x_{k+1} = x_k - J_f(x_k)^{-1} f(x_k),

    where :math:`J_f(x_k)` denotes the Jacobian of :math:`f` at :math:`x_k`.
    """

    # TODO: API: use a strenum
    __newton_linear_solver_name: str
    """The name of the linear solver for the Newton method.

    Available names are "DEFAULT", "GMRES" and "BICGSTAB".
    """

    __newton_linear_solver_options: Mapping[str, Any]
    """The options of the Newton linear solver."""

    _newton_coupling_names: list[str]
    """The coupling data names used to form the residuals."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        max_mda_iter: int = 10,
        relax_factor: float | None = None,  # TODO: API: Remove the argument.
        name: str | None = None,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        linear_solver: str = "DEFAULT",
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        coupling_structure: MDOCouplingStructure | None = None,
        log_convergence: bool = False,
        linear_solver_options: Mapping[str, Any] | None = None,
        newton_linear_solver_name: str = "DEFAULT",
        newton_linear_solver_options: Mapping[str, Any] | None = None,
        parallel: bool = False,
        use_threading: bool = True,
        n_processes: int = MDARoot.N_CPUS,
        acceleration_method: AccelerationMethod = AccelerationMethod.NONE,
        over_relaxation_factor: float = 0.99,
    ) -> None:
        """
        Args:
            relax_factor: The relaxation factor.
            newton_linear_solver_name: The name of the linear solver for the Newton
                method.
            newton_linear_solver_options: The options for the Newton linear solver.

        Raises:
            ValueError: When there are no coupling variables, or when there are weakly
                coupled disciplines. In these cases, use MDAChain.
        """  # noqa:D205 D212 D415
        # TODO API: Remove the old name and attributes for over-relaxatino factor.
        if relax_factor is not None:
            over_relaxation_factor = relax_factor

        super().__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            grammar_type=grammar_type,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            use_lu_fact=use_lu_fact,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
            coupling_structure=coupling_structure,
            log_convergence=log_convergence,
            parallel=parallel,
            use_threading=use_threading,
            n_processes=n_processes,
            acceleration_method=acceleration_method,
            over_relaxation_factor=over_relaxation_factor,
        )

        # We use all couplings to form the Newton matrix otherwise the effect of the
        # weak couplings are not taken into account in the coupling updates.
        self._newton_coupling_names = sorted(self.all_couplings)

        self.linear_solver = linear_solver
        self.__newton_linear_solver_name = newton_linear_solver_name
        self.__newton_linear_solver_options = newton_linear_solver_options or {}

        if not self._newton_coupling_names:
            raise ValueError(
                "There is no couplings to compute. Please consider using MDAChain."
            )

        strongly_coupled = self.coupling_structure.get_strongly_coupled_disciplines()

        if len(strongly_coupled) < len(self.disciplines):
            raise ValueError(
                "The MDANewtonRaphson has weakly coupled disciplines, "
                "which is not supported. Please consider using a MDAChain with "
                "the option inner_mda_name='MDANewtonRaphson'."
            )

        self.__set_differentiated_ios()

    # TODO: API: Remove the property and its setter.
    @property
    def relax_factor(self) -> float:
        """The over-relaxation factor."""
        return self.over_relaxation_factor

    @relax_factor.setter
    def relax_factor(self, relax_factor: float) -> None:
        self.over_relaxation_factor = relax_factor

    def __set_differentiated_ios(self) -> None:
        """Set the differentiated inputs and outputs for the Newton algorithm.

        Also ensures that :attr:`.JacobianAssembly.sizes` contains the sizes of all
        the coupling sizes needed for Newton.

        Args:
            couplings: The coupling variables.
            residual_variables: a mapping of residuals of disciplines to their
                respective state variables.
        """
        for discipline in self.disciplines:
            inputs_to_linearize = set(discipline.get_input_data_names()).intersection(
                self._resolved_variable_names
            )
            outputs_to_linearize = set(discipline.get_output_data_names()).intersection(
                self._resolved_variable_names
            )

            if (
                set(discipline.residual_variables.values())
                & set(self._resolved_variable_names)
                != set()
            ):
                outputs_to_linearize |= discipline.residual_variables.keys()

            # If outputs and inputs to linearize not empty, then linearize
            if inputs_to_linearize and outputs_to_linearize:
                discipline.add_differentiated_inputs(inputs_to_linearize)
                discipline.add_differentiated_outputs(outputs_to_linearize)

    # TODO: API: prepend with verb.
    def _newton_step(
        self,
        input_data: dict[str, Any] | DisciplineData,
    ) -> NDArray:
        """Compute the full Newton step without relaxation.

        The Newton's step is defined as :math:`-[∂R/∂Y(y)]^{-1} R(y)`, where R(y) is the
        vector of coupling residuals and Y is the vector of couplings.

        Args:
            input_data: The input data for the disciplines.
            residuals: The vector of coupling residuals.

        Returns:
            The Newton step.
        """
        newton_step, is_converged = self.assembly.compute_newton_step(
            input_data,
            self._resolved_variable_names,
            self.__newton_linear_solver_name,
            matrix_type=self.matrix_type,
            residuals=self.get_current_resolved_residual_vector(),
            resolved_residual_names=self._resolved_residual_names,
            **self.__newton_linear_solver_options,
        )

        if not is_converged:
            LOGGER.warning(
                "The linear solver %s failed "
                "to converge during the Newton's step computation.",
                self.__newton_linear_solver_name,
            )

        return newton_step

    def _run(self) -> None:
        super()._run()

        while True:
            input_data = self.local_data.copy()
            input_couplings = self.get_current_resolved_variables_vector()

            self.execute_all_disciplines(self.local_data)
            self.linearize_all_disciplines(input_data, execute=False)

            self._update_residuals(input_data)
            newton_step = self._newton_step(input_data)

            new_couplings = self._sequence_transformer.compute_transformed_iterate(
                input_couplings + newton_step,
                newton_step,
            )

            self._update_local_data(new_couplings)
            self._compute_residual(log_normed_residual=self._log_convergence)

            if self._stop_criterion_is_reached:
                break
