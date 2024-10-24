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
from typing import ClassVar

from strenum import StrEnum

from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.mda.base_mda import _BaseMDAProcessFlow
from gemseo.mda.base_mda_root import BaseMDARoot
from gemseo.utils.constants import N_CPUS
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import NDArray

    from gemseo.core._process_flow.base_process_flow import BaseProcessFlow
    from gemseo.core.coupling_structure import CouplingStructure
    from gemseo.core.discipline import Discipline
    from gemseo.core.discipline.discipline_data import DisciplineData
    from gemseo.typing import StrKeyMapping


LOGGER = logging.getLogger(__name__)


class _ProcessFlow(_BaseMDAProcessFlow):
    """The process data and execution flow."""


class MDANewtonRaphson(BaseMDARoot):
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

    _process_flow_class: ClassVar[type[BaseProcessFlow]] = _ProcessFlow

    class NewtonLinearSolver(StrEnum):
        """A linear solver for the Newton method."""

        DEFAULT = "DEFAULT"
        GMRES = "GEMRES"
        BICGSTAB = "BICGSTAB"

    __newton_linear_solver_options: StrKeyMapping
    """The options of the Newton linear solver."""

    _newton_coupling_names: list[str]
    """The coupling data names used to form the residuals."""

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        max_mda_iter: int = 10,
        name: str = "",
        linear_solver: str = "DEFAULT",
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        coupling_structure: CouplingStructure | None = None,
        log_convergence: bool = False,
        linear_solver_options: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        newton_linear_solver_name: NewtonLinearSolver = NewtonLinearSolver.DEFAULT,
        newton_linear_solver_options: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        use_threading: bool = True,
        n_processes: int = N_CPUS,
        acceleration_method: AccelerationMethod = AccelerationMethod.NONE,
        over_relaxation_factor: float = 0.99,
        execute_before_linearizing: bool = False,
    ) -> None:
        """
        Args:
            newton_linear_solver_name: The name of the linear solver for the Newton
                method.
            newton_linear_solver_options: The options for the Newton linear solver.

        Raises:
            ValueError: When there are no coupling variables, or when there are weakly
                coupled disciplines. In these cases, use MDAChain.
        """  # noqa:D205 D212 D415
        super().__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            use_lu_fact=use_lu_fact,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
            coupling_structure=coupling_structure,
            log_convergence=log_convergence,
            use_threading=use_threading,
            n_processes=n_processes,
            acceleration_method=acceleration_method,
            over_relaxation_factor=over_relaxation_factor,
            execute_before_linearizing=execute_before_linearizing,
        )

        # We use all couplings to form the Newton matrix otherwise the effect of the
        # weak couplings are not taken into account in the coupling updates.
        self._newton_coupling_names = sorted(self.all_couplings)

        self.linear_solver = linear_solver
        self.__newton_linear_solver_name = newton_linear_solver_name
        self.__newton_linear_solver_options = newton_linear_solver_options or {}

        if not self._newton_coupling_names:
            msg = "There is no couplings to compute. Please consider using MDAChain."
            raise ValueError(msg)

        strongly_coupled = self.coupling_structure.get_strongly_coupled_disciplines()

        if len(strongly_coupled) < len(self.disciplines):
            msg = (
                "The MDANewtonRaphson has weakly coupled disciplines, "
                "which is not supported. Please consider using a MDAChain with "
                "the option inner_mda_name='MDANewtonRaphson'."
            )
            raise ValueError(msg)

        self._set_differentiated_ios()

    def _set_differentiated_ios(self) -> None:
        """Set the differentiated inputs and outputs for the Newton algorithm.

        Also ensures that :attr:`.JacobianAssembly.sizes` contains the sizes of all
        the coupling sizes needed for Newton.
        """
        for discipline in self.disciplines:
            inputs_to_linearize = set(discipline.io.input_grammar.names).intersection(
                self._resolved_variable_names
            )
            outputs_to_linearize = set(discipline.io.output_grammar.names).intersection(
                self._resolved_variable_names
            )

            if (
                set(discipline.io.residual_to_state_variable.values())
                & set(self._resolved_variable_names)
                != set()
            ):
                outputs_to_linearize |= discipline.io.residual_to_state_variable.keys()

            # If outputs and inputs to linearize not empty, then linearize
            if inputs_to_linearize and outputs_to_linearize:
                discipline.add_differentiated_inputs(inputs_to_linearize)
                discipline.add_differentiated_outputs(outputs_to_linearize)

    def __compute_newton_step(
        self,
        input_data: dict[str, Any] | DisciplineData,
    ) -> NDArray:
        """Compute the full Newton step without relaxation.

        The Newton's step is defined as :math:`-[∂R/∂Y(y)]^{-1} R(y)`, where R(y) is the
        vector of coupling residuals and Y is the vector of couplings.

        Args:
            input_data: The input data for the disciplines.

        Returns:
            The Newton step.
        """
        self._linearize_disciplines(input_data)

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
            local_data_before_execution = self.io.data.copy()
            input_couplings = self.get_current_resolved_variables_vector()

            self._execute_disciplines_and_update_local_data()
            self._compute_residuals(local_data_before_execution)

            if self._stop_criterion_is_reached:
                break

            newton_step = self.__compute_newton_step(local_data_before_execution)
            updated_couplings = self._sequence_transformer.compute_transformed_iterate(
                input_couplings + newton_step,
                newton_step,
            )

            self._update_local_data_from_array(updated_couplings)
