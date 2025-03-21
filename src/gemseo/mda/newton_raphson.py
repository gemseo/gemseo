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

from gemseo.mda.base_mda import _BaseMDAProcessFlow
from gemseo.mda.base_parallel_mda_solver import BaseParallelMDASolver
from gemseo.mda.newton_raphson_settings import MDANewtonRaphson_Settings

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import NDArray

    from gemseo.core._process_flow.base_process_flow import BaseProcessFlow
    from gemseo.core.discipline import Discipline
    from gemseo.core.discipline.discipline_data import DisciplineData


LOGGER = logging.getLogger(__name__)


class _ProcessFlow(_BaseMDAProcessFlow):
    """The process data and execution flow."""


class MDANewtonRaphson(BaseParallelMDASolver):
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

    Settings: ClassVar[type[MDANewtonRaphson_Settings]] = MDANewtonRaphson_Settings
    """The pydantic model for the settings."""

    settings: MDANewtonRaphson_Settings
    """The settings of the MDA"""

    _process_flow_class: ClassVar[type[BaseProcessFlow]] = _ProcessFlow

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        settings_model: MDANewtonRaphson_Settings | None = None,
        **settings: Any,
    ) -> None:
        """
        Raises:
            ValueError: When there are no coupling variables, or when there are weakly
                coupled disciplines. In these cases, use MDAChain.
        """  # noqa:D205 D212 D415
        super().__init__(disciplines, settings_model=settings_model, **settings)

        # We use all couplings to form the Newton matrix otherwise the effect of the
        # weak couplings are not taken into account in the coupling updates.
        if not sorted(self.coupling_structure.all_couplings):
            msg = "There is no couplings to compute. Please consider using MDAChain."
            raise ValueError(msg)

        strongly_coupled = self.coupling_structure.get_strongly_coupled_disciplines()

        if len(strongly_coupled) < len(self._disciplines):
            msg = (
                "The MDANewtonRaphson has weakly coupled disciplines, "
                "which is not supported. Please consider using a MDAChain with "
                "the option inner_mda_name='MDANewtonRaphson'."
            )
            raise ValueError(msg)

        self._set_resolved_variables(self.coupling_structure.strong_couplings)
        self._set_differentiated_ios()

    def _set_differentiated_ios(self) -> None:
        """Set the differentiated inputs and outputs for the Newton algorithm.

        Also ensures that :attr:`.JacobianAssembly.sizes` contains the sizes of all
        the coupling sizes needed for Newton.
        """
        for discipline in self._disciplines:
            inputs_to_linearize = set(discipline.io.input_grammar).intersection(
                self._resolved_variable_names
            )
            outputs_to_linearize = set(discipline.io.output_grammar).intersection(
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
            self.settings.newton_linear_solver_name,
            matrix_type=self.matrix_type,
            residuals=self.get_current_resolved_residual_vector(),
            resolved_residual_names=self._resolved_residual_names,
            **self.settings.newton_linear_solver_settings,
        )

        if not is_converged:
            LOGGER.warning(
                "The linear solver %s failed "
                "to converge during the Newton's step computation.",
                self.settings.newton_linear_solver_name,
            )

        return newton_step

    def _execute(self) -> None:
        super()._execute()

        while True:
            local_data_before_execution = self.io.data.copy()
            input_couplings = self.get_current_resolved_variables_vector()

            self._execute_disciplines_and_update_local_data()
            self._compute_residuals(local_data_before_execution)

            if self._check_stopping_criteria():
                break

            newton_step = self.__compute_newton_step(local_data_before_execution)
            updated_couplings = self._sequence_transformer.compute_transformed_iterate(
                input_couplings + newton_step,
                newton_step,
            )

            self._update_local_data_from_array(updated_couplings)
