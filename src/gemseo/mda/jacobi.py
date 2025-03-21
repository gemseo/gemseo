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
"""A Jacobi algorithm for solving MDAs."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.mda.base_mda import BaseProcessFlow
from gemseo.mda.base_mda import _BaseMDAProcessFlow
from gemseo.mda.base_parallel_mda_solver import BaseParallelMDASolver
from gemseo.mda.jacobi_settings import MDAJacobi_Settings

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import ClassVar

    from gemseo.core.coupling_structure import DependencyGraph
    from gemseo.core.discipline import Discipline


class _ProcessFlow(_BaseMDAProcessFlow):
    """The process data and execution flow."""

    def _get_disciplines_couplings(
        self, graph: DependencyGraph
    ) -> list[tuple[str, str, list[str]]]:
        couplings_results = []
        for disc in self._node.disciplines:
            in_data = graph.graph.get_edge_data(self._node, disc)
            if in_data:
                couplings_results.append((
                    self._node,
                    disc,
                    sorted(in_data["io"]),
                ))
            out_data = graph.graph.get_edge_data(disc, self._node)
            if out_data:
                couplings_results.append((
                    disc,
                    self._node,
                    sorted(out_data["io"]),
                ))

        return couplings_results


class MDAJacobi(BaseParallelMDASolver):
    r"""Perform an MDA using the Jacobi algorithm.

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

    Beginning with :math:`x_1^{(0)}, \dots, x_n^{(0)}`, the iterates are obtained as the
    solution of the following :math:`n` **independent** non-linear equations:

    .. math::

        \left\{
            \begin{matrix}
                r_1\left( x_1^{(i+1)} \right) =
                    F_1(x_1^{(i+1)}, x_2^{(i)}, \dots, x_n^{(i)}) = 0 \\
                r_2\left( x_2^{(i+1)} \right) =
                    F_2(x_1^{(i)}, x_2^{(i+1)}, \dots, x_n^{(i)}) = 0 \\
                \vdots \\
                r_n\left( x_n^{(i+1)} \right) =
                F_n(x_1^{(i)}, x_2^{(i)}, \dots, x_n^{(i+1)}) = 0
            \end{matrix}
        \right.
    """

    Settings: ClassVar[type[MDAJacobi_Settings]] = MDAJacobi_Settings
    """The pydantic model for the settings."""

    settings: MDAJacobi_Settings
    """The settings of the MDA"""

    _process_flow_class: ClassVar[type[BaseProcessFlow]] = _ProcessFlow

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        settings_model: MDAJacobi_Settings | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(disciplines, settings_model=settings_model, **settings)

        self._compute_input_coupling_names()
        self._set_resolved_variables(self._input_couplings)

    def get_process_flow(self) -> BaseProcessFlow:  # noqa: D102
        process_flow = super().get_process_flow()
        process_flow.is_parallel = self.settings.n_processes > 1
        return process_flow

    def _compute_input_coupling_names(self) -> None:
        """Compute the coupling variables that are inputs of the MDA.

        This must be overloaded here because the Jacobi algorithm induces a delay
        between the couplings, the strong couplings may be fully resolved but the weak
        ones may need one more iteration. The base MDA class uses strong couplings only,
        which is not satisfying here if some disciplines are not strongly coupled.
        """
        if len(self.coupling_structure.strongly_coupled_disciplines) == len(
            self._disciplines
        ):
            return super()._compute_input_coupling_names()

        self._input_couplings = sorted(
            set(self.coupling_structure.all_couplings).intersection(
                self.io.input_grammar
            )
        )

        self._numeric_input_couplings = sorted(
            set(self._input_couplings).difference(self._non_numeric_array_variables)
        )

        return None

    def _execute(self) -> None:
        super()._execute()

        while True:
            local_data_before_execution = self.io.data.copy()
            self._execute_disciplines_and_update_local_data()
            self._compute_residuals(local_data_before_execution)

            if self._check_stopping_criteria():
                break

            updated_couplings = self._sequence_transformer.compute_transformed_iterate(
                self.get_current_resolved_variables_vector(),
                self.get_current_resolved_residual_vector(),
            )

            self._update_local_data_from_array(updated_couplings)
