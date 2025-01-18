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
"""A Gauss-Seidel algorithm for solving MDAs."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.mda.base_mda import BaseProcessFlow
from gemseo.mda.base_mda import _BaseMDAProcessFlow
from gemseo.mda.base_mda_solver import BaseMDASolver
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.coupling_structure import DependencyGraph
    from gemseo.core.discipline import Discipline
    from gemseo.typing import StrKeyMapping


class _ProcessFlow(_BaseMDAProcessFlow):
    """The process data and execution flow."""

    def _get_disciplines_couplings(
        self, graph: DependencyGraph
    ) -> list[tuple[str, str, list[str]]]:
        couplings_results = []
        disc_already_seen = set()

        disciplines = BaseProcessFlow.get_disciplines_in_data_flow(self)

        for disc in disciplines:
            couplings_with_mda_to_be_removed = set()
            predecessors = (
                set(graph.graph.predecessors(disc)) - {self._node} & disc_already_seen
            )
            for predecessor in sorted(predecessors, key=lambda p: p.name):
                current_couplings = graph.graph.get_edge_data(predecessor, disc)["io"]
                couplings_results.append((predecessor, disc, sorted(current_couplings)))
                couplings_with_mda_to_be_removed.update(current_couplings)

            in_data = graph.graph.get_edge_data(self._node, disc)
            if in_data:
                couplings_with_mda = in_data["io"] - couplings_with_mda_to_be_removed
                if couplings_with_mda:
                    couplings_results.append((
                        self._node,
                        disc,
                        sorted(couplings_with_mda),
                    ))

            out_data = graph.graph.get_edge_data(disc, self._node)
            if out_data:
                couplings_results.append((
                    disc,
                    self._node,
                    sorted(out_data["io"]),
                ))

            disc_already_seen.add(disc)

        return couplings_results


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

    Settings: ClassVar[type[MDAGaussSeidel_Settings]] = MDAGaussSeidel_Settings
    """The pydantic model for the settings."""

    settings: MDAGaussSeidel_Settings
    """The settings of the MDA"""

    _process_flow_class: ClassVar[type[BaseProcessFlow]] = _ProcessFlow

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        settings_model: MDAGaussSeidel_Settings | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(disciplines, settings_model=settings_model, **settings)
        self._compute_input_coupling_names()
        self._set_resolved_variables(self.coupling_structure.strong_couplings)
        if self.settings.max_mda_iter == 0:
            del self.io.output_grammar[self.NORMALIZED_RESIDUAL_NORM]

    def _initialize_grammars(self) -> None:
        """Define the input and output grammars from the disciplines' ones."""
        for discipline in self._disciplines:
            self.io.input_grammar.update(
                discipline.io.input_grammar,
                excluded_names=self.io.output_grammar,
            )
            self.io.output_grammar.update(discipline.io.output_grammar)

    def _execute_disciplines_and_update_local_data(
        self, input_data: StrKeyMapping = READ_ONLY_EMPTY_DICT
    ) -> None:
        input_data = input_data or self.io.data
        for discipline in self._disciplines:
            discipline.execute(input_data)
            self.io.data.update(discipline.io.get_output_data())

        self._compute_names_to_slices()

    def _execute(self) -> None:
        super()._execute()
        self._execute_disciplines_and_update_local_data()
        if self.settings.max_mda_iter == 0:
            return

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
