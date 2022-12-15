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
"""A formulation for uncoupled or weakly coupled problems."""
from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.core.chain import MDOChain
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequence
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.formulation import MDOFormulation
from gemseo.disciplines.utils import get_all_inputs


class DisciplinaryOpt(MDOFormulation):
    """The disciplinary optimization.

    This formulation draws the architecture of a mono-disciplinary optimization process
    from an ordered list of disciplines, an objective function and a design space.
    """

    def __init__(  # noqa:D107
        self,
        disciplines: list[MDODiscipline],
        objective_name: str,
        design_space: DesignSpace,
        maximize_objective: bool = False,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
    ) -> None:
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            maximize_objective=maximize_objective,
            grammar_type=grammar_type,
        )
        self.chain = None
        if len(disciplines) > 1:
            self.chain = MDOChain(disciplines, grammar_type=grammar_type)
        self._filter_design_space()
        self._set_default_input_values_from_design_space()
        # Build the objective from its objective name
        self._build_objective_from_disc(objective_name)

    def get_expected_workflow(  # noqa:D102
        self,
    ) -> list[ExecutionSequence, tuple[ExecutionSequence]]:
        if self.chain is None:
            return ExecutionSequenceFactory.serial(self.disciplines[0])
        return self.chain.get_expected_workflow()

    def get_expected_dataflow(  # noqa:D102
        self,
    ) -> list[tuple[MDODiscipline, MDODiscipline, list[str]]]:
        if self.chain is None:
            return []
        return self.chain.get_expected_dataflow()

    def get_top_level_disc(self) -> list[MDODiscipline]:  # noqa:D102
        if self.chain is not None:
            return [self.chain]
        return self.disciplines

    def _filter_design_space(self) -> None:
        """Filter the design space to keep only available variables."""
        all_inpts = get_all_inputs(self.get_top_level_disc())
        kept = set(self.design_space.variables_names) & set(all_inpts)
        self.design_space.filter(kept)
