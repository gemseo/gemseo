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
"""The Multi-disciplinary Design Feasible (MDF) formulation."""
from __future__ import annotations

from typing import Any

from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequence
from gemseo.core.formulation import MDOFormulation
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.mda.mda_factory import MDAFactory


class MDF(MDOFormulation):
    """The Multidisciplinary Design Feasible (MDF) formulation.

    This formulation draws an optimization architecture
    where:

    - the coupling of strongly coupled disciplines is made consistent
      by means of a Multidisciplinary Design Analysis (MDA),
    - the optimization problem
      with respect to the local and global design variables is made at the top level.

    Note that the multidisciplinary analysis is made at each optimization iteration.
    """

    def __init__(
        self,
        disciplines: list[MDODiscipline],
        objective_name: str,
        design_space: DesignSpace,
        maximize_objective: bool = False,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
        main_mda_name: str = "MDAChain",
        inner_mda_name: str = "MDAJacobi",
        **main_mda_options: Any,
    ):
        """
        Args:
            main_mda_name: The name of the class used for the main MDA,
                typically the :class:`.MDAChain`,
                but one can force to use :class:`.MDAGaussSeidel` for instance.
            inner_mda_name: The name of the class used for the inner-MDA of the main
                MDA, if any; typically when the main MDA is an :class:`.MDAChain`.
            **main_mda_options: The options of the main MDA, which may include
                those of the inner-MDA.
        """  # noqa: D205, D212, D415
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            maximize_objective=maximize_objective,
            grammar_type=grammar_type,
        )
        self.mda = None
        self._main_mda_name = main_mda_name
        self._mda_factory = MDAFactory()
        self._instantiate_mda(main_mda_name, inner_mda_name, **main_mda_options)
        self._update_design_space()
        self._build_objective()

    def get_top_level_disc(self) -> list[MDODiscipline]:  # noqa:D102
        return [self.mda]

    def _instantiate_mda(
        self,
        main_mda_name: str = "MDAChain",
        inner_mda_name: str = "MDAJacobi",
        **mda_options: Any,
    ) -> None:
        """Create the MDA discipline.

        Args:
            main_mda_name: The name of the class of the main MDA.
            inner_mda_name: The name of the class of the inner-MDA used by the
                main MDA.
        """
        if main_mda_name == "MDAChain":
            mda_options["inner_mda_name"] = inner_mda_name

        self.mda = self._mda_factory.create(
            main_mda_name,
            self.disciplines,
            grammar_type=self._grammar_type,
            **mda_options,
        )

    @classmethod
    def get_sub_options_grammar(cls, **options: str) -> JSONGrammar:  # noqa:D102
        main_mda = options.get("main_mda_name")
        if main_mda is None:
            raise ValueError(
                "main_mda_name option required to deduce the sub options of MDF."
            )
        factory = MDAFactory().factory
        return factory.get_options_grammar(main_mda)

    @classmethod
    def get_default_sub_options_values(cls, **options: str) -> dict:  # noqa:D102
        main_mda = options.get("main_mda_name")
        if main_mda is None:
            raise ValueError(
                "main_mda_name option required to deduce the sub options of MDF."
            )
        factory = MDAFactory().factory
        return factory.get_default_options_values(main_mda)

    def _build_objective(self) -> None:
        """Build the objective function from the MDA and the objective name."""
        self._build_objective_from_disc(self._objective_name, discipline=self.mda)

    def get_expected_workflow(  # noqa:D102
        self,
    ) -> list[ExecutionSequence, tuple[ExecutionSequence]]:
        return self.mda.get_expected_workflow()

    def get_expected_dataflow(  # noqa:D102
        self,
    ) -> list[tuple[MDODiscipline, MDODiscipline, list[str]]]:
        return self.mda.get_expected_dataflow()

    def _update_design_space(self) -> None:
        """Update the design space by removing the coupling variables."""
        self._set_default_input_values_from_design_space()
        # No couplings in design space (managed by MDA)
        self._remove_couplings_from_ds()
        # Cleanup
        self._remove_unused_variables()

    def _remove_couplings_from_ds(self) -> None:
        """Remove the coupling variables from the design space."""
        design_space = self.opt_problem.design_space
        for coupling in self.mda.all_couplings:
            if coupling in design_space.variables_names:
                design_space.remove_variable(coupling)
