# -*- coding: utf-8 -*-
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
from __future__ import division, unicode_literals

import logging
from typing import Any, Dict, List, Sequence, Tuple

from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequence
from gemseo.core.formulation import MDOFormulation
from gemseo.core.json_grammar import JSONGrammar
from gemseo.mda.mda_factory import MDAFactory

LOGGER = logging.getLogger(__name__)


class MDF(MDOFormulation):
    """The Multidisciplinary Design Feasible (MDF) formulation.

    This formulation draws an optimization architecture
    where:

    - the coupling of strongly coupled disciplines is made consistent
      by means of a Multidisciplinary Design Analysis (MDA),
    - the optimization problem
      with respect to the local and global design variables is made at the top level.

    Note that the multidisciplinary analysis is made at a each optimization iteration.
    """

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        objective_name,  # type: str
        design_space,  # type: DesignSpace
        maximize_objective=False,  # type: bool
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        main_mda_class="MDAChain",  # type: str
        sub_mda_class="MDAJacobi",  # type: str
        **mda_options  # type: Any
    ):
        """
        Args:
            main_mda_class: The name of the class used for the main MDA,
                typically the :class:`.MDAChain`,
                but one can force to use :class:`.MDAGaussSeidel` for instance.
            sub_mda_class: The name of the class used for the sub-MDA.
            **mda_options: The options passed to the MDA at construction.
        """
        super(MDF, self).__init__(
            disciplines,
            objective_name,
            design_space,
            maximize_objective=maximize_objective,
            grammar_type=grammar_type,
        )
        self.mda = None
        self._main_mda_class = main_mda_class
        self._mda_factory = MDAFactory()
        self._instantiate_mda(main_mda_class, sub_mda_class, **mda_options)
        self._update_design_space()
        self._build_objective()

    def get_top_level_disc(self):  # type: (...) -> List[MDODiscipline]
        return [self.mda]

    def _instantiate_mda(
        self,
        main_mda_class="MDAChain",  # type: str
        sub_mda_class="MDAJacobi",  # type: str
        **mda_options  # type:Any
    ):  # type: (...) -> None
        """Create the MDA discipline.

        Args:
            main_mda_class: The name of the class of the main MDA.
            sub_mda_class: The name of the class of the sub-MDA used by the main MDA.
        """
        if main_mda_class == "MDAChain":
            mda_options["sub_mda_class"] = sub_mda_class

        self.mda = self._mda_factory.create(
            main_mda_class,
            self.disciplines,
            grammar_type=self._grammar_type,
            **mda_options
        )

    @classmethod
    def get_sub_options_grammar(
        cls, **options  # type: str
    ):  # type: (...) -> JSONGrammar
        main_mda = options.get("main_mda_class")
        if main_mda is None:
            raise ValueError(
                "main_mda_class option required to deduce the sub options of MDF."
            )
        factory = MDAFactory().factory
        return factory.get_options_grammar(main_mda)

    @classmethod
    def get_default_sub_options_values(
        cls, **options  # type:str
    ):  # type: (...) -> Dict
        main_mda = options.get("main_mda_class")
        if main_mda is None:
            raise ValueError(
                "main_mda_class option required to deduce the sub options of MDF."
            )
        factory = MDAFactory().factory
        return factory.get_default_options_values(main_mda)

    def _build_objective(self):  # type: (...) -> None
        """Build the objective function from the MDA and the objective name."""
        self._build_objective_from_disc(self._objective_name, discipline=self.mda)

    def get_expected_workflow(
        self,
    ):  # type: (...) -> List[ExecutionSequence,Tuple[ExecutionSequence]]
        return self.mda.get_expected_workflow()

    def get_expected_dataflow(
        self,
    ):  # type: (...) -> List[Tuple[MDODiscipline,MDODiscipline,List[str]]]
        return self.mda.get_expected_dataflow()

    def _update_design_space(self):  # type: (...) -> None
        """Update the design space by removing the coupling variables."""
        self._set_defaultinputs_from_ds()
        # No couplings in design space (managed by MDA)
        self._remove_couplings_from_ds()
        # Cleanup
        self._remove_unused_variables()

    def _remove_couplings_from_ds(self):  # type: (...) -> None
        """Remove the coupling variables from the design space."""
        design_space = self.opt_problem.design_space
        for coupling in self.mda.all_couplings:
            if coupling in design_space.variables_names:
                design_space.remove_variable(coupling)
