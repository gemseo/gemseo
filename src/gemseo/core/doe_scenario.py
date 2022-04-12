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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                        documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A scenario whose driver is a design of experiments."""
from __future__ import division
from __future__ import unicode_literals

import logging
from typing import Any
from typing import Optional
from typing import Sequence

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.core.scenario import Scenario

# The detection of formulations requires to import them,
# before calling get_formulation_from_name
LOGGER = logging.getLogger(__name__)


class DOEScenario(Scenario):
    """A multidisciplinary scenario to be executed by a design of experiments (DOE).

    A :class:`.DOEScenario` is a particular :class:`.Scenario`
    whose driver is a DOE.
    This DOE must be implemented in a :class:`.DOELibrary`.

    Attributes:
        seed (int): The seed used by the random number generators for replicability.
    """

    # Constants for input variables in json schema
    N_SAMPLES = "n_samples"
    EVAL_JAC = "eval_jac"
    SEED = "seed"

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        formulation,  # type: str
        objective_name,  # type: str
        design_space,  # type: DesignSpace
        name=None,  # type: Optional[str]
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        **formulation_options,  # type: Any
    ):  # type: (...) -> None
        """
        Args:
            disciplines: The disciplines
                used to compute the objective, constraints and observables
                from the design variables.
            formulation: The name of the MDO formulation,
                also the name of a class inheriting from :class:`.MDOFormulation`.
            objective_name: The name of the objective.
            design_space: The design space.
            name: The name to be given to this scenario.
                If None, use the name of the class.
            grammar_type: The type of grammar to use for IO declaration
                , e.g. JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE.
            **formulation_options: The options
                to be passed to the :class:`.MDOFormulation`.
        """
        # This loads the right json grammars from class name
        super(DOEScenario, self).__init__(
            disciplines,
            formulation,
            objective_name,
            design_space,
            name,
            grammar_type,
            **formulation_options,
        )
        self.seed = 0
        self.default_inputs = {self.EVAL_JAC: False, self.ALGO: "lhs"}
        self.__samples = None

    def _init_algo_factory(self):  # type: (...) -> None
        self._algo_factory = DOEFactory()

    def _run_algorithm(self):  # type: (...) -> None
        self.seed += 1

        algo_name = self.local_data[self.ALGO]
        options = self.local_data.get(self.ALGO_OPTIONS)
        if options is None:
            options = {}

        # Store the lib in case we rerun the same algorithm,
        # for multilevel scenarios for instance
        # This significantly speedups the process
        # also because of the option grammar that is long to create
        if self._algo_name is not None and self._algo_name == algo_name:
            lib = self._lib
        else:
            lib = self._algo_factory.create(algo_name)
            lib.init_options_grammar(algo_name)
            self._lib = lib
            self._algo_name = algo_name

        if self.SEED in lib.opt_grammar.get_data_names() and self.SEED not in options:
            options[self.SEED] = self.seed

        if self.N_SAMPLES in lib.opt_grammar.get_data_names():
            n_samples = self.local_data.get(self.N_SAMPLES)
            if self.N_SAMPLES in options:
                LOGGER.warning(
                    "Double definition of algorithm option n_samples, keeping value: %s.",
                    n_samples,
                )
            options[self.N_SAMPLES] = n_samples

        self.optimization_result = lib.execute(self.formulation.opt_problem, **options)
        self.__samples = lib.samples

        return self.optimization_result

    def _update_grammar_input(self):  # type: (...) -> None
        self.input_grammar.update_elements(
            algo=str, n_samples=int, algo_options=dict, python_typing=True
        )
        self.input_grammar.update_required_elements(
            algo=True, n_samples=False, algo_options=False
        )

    def export_to_dataset(
        self,
        name=None,  # type: Optional[str]
        by_group=True,  # type: bool
        categorize=True,  # type: bool
        opt_naming=True,  # type: bool
        export_gradients=False,  # type: bool
    ):  # type: (...) -> Dataset
        return self.formulation.opt_problem.export_to_dataset(
            name=name,
            by_group=by_group,
            categorize=categorize,
            opt_naming=opt_naming,
            export_gradients=export_gradients,
            input_values=self.__samples,
        )
