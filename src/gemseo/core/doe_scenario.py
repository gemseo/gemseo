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
from __future__ import annotations

import logging
from typing import Any
from typing import Mapping
from typing import Sequence

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.core.discipline import MDODiscipline
from gemseo.core.scenario import Scenario
from gemseo.datasets.dataset import Dataset

# The detection of formulations requires to import them,
# before calling get_formulation_from_name
LOGGER = logging.getLogger(__name__)


class DOEScenario(Scenario):
    """A multidisciplinary scenario to be executed by a design of experiments (DOE).

    A :class:`.DOEScenario` is a particular :class:`.Scenario` whose driver is a DOE.
    This DOE must be implemented in a :class:`.DOELibrary`.
    """

    # Constants for input variables in json schema
    N_SAMPLES = "n_samples"
    EVAL_JAC = "eval_jac"

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[MDODiscipline],
        formulation: str,
        objective_name: str | Sequence[str],
        design_space: DesignSpace,
        name: str | None = None,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        maximize_objective: bool = False,
        **formulation_options: Any,
    ) -> None:
        # This loads the right json grammars from class name
        super().__init__(
            disciplines,
            formulation,
            objective_name,
            design_space,
            name=name,
            grammar_type=grammar_type,
            maximize_objective=maximize_objective,
            **formulation_options,
        )
        self.default_inputs = {self.EVAL_JAC: False, self.ALGO: "lhs"}
        self.__samples = ()

    def _init_algo_factory(self) -> None:
        self._algo_factory = DOEFactory(use_cache=True)

    def _run_algorithm(self) -> None:
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

        options = dict(options)
        if self.N_SAMPLES in lib.opt_grammar:
            n_samples = self.local_data.get(self.N_SAMPLES)
            if self.N_SAMPLES in options:
                LOGGER.warning(
                    "Double definition of algorithm option n_samples, "
                    "keeping value: %s.",
                    n_samples,
                )
            options[self.N_SAMPLES] = n_samples

        self.optimization_result = lib.execute(self.formulation.opt_problem, **options)
        self.__samples = lib.samples
        return self.optimization_result

    def _update_input_grammar(self) -> None:  # noqa: D102
        super()._update_input_grammar()
        if self.grammar_type == self.GrammarType.SIMPLE:
            self.input_grammar.update_from_types(
                {
                    self.EVAL_JAC: bool,
                    "n_samples": int,
                    "algo_options": dict,
                }
            )
            for name in ("n_samples", "algo_options"):
                self.input_grammar.required_names.remove(name)

    def to_dataset(  # noqa: D102
        self,
        name: str | None = None,
        categorize: bool = True,
        opt_naming: bool = True,
        export_gradients: bool = False,
    ) -> Dataset:
        return self.formulation.opt_problem.to_dataset(
            name=name,
            categorize=categorize,
            opt_naming=opt_naming,
            export_gradients=export_gradients,
            input_values=self.__samples,
        )

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        super().__setstate__(state)
        # DOELibrary objects cannot be serialized, _algo_name and _lib are set to None
        # to force the lib creation in _run_algorithm.
        self._algo_name = None
        self._lib = None
