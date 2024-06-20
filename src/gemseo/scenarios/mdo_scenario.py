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
#        :author: Pierre-Jean Barjhoux, Benoit Pauwels - MDOScenarioAdapter
#                                                        Jacobian computation
"""A scenario whose driver is an optimization algorithm."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.core.discipline import MDODiscipline
from gemseo.scenarios.scenario import Scenario

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.optimization_result import OptimizationResult
    from gemseo.typing import StrKeyMapping

# The detection of formulations requires to import them,
# before calling get_formulation_from_name


LOGGER = logging.getLogger(__name__)


class MDOScenario(Scenario):
    """A multidisciplinary scenario to be executed by an optimizer.

    an :class:`.MDOScenario` is a particular :class:`.Scenario` whose driver is an
    optimization algorithm. This algorithm must be implemented in an
    :class:`.BaseOptimizationLibrary`.
    """

    clear_history_before_run: bool
    """If ``True``, clear history before run."""

    # Constants for input variables in json schema
    MAX_ITER = "max_iter"
    X_OPT = "x_opt"

    def __init__(  # noqa:D107
        self,
        disciplines: Sequence[MDODiscipline],
        formulation: str,
        objective_name: str | Sequence[str],
        design_space: DesignSpace,
        name: str = "",
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

    def _run_algorithm(self) -> OptimizationResult:
        problem = self.formulation.optimization_problem
        algo_name = self.local_data[self.ALGO]
        max_iter = self.local_data[self.MAX_ITER]
        options = self.local_data.get(self.ALGO_OPTIONS)
        if options is None:
            options = {}
        if self.MAX_ITER in options:
            LOGGER.warning(
                "Double definition of algorithm option max_iter, keeping value: %s",
                max_iter,
            )
            options.pop(self.MAX_ITER)

        # Store the lib in case we rerun the same algorithm,
        # for multilevel scenarios for instance
        # This significantly speedups the process also because
        # of the option grammar that is long to create
        if self._algo_name is not None and self._algo_name == algo_name:
            lib = self._lib
        else:
            lib = self._algo_factory.create(algo_name)
            self._lib = lib

        self.optimization_result = lib.execute(problem, max_iter=max_iter, **options)
        return self.optimization_result

    def _init_algo_factory(self) -> None:
        self._algo_factory = OptimizationLibraryFactory(use_cache=True)

    def _update_input_grammar(self) -> None:
        super()._update_input_grammar()
        if self.grammar_type != self.GrammarType.JSON:
            self.input_grammar.update_from_types({
                "max_iter": int,
                "algo_options": dict,
            })
            self.input_grammar.required_names.remove("algo_options")

    def __setstate__(self, state: StrKeyMapping) -> None:
        super().__setstate__(state)
        # BaseOptimizationLibrary objects cannot be serialized, _algo_name and _lib are
        # set to None to force the lib creation in _run_algorithm.
        self._algo_name = None
        self._lib = None
