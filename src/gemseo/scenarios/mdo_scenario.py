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
from typing import ClassVar

from pydantic import Field
from pydantic import model_validator

from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.scenarios.scenario import Scenario

if TYPE_CHECKING:
    from typing_extensions import Self

LOGGER = logging.getLogger(__name__)


class MDOScenario(Scenario):
    """A multidisciplinary scenario to be executed by an optimizer.

    an :class:`.MDOScenario` is a particular :class:`.Scenario` whose driver is an
    optimization algorithm. This algorithm must be implemented in an
    :class:`.BaseOptimizationLibrary`.
    """

    _ALGO_FACTORY_CLASS: ClassVar[type[OptimizationLibraryFactory]] = (
        OptimizationLibraryFactory
    )

    class _BaseSettings(Scenario._BaseSettings):
        max_iter: int = Field(..., gt=0, description="The maximum number of iterations")

        @model_validator(mode="after")
        def check_max_iter(self) -> Self:
            if "max_iter" in self.algo_options:
                LOGGER.warning(
                    "Double definition of algorithm option max_iter, keeping value: %s",
                    self.max_iter,
                )
                self.algo_options.pop("max_iter")
            return self

    def _run(self) -> None:
        algo = self._algo_factory.create(self._settings.algo)
        self.optimization_result = algo.execute(
            self.formulation.optimization_problem,
            max_iter=self._settings.max_iter,
            **self._settings.algo_options,
        )
