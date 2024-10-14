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
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from pydantic import Field
from pydantic import model_validator

from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.scenarios.scenario import Scenario

if TYPE_CHECKING:
    from typing_extensions import Self

    from gemseo.datasets.dataset import Dataset

# The detection of formulations requires to import them,
# before calling get_formulation_from_name
LOGGER = logging.getLogger(__name__)

_ALGO_FACTORY_CLASS = DOELibraryFactory

# This is only used for validation in the settings model.
_SETTINGS_ALGO_FACTORY: Final[_ALGO_FACTORY_CLASS] = _ALGO_FACTORY_CLASS(use_cache=True)


class DOEScenario(Scenario):
    """A multidisciplinary scenario to be executed by a design of experiments (DOE).

    A :class:`.DOEScenario` is a particular :class:`.Scenario` whose driver is a DOE.
    This DOE must be implemented in a :class:`.BaseDOELibrary`.
    """

    _ALGO_FACTORY_CLASS: ClassVar[type[DOELibraryFactory]] = _ALGO_FACTORY_CLASS

    class _BaseSettings(Scenario._BaseSettings):
        n_samples: int = Field(0, ge=0, description="The number of samples.")

        @model_validator(mode="after")
        def check_max_iter(self) -> Self:
            n_samples = self.n_samples
            if n_samples > 0:
                algo = _SETTINGS_ALGO_FACTORY.create(self.algo)
                if "n_samples" in algo.ALGORITHM_INFOS[self.algo].Settings.model_fields:
                    if "n_samples" in self.algo_options:
                        LOGGER.warning(
                            "Double definition of the algorithm setting n_samples, "
                            "keeping value: %s.",
                            n_samples,
                        )
                    self.algo_options["n_samples"] = n_samples
            return self

    default_input_data = MappingProxyType({"algo": "lhs"})

    def _run(self) -> None:
        algo = self._algo_factory.create(self._settings.algo)
        self.optimization_result = algo.execute(
            self.formulation.optimization_problem,
            **self._settings.algo_options,
        )

    def to_dataset(  # noqa: D102
        self,
        name: str = "",
        categorize: bool = True,
        opt_naming: bool = True,
        export_gradients: bool = False,
    ) -> Dataset:
        # The algo is not instantiated again since it is in the factory cache.
        algo = self._algo_factory.create(self._settings.algo)
        return self.formulation.optimization_problem.to_dataset(
            name=name,
            categorize=categorize,
            opt_naming=opt_naming,
            export_gradients=export_gradients,
            input_values=algo.samples,
        )
