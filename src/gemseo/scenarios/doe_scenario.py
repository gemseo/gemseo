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
"""A multidisciplinary scenario to be executed by a design of experiments (DOE)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.scenarios.base_scenario import BaseScenario

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset


class DOEScenario(BaseScenario):
    """A multidisciplinary scenario to be executed by a design of experiments (DOE)."""

    _ALGO_FACTORY_CLASS: ClassVar[type[DOELibraryFactory]] = DOELibraryFactory

    def to_dataset(  # noqa: D102
        self,
        name: str = "",
        categorize: bool = True,
        opt_naming: bool = True,
        export_gradients: bool = False,
    ) -> Dataset:
        # The algo is not instantiated again since it is in the factory cache.
        algo = self._algo_factory.create(self._settings.algo_name)
        return self.formulation.optimization_problem.to_dataset(
            name=name,
            categorize=categorize,
            opt_naming=opt_naming,
            export_gradients=export_gradients,
            input_values=algo.samples,
        )
