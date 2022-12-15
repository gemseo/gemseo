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
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Class for the empirical estimation of statistics from a dataset.

Overview
--------

The :class:`.EmpiricalStatistics` class inherits
from the abstract :class:`.Statistics` class
and aims to estimate statistics from a :class:`.Dataset`,
based on empirical estimators.

Construction
------------

A :class:`.EmpiricalStatistics` is built from a :class:`.Dataset`
and optionally variables names.
In this case,
statistics are only computed for these variables.
Otherwise,
statistics are computed for all the variable available in the dataset.
Lastly,
the user can give a name to its :class:`.EmpiricalStatistics` object.
By default,
this name is the concatenation of 'EmpiricalStatistics'
and the name of the :class:`.Dataset`.
"""
from __future__ import annotations

from typing import Iterable

from numpy import all as np_all
from numpy import max as np_max
from numpy import mean
from numpy import min as np_min
from numpy import ndarray
from numpy import quantile
from numpy import std
from numpy import var
from scipy.stats import moment

from gemseo.core.dataset import Dataset
from gemseo.uncertainty.statistics.statistics import Statistics


class EmpiricalStatistics(Statistics):
    """Empirical estimation of statistics.

    Examples:
        >>> from gemseo.api import (
        ...     create_discipline,
        ...     create_parameter_space,
        ...     create_scenario)
        >>> from gemseo.uncertainty.statistics.empirical import EmpiricalStatistics
        >>>
        >>> expressions = {"y1": "x1+2*x2", "y2": "x1-3*x2"}
        >>> discipline = create_discipline(
        ...     "AnalyticDiscipline", expressions=expressions
        ... )
        >>>
        >>> parameter_space = create_parameter_space()
        >>> parameter_space.add_random_variable(
        ...     "x1", "OTUniformDistribution", minimum=-1, maximum=1
        ... )
        >>> parameter_space.add_random_variable(
        ...     "x2", "OTUniformDistribution", minimum=-1, maximum=1
        ... )
        >>>
        >>> scenario = create_scenario(
        ...     [discipline],
        ...     "DisciplinaryOpt",
        ...     "y1",
        ...     parameter_space,
        ...     scenario_type="DOE"
        ... )
        >>> scenario.execute({'algo': 'OT_MONTE_CARLO', 'n_samples': 100})
        >>>
        >>> dataset = scenario.export_to_dataset(opt_naming=False)
        >>>
        >>> statistics = EmpiricalStatistics(dataset)
        >>> mean = statistics.mean()
    """

    def __init__(  # noqa: D107
        self,
        dataset: Dataset,
        variables_names: Iterable[str] | None = None,
        name: str | None = None,
    ) -> None:
        name = name or dataset.name
        super().__init__(dataset, variables_names, name)

    def compute_maximum(self) -> dict[str, ndarray]:  # noqa: D102
        result = {name: np_max(self.dataset[name], 0) for name in self.names}
        return result

    def compute_mean(self) -> dict[str, ndarray]:  # noqa: D102
        result = {name: mean(self.dataset[name], 0) for name in self.names}
        return result

    def compute_minimum(self) -> dict[str, ndarray]:  # noqa: D102
        result = {name: np_min(self.dataset[name], 0) for name in self.names}
        return result

    def compute_probability(  # noqa: D102
        self,
        thresh: float,
        greater: bool = True,
    ) -> dict[str, ndarray]:
        if greater:
            result = {
                name: mean(np_all(self.dataset[name] >= thresh[name], 1))
                for name in self.names
            }
        else:
            result = {
                name: mean(np_all(self.dataset[name] <= thresh[name], 1))
                for name in self.names
            }
        return result

    def compute_quantile(  # noqa: D102
        self,
        prob: float,
    ) -> dict[str, ndarray]:
        result = {name: quantile(self.dataset[name], prob, 0) for name in self.names}
        return result

    def compute_standard_deviation(  # noqa: D102
        self,
    ) -> dict[str, ndarray]:
        result = {name: std(self.dataset[name], 0) for name in self.names}
        return result

    def compute_variance(self) -> dict[str, ndarray]:  # noqa: D102
        result = {name: var(self.dataset[name], 0) for name in self.names}
        return result

    def compute_moment(  # noqa: D102
        self,
        order: int,
    ) -> dict[str, ndarray]:
        result = {name: moment(self.dataset[name], order) for name in self.names}
        return result

    def compute_range(self) -> dict[str, ndarray]:  # noqa: D102
        lower = self.compute_minimum()
        upper = self.compute_maximum()
        result = {name: upper[name] - lower[name] for name in self.names}
        return result
