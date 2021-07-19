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

from __future__ import division, unicode_literals

import logging
from typing import Dict, Iterable, Optional

from numpy import all as np_all
from numpy import max as np_max
from numpy import mean
from numpy import min as np_min
from numpy import ndarray, quantile, std, var
from scipy.stats import moment

from gemseo.core.dataset import Dataset
from gemseo.uncertainty.statistics.statistics import Statistics

LOGGER = logging.getLogger(__name__)


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
        ...     "AnalyticDiscipline", expressions_dict=expressions
        ... )
        >>> discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
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
        >>> dataset = discipline.cache.export_to_dataset()
        >>>
        >>> statistics = EmpiricalStatistics(dataset)
        >>> mean = statistics.mean()
    """

    def __init__(
        self,
        dataset,  # type: Dataset,
        variables_names=None,  # type: Optional[Iterable[str]]
        name=None,  # type: Optional[str]
    ):  # type: (...) -> None # noqa: D107,D205,D212,D415
        name = name or dataset.name
        super(EmpiricalStatistics, self).__init__(dataset, variables_names, name)

    def compute_maximum(self):  # type: (...) -> Dict[str, ndarray]  # noqa: D102
        result = {name: np_max(self.dataset[name], 0) for name in self.names}
        return result

    def compute_mean(self):  # type: (...) -> Dict[str, ndarray]  # noqa: D102
        result = {name: mean(self.dataset[name], 0) for name in self.names}
        return result

    def compute_minimum(self):  # type: (...) -> Dict[str, ndarray] # noqa: D102
        result = {name: np_min(self.dataset[name], 0) for name in self.names}
        return result

    def compute_probability(
        self,
        thresh,  # type: float
        greater=True,  # type: bool
    ):  # type: (...) -> Dict[str,ndarray]  # noqa: D102
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

    def compute_quantile(
        self,
        prob,  # type:float
    ):  # type: (...) -> Dict[str, ndarray] # noqa: D102
        result = {name: quantile(self.dataset[name], prob, 0) for name in self.names}
        return result

    def compute_standard_deviation(
        self,
    ):  # type: (...) -> Dict[str, ndarray]  # noqa: D102
        result = {name: std(self.dataset[name], 0) for name in self.names}
        return result

    def compute_variance(self):  # type: (...) -> Dict[str, ndarray]  # noqa: D102
        result = {name: var(self.dataset[name], 0) for name in self.names}
        return result

    def compute_moment(
        self,
        order,  # type: int
    ):  # type: (...) -> Dict[str, ndarray]  # noqa: D102
        result = {name: moment(self.dataset[name], order, 0) for name in self.names}
        return result

    def compute_range(self):  # type: (...) -> Dict[str, ndarray]  # noqa: D102
        lower = self.compute_minimum()
        upper = self.compute_maximum()
        result = {name: upper[name] - lower[name] for name in self.names}
        return result
