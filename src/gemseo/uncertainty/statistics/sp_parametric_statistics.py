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
"""SciPy-based parametric estimation of statistics from a dataset."""

from __future__ import annotations

from typing import ClassVar

from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.distribution_fitter import (
    SPDistributionFitter,
)
from gemseo.uncertainty.statistics.base_parametric_statistics import (
    BaseParametricStatistics,
)


class SPParametricStatistics(
    BaseParametricStatistics[
        SPDistribution,
        SPDistributionFitter.default_fitting_criterion,
        SPDistributionFitter.DistributionName,
        SPDistributionFitter.FittingCriterion,
        SPDistributionFitter.SignificanceTest,
    ]
):
    """A toolbox to compute statistics using SciPy probability distribution-fitting.

    Examples:
        >>> from gemseo import (
        ...     create_discipline,
        ...     create_parameter_space,
        ...     sample_disciplines,
        ... )
        >>> from gemseo.uncertainty.statistics.sp_parametric_statistics import (
        ...     SPParametricStatistics,
        ... )
        >>>
        >>> discipline = create_discipline(
        ...     "AnalyticDiscipline", {"y1": "x1+2*x2", "y2": "x1-3*x2"}
        ... )
        >>>
        >>> parameter_space = create_parameter_space()
        >>> parameter_space.add_random_variable(
        ...     "x1", "SPUniformDistribution", minimum=-1, maximum=1
        ... )
        >>> parameter_space.add_random_variable(
        ...     "x2", "SPNormalDistribution", mu=0.5, sigma=2
        ... )
        >>>
        >>> dataset = sample_disciplines(
        ...     [discipline],
        ...     parameter_space,
        ...     ["y1"],
        ...     algo_name="OT_MONTE_CARLO",
        ...     n_samples=100,
        ... )
        >>>
        >>> statistics = SPParametricStatistics(dataset, ["norm", "uniform", "triang"])
        >>> fitting_matrix = statistics.get_fitting_matrix()
        >>> mean = statistics.compute_mean()
    """

    DistributionName: ClassVar[SPDistributionFitter.DistributionName] = (
        SPDistributionFitter.DistributionName
    )
    FittingCriterion: ClassVar[SPDistributionFitter.FittingCriterion] = (
        SPDistributionFitter.FittingCriterion
    )
    SignificanceTest: ClassVar[SPDistributionFitter.SignificanceTest] = (
        SPDistributionFitter.SignificanceTest
    )
    _DISTRIBUTION_FITTER: ClassVar[SPDistributionFitter] = SPDistributionFitter
