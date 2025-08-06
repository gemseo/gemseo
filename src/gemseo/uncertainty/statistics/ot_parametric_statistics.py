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
"""OpenTURNS-based parametric estimation of statistics from a dataset."""

from __future__ import annotations

from typing import ClassVar

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.distribution_fitter import (
    OTDistributionFitter,
)
from gemseo.uncertainty.statistics.base_parametric_statistics import (
    BaseParametricStatistics,
)


class OTParametricStatistics(
    BaseParametricStatistics[
        OTDistribution,
        OTDistributionFitter.default_fitting_criterion,
        OTDistributionFitter.DistributionName,
        OTDistributionFitter.FittingCriterion,
        OTDistributionFitter.SignificanceTest,
    ]
):
    """A toolbox to compute statistics using OpenTURNS probability distribution-fitting.

    Examples:
        >>> from gemseo import (
        ...     create_discipline,
        ...     create_parameter_space,
        ...     sample_disciplines,
        ... )
        >>> from gemseo.uncertainty.statistics.parametric_statistics import (
        ...     OTParametricStatistics,
        ... )
        >>>
        >>> discipline = create_discipline(
        ...     "AnalyticDiscipline", {"y1": "x1+2*x2", "y2": "x1-3*x2"}
        ... )
        >>> parameter_space = create_parameter_space()
        >>> parameter_space.add_random_variable(
        ...     "x1", "OTUniformDistribution", minimum=-1, maximum=1
        ... )
        >>> parameter_space.add_random_variable(
        ...     "x2", "OTNormalDistribution", mu=0.5, sigma=2
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
        >>> statistics = OTParametricStatistics(
        ...     dataset, ["Normal", "Uniform", "Triangular"]
        ... )
        >>> fitting_matrix = statistics.get_fitting_matrix()
        >>> mean = statistics.compute_mean()
    """

    DistributionName: ClassVar[OTDistributionFitter.DistributionName] = (
        OTDistributionFitter.DistributionName
    )
    FittingCriterion: ClassVar[OTDistributionFitter.FittingCriterion] = (
        OTDistributionFitter.FittingCriterion
    )
    SignificanceTest: ClassVar[OTDistributionFitter.SignificanceTest] = (
        OTDistributionFitter.SignificanceTest
    )
    _DISTRIBUTION_FITTER: ClassVar[OTDistributionFitter] = OTDistributionFitter
