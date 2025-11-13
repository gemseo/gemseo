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
"""Settings of the probability distributions."""

from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.beta_settings import (
    OTBetaDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.dirac_settings import (
    OTDiracDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.exponential_settings import (
    OTExponentialDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.log_normal_settings import (
    OTLogNormalDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.triangular_settings import (
    OTTriangularDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.weibull_settings import (
    OTWeibullDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.beta_settings import (
    SPBetaDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.distribution_settings import (
    SPDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.exponential_settings import (
    SPExponentialDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.log_normal_settings import (
    SPLogNormalDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.normal_settings import (
    SPNormalDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.triangular_settings import (
    SPTriangularDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.uniform_settings import (
    SPUniformDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.weibull_settings import (
    SPWeibullDistribution_Settings,
)

__all__ = [
    "OTBetaDistribution_Settings",
    "OTDiracDistribution_Settings",
    "OTDistribution_Settings",
    "OTExponentialDistribution_Settings",
    "OTLogNormalDistribution_Settings",
    "OTNormalDistribution_Settings",
    "OTTriangularDistribution_Settings",
    "OTUniformDistribution_Settings",
    "OTWeibullDistribution_Settings",
    "SPBetaDistribution_Settings",
    "SPDistribution_Settings",
    "SPExponentialDistribution_Settings",
    "SPLogNormalDistribution_Settings",
    "SPNormalDistribution_Settings",
    "SPTriangularDistribution_Settings",
    "SPUniformDistribution_Settings",
    "SPWeibullDistribution_Settings",
]
