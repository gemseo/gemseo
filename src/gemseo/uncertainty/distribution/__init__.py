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
"""Capabilities to create and manipulate probability distributions.

This package contains:

- an abstract class
  [BaseDistribution][gemseo.uncertainty.distribution.core.base.BaseDistribution]
  to define the concept of probability distribution,
- an abstract class
  [BaseJointDistribution][gemseo.uncertainty.distribution.core.base_joint.BaseJointDistribution]
  to define the concept of joint probability distribution
  by composing several instances of
  [BaseDistribution][gemseo.uncertainty.distribution.core.base.BaseDistribution],
- a factory
  [DistributionFactory][gemseo.uncertainty.distribution.factory.DistributionFactory]
  to create instances of
  [BaseDistribution][gemseo.uncertainty.distribution.core.base.BaseDistribution],
- concrete classes implementing these abstracts concepts, by interfacing:

  - the OpenTURNS library:
    [OTDistribution][gemseo.uncertainty.distribution.openturns.distribution.OTDistribution]
    and
    [OTJointDistribution][gemseo.uncertainty.distribution.openturns.joint.OTJointDistribution],
  - the Scipy library:
    [SPDistribution][gemseo.uncertainty.distribution.scipy.distribution.SPDistribution]
    and
    [SPJointDistribution][gemseo.uncertainty.distribution.scipy.joint.SPJointDistribution].

Lastly,
the class
[OTDistributionFitter][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter]
offers the possibility
to fit an
[OTDistribution][gemseo.uncertainty.distribution.openturns.distribution.OTDistribution]
from data based on OpenTURNS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.uncertainty.distribution.factory import DISTRIBUTION_FACTORY  # noqa: F401
    from gemseo.uncertainty.distribution.openturns.beta import OTBetaDistribution  # noqa: F401
    from gemseo.uncertainty.distribution.openturns.beta_settings import (
        OTBetaDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.openturns.dirac import OTDiracDistribution  # noqa: F401
    from gemseo.uncertainty.distribution.openturns.dirac_settings import (
        OTDiracDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.openturns.distribution import OTDistribution  # noqa: F401
    from gemseo.uncertainty.distribution.openturns.distribution_fitter import (
        OTDistributionFitter,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.openturns.distribution_settings import (
        OTDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.openturns.exponential import (
        OTExponentialDistribution,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.openturns.exponential_settings import (
        OTExponentialDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.openturns.finite_discrete import (
        OTFiniteDiscreteDistribution,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.openturns.finite_discrete_settings import (
        OTFiniteDiscreteDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.openturns.joint import OTJointDistribution  # noqa: F401
    from gemseo.uncertainty.distribution.openturns.log_normal import (
        OTLogNormalDistribution,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.openturns.log_normal_settings import (
        OTLogNormalDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.openturns.normal import OTNormalDistribution  # noqa: F401
    from gemseo.uncertainty.distribution.openturns.normal_settings import (
        OTNormalDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.openturns.triangular import (
        OTTriangularDistribution,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.openturns.triangular_settings import (
        OTTriangularDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.openturns.uniform import OTUniformDistribution  # noqa: F401
    from gemseo.uncertainty.distribution.openturns.uniform_settings import (
        OTUniformDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.openturns.weibull import OTWeibullDistribution  # noqa: F401
    from gemseo.uncertainty.distribution.openturns.weibull_settings import (
        OTWeibullDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.scipy.beta import SPBetaDistribution  # noqa: F401
    from gemseo.uncertainty.distribution.scipy.beta_settings import (
        SPBetaDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.scipy.distribution import SPDistribution  # noqa: F401
    from gemseo.uncertainty.distribution.scipy.distribution_fitter import (
        SPDistributionFitter,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.scipy.distribution_settings import (
        SPDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.scipy.exponential import (
        SPExponentialDistribution,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.scipy.exponential_settings import (
        SPExponentialDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.scipy.joint import SPJointDistribution  # noqa: F401
    from gemseo.uncertainty.distribution.scipy.log_normal import SPLogNormalDistribution  # noqa: F401
    from gemseo.uncertainty.distribution.scipy.log_normal_settings import (
        SPLogNormalDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.scipy.normal import SPNormalDistribution  # noqa: F401
    from gemseo.uncertainty.distribution.scipy.normal_settings import (
        SPNormalDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.scipy.triangular import (
        SPTriangularDistribution,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.scipy.triangular_settings import (
        SPTriangularDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.scipy.uniform import SPUniformDistribution  # noqa: F401
    from gemseo.uncertainty.distribution.scipy.uniform_settings import (
        SPUniformDistribution_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.distribution.scipy.weibull import SPWeibullDistribution  # noqa: F401
    from gemseo.uncertainty.distribution.scipy.weibull_settings import (
        SPWeibullDistribution_Settings,  # noqa: F401
    )

# Class name -> defining submodule (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "DISTRIBUTION_FACTORY": "factory",
    "OTBetaDistribution": "openturns.beta",
    "OTBetaDistribution_Settings": "openturns.beta_settings",
    "OTDiracDistribution": "openturns.dirac",
    "OTDiracDistribution_Settings": "openturns.dirac_settings",
    "OTDistribution": "openturns.distribution",
    "OTDistributionFitter": "openturns.distribution_fitter",
    "OTDistribution_Settings": "openturns.distribution_settings",
    "OTExponentialDistribution": "openturns.exponential",
    "OTExponentialDistribution_Settings": "openturns.exponential_settings",
    "OTFiniteDiscreteDistribution": "openturns.finite_discrete",
    "OTFiniteDiscreteDistribution_Settings": "openturns.finite_discrete_settings",
    "OTJointDistribution": "openturns.joint",
    "OTLogNormalDistribution": "openturns.log_normal",
    "OTLogNormalDistribution_Settings": "openturns.log_normal_settings",
    "OTNormalDistribution": "openturns.normal",
    "OTNormalDistribution_Settings": "openturns.normal_settings",
    "OTTriangularDistribution": "openturns.triangular",
    "OTTriangularDistribution_Settings": "openturns.triangular_settings",
    "OTUniformDistribution": "openturns.uniform",
    "OTUniformDistribution_Settings": "openturns.uniform_settings",
    "OTWeibullDistribution": "openturns.weibull",
    "OTWeibullDistribution_Settings": "openturns.weibull_settings",
    "SPBetaDistribution": "scipy.beta",
    "SPBetaDistribution_Settings": "scipy.beta_settings",
    "SPDistribution": "scipy.distribution",
    "SPDistributionFitter": "scipy.distribution_fitter",
    "SPDistribution_Settings": "scipy.distribution_settings",
    "SPExponentialDistribution": "scipy.exponential",
    "SPExponentialDistribution_Settings": "scipy.exponential_settings",
    "SPJointDistribution": "scipy.joint",
    "SPLogNormalDistribution": "scipy.log_normal",
    "SPLogNormalDistribution_Settings": "scipy.log_normal_settings",
    "SPNormalDistribution": "scipy.normal",
    "SPNormalDistribution_Settings": "scipy.normal_settings",
    "SPTriangularDistribution": "scipy.triangular",
    "SPTriangularDistribution_Settings": "scipy.triangular_settings",
    "SPUniformDistribution": "scipy.uniform",
    "SPUniformDistribution_Settings": "scipy.uniform_settings",
    "SPWeibullDistribution": "scipy.weibull",
    "SPWeibullDistribution_Settings": "scipy.weibull_settings",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
