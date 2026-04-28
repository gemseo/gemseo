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
r"""The base class for probability distributions.

The base class
[BaseDistribution][gemseo.uncertainty.distributions.base.BaseDistribution]
implements the concept of
[probability distribution](https://en.wikipedia.org/wiki/Probability_distribution),
which is a mathematical function giving the probabilities of occurrence
of different possible outcomes of a random variable for an experiment.
The [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)
with its famous *bell curve* is a well-known example of probability distribution.

See Also:
    This base class is enriched by concrete ones,
    such as
    [OTDistribution][gemseo.uncertainty.distributions.openturns.distribution.OTDistribution]
    interfacing the OpenTURNS probability distributions
    and
    [SPDistribution][gemseo.uncertainty.distributions.scipy.distribution.SPDistribution]
    interfacing the SciPy probability distributions.

The
[BaseDistribution][gemseo.uncertainty.distributions.base.BaseDistribution]
of a given uncertain variable is built
from a distribution name (e.g. `'Normal'` for OpenTURNS or `'norm'` for SciPy),
a set of parameters
and optionally a standard representation of these parameters.

From a
[BaseDistribution][gemseo.uncertainty.distributions.base.BaseDistribution],
we can easily get statistics, such as
[mean][gemseo.uncertainty.distributions.base.BaseDistribution.mean]
and
[standard_deviation][gemseo.uncertainty.distributions.base.BaseDistribution.standard_deviation]
We can also get the numerical
[range][gemseo.uncertainty.distributions.base.BaseDistribution.range]
and the mathematical
[support][gemseo.uncertainty.distributions.base.BaseDistribution.support].

Note:
    We call mathematical *support* the set of values that the random variable
    can take in theory, e.g. $]-\infty,+\infty[$ for a Gaussian variable,
    and numerical *range* the set of values that it can take in practice,
    taking into account the values rounded to zero double precision.
    Both support and range are described in terms of lower and upper bounds

We can also evaluate the cumulative distribution function
([compute_cdf()][gemseo.uncertainty.distributions.base.BaseDistribution.compute_cdf])
for the different marginals of the random variable,
as well as the inverse cumulative density function
([compute_inverse_cdf()][gemseo.uncertainty.distributions.base.BaseDistribution.compute_inverse_cdf]).

Lastly, we can compute realizations of the random variable
using the
[compute_samples()][gemseo.uncertainty.distributions.base.BaseDistribution.compute_samples])
method.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import Generic
from typing import TypeVar

from numpy import array

from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from gemseo.typing import RealArray
    from gemseo.uncertainty.distributions.base_settings import BaseDistributionSettings

_DistributionT = TypeVar("_DistributionT")
_ParametersT = TypeVar("_ParametersT")
_VariableT = TypeVar("_VariableT")


class BaseDistribution(
    Generic[_VariableT, _ParametersT, _DistributionT],
    metaclass=ABCGoogleDocstringInheritanceMeta,
):
    """The base class for probability distributions.

    By default,
    this base class models the probability distribution of a scalar random variable.
    Child classes need to be adapted to model other types of random variables,
    e.g. random vectors
    (see
    [BaseJointDistribution][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution]
    ).
    """

    settings_class: ClassVar[type[BaseDistributionSettings]]
    """The Pydantic model for the settings."""

    distribution: _DistributionT
    """The probability distribution of the random variable."""

    math_lower_bound: _VariableT
    """The mathematical lower bound of the random variable."""

    math_upper_bound: _VariableT
    """The mathematical upper bound of the random variable."""

    num_lower_bound: _VariableT
    """The numerical lower bound of the random variable."""

    num_upper_bound: _VariableT
    """The numerical upper bound of the random variable."""

    _transformation: str
    """The transformation applied to the random variable noted `"x"`.

    E.g. `"sin(x)"`.
    """

    _ALPHA: Final[str] = "alpha"
    _BETA: Final[str] = "beta"
    _LOC: Final[str] = "loc"
    _LOWER: Final[str] = "lower"
    _MODE: Final[str] = "mode"
    _MU: Final[str] = "mu"
    _RATE: Final[str] = "rate"
    _SCALE: Final[str] = "scale"
    _SHAPE: Final[str] = "shape"
    _LOCATION: Final[str] = "location"
    _SIGMA: Final[str] = "sigma"
    _UPPER: Final[str] = "upper"

    DEFAULT_VARIABLE_NAME: Final[str] = "x"
    """The default name of the variable."""

    _file_path_manager: FilePathManager
    """A manager of file paths for a given type of file and with default settings."""

    _settings: BaseDistributionSettings
    """The settings of the probability distribution."""

    _WEBSITE: ClassVar[str]
    """The website of the library implementing the probability distributions."""

    """The settings of the probability distribution used to create it."""

    def __init__(self, settings: BaseDistributionSettings | None = None) -> None:
        """
        Args:
            settings: The settings of the probability distribution.
                If `None`, use the default ones.
        """  # noqa: D205,D212,D415
        if settings is None:
            settings = self.settings_class()

        self._settings = settings
        self._transformation = self.DEFAULT_VARIABLE_NAME
        self._file_path_manager = FilePathManager(
            FilePathManager.FileType.FIGURE,
            default_name="distribution",
        )
        self._create_distribution(settings)

    @abstractmethod
    def _create_distribution(self, settings: BaseDistributionSettings) -> None:
        """Create the probability distribution.

        Args:
            settings: settings of the probability distribution.
        """

    @abstractmethod
    def compute_samples(
        self,
        n_samples: int = 1,
    ) -> RealArray:
        """Sample the random variable.

        Args:
            n_samples: The number of samples.

        Returns:
            The samples of the random variable.
        """

    @abstractmethod
    def compute_cdf(
        self,
        value: _VariableT,
    ) -> _VariableT:
        """Evaluate the cumulative density function (CDF).

        Args:
            value: The value of the random variable for which to evaluate the CDF.

        Returns:
            The value of the CDF.
        """

    @abstractmethod
    def compute_inverse_cdf(
        self,
        value: _VariableT,
    ) -> _VariableT:
        """Evaluate the inverse cumulative density function (ICDF).

        Args:
            value: The probability for which to evaluate the ICDF.

        Returns:
            The value of the ICDF.
        """

    @property
    @abstractmethod
    def mean(self) -> _VariableT:
        """The expectation of the random variable."""

    @property
    @abstractmethod
    def standard_deviation(self) -> _VariableT:
        """The standard deviation of the random variable."""

    @property
    def range(self) -> RealArray:  # noqa: A003
        """The numerical range.

        The numerical range is the interval defined by the lower and upper bounds
        numerically reachable by the random variable.
        """
        return array([self.num_lower_bound, self.num_upper_bound])

    @property
    def support(self) -> RealArray:
        """The mathematical support.

        The mathematical support is the interval defined by the theoretical lower and
        upper bounds of the random variable.
        """
        return array([self.math_lower_bound, self.math_upper_bound])
