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

The base class :class:`.BaseDistribution` implements the concept of
`probability distribution <https://en.wikipedia.org/wiki/Probability_distribution>`_,
which is a mathematical function giving the probabilities of occurrence
of different possible outcomes of a random variable for an experiment.
The `normal distribution <https://en.wikipedia.org/wiki/Normal_distribution>`_
with its famous *bell curve* is a well-known example of probability distribution.

.. seealso::

    This base class is enriched by concrete ones,
    such as :class:`.OTDistribution` interfacing the OpenTURNS probability distributions
    and :class:`.SPDistribution` interfacing the SciPy probability distributions.

The :class:`.BaseDistribution` of a given uncertain variable is built
from a distribution name (e.g. ``'Normal'`` for OpenTURNS or ``'norm'`` for SciPy),
a set of parameters
and optionally a standard representation of these parameters.

From a :class:`.BaseDistribution`,
we can easily get statistics, such as :attr:`.mean` and :attr:`.standard_deviation`.
We can also get the numerical :attr:`.range` and the mathematical :attr:`.support`.

.. note::

    We call mathematical *support* the set of values that the random variable
    can take in theory, e.g. :math:`]-\infty,+\infty[` for a Gaussian variable,
    and numerical *range* the set of values that it can take in practice,
    taking into account the values rounded to zero double precision.
    Both support and range are described in terms of lower and upper bounds

We can also evaluate the cumulative distribution function
(:meth:`.BaseDistribution.compute_cdf`)
for the different marginals of the random variable,
as well as the inverse cumulative density function (:meth:`.compute_inverse_cdf`).
We can also plot them (:meth:`.plot`).

Lastly, we can compute realizations of the random variable
using the :meth:`.compute_samples` method.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Generic
from typing import TypeVar
from typing import Union

from numpy import array

from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from types import ModuleType

    from gemseo.typing import RealArray

StandardParametersType = Mapping[str, Union[str, int, float]]
ParametersType = Union[tuple[str, int, float], StandardParametersType]

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
    e.g. random vectors (see :class:`.BaseJointDistribution`).
    """

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

    transformation: str
    """The transformation applied to the random variable noted ``"x"``.

    E.g. ``"sin(x)"``.
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

    _string_representation: str
    """The string representation of the distribution."""

    _WEBSITE: ClassVar[str]
    """The website of the library implementing the probability distributions."""

    def __init__(
        self,
        interfaced_distribution: str,
        parameters: _ParametersT,
        standard_parameters: StandardParametersType = READ_ONLY_EMPTY_DICT,
        **options: Any,
    ) -> None:
        """
        Args:
            interfaced_distribution: The name of the probability distribution,
                typically the name of a class wrapped from an external library,
                such as ``"Normal"`` for OpenTURNS or ``"norm"`` for SciPy.
            parameters: The parameters of the probability distribution.
            standard_parameters: The parameters of the probability distribution
                used for string representation only
                (use ``parameters`` for computation).
                If empty, use ``parameters`` instead.
                For instance,
                let us consider an interfaced distribution
                named ``"Dirac"``
                with positional parameters
                (this is the case of :class:`.OTDistribution`).
                Then,
                the string representation of
                ``BaseDistribution("x", "Dirac", (1,), 1, {"loc": 1})``
                is ``"Dirac(loc=1)"``
                while the string representation of
                ``BaseDistribution("x", "Dirac", (1,))``
                is ``"Dirac(1)"``.
                The same mechanism works for keyword parameters
                (this is the case of :class:`.SPDistribution`).
            **options: The options of the probability distribution.
        """  # noqa: D205,D212,D415
        self.transformation = self.DEFAULT_VARIABLE_NAME
        self._file_path_manager = FilePathManager(
            FilePathManager.FileType.FIGURE,
            default_name="distribution",
        )
        self._get_string_representation = (
            f"{interfaced_distribution}"
            f"({pretty_str(standard_parameters or parameters, sort=False)})"
        )
        self._create_distribution(interfaced_distribution, parameters, **options)

    @abstractmethod
    def _create_distribution(
        self,
        distribution_name: str,
        parameters: _ParametersT,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Create the probability distribution.

        Args:
            distribution_name: The name of the probability distribution
                in the interfaced library.
            parameters: The parameters of the probability distribution
                in the interfaced library.
            *args: The options of the probability distribution as positional arguments.
            **kwargs: The options of the probability distribution as keyword arguments.
        """

    def _create_distribution_from_module(
        self,
        module: ModuleType,
        distribution_name: str,
        parameters: _ParametersT,
    ) -> Any:
        """Create a distribution from a module.

        Args:
            module: The module.
            distribution_name: The name of the distribution.
            parameters: The parameters of the distributions.

        Returns:
            The distribution.

        Raises:
            ValueError: When the distribution cannot be created from the module.
        """
        try:
            create_distribution = getattr(module, distribution_name)
        except BaseException:  # noqa: BLE001
            msg = f"{distribution_name} cannot be imported from {module.__name__}."
            raise ImportError(msg) from None

        try:
            if isinstance(parameters, Mapping):
                return create_distribution(**parameters)

            return create_distribution(*parameters)
        except BaseException:  # noqa: BLE001
            msg = (
                f"The arguments of {self._get_string_representation} are wrong; "
                f"more details on {self._WEBSITE}."
            )
            raise ValueError(msg) from None

    def __repr__(self) -> str:
        return self._get_string_representation

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
