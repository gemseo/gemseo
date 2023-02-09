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
r"""Abstract class defining the concept of probability distribution.

Overview
--------

The abstract :class:`.Distribution` class implements the concept of
`probability distribution <https://en.wikipedia.org/wiki/Probability_distribution>`_,
which is a mathematical function giving the probabilities of occurrence
of different possible outcomes of a random variable for an experiment.
The `normal distribution <https://en.wikipedia.org/wiki/Normal_distribution>`_
with its famous *bell curve* is a well-known example of probability distribution.

.. seealso::

    This abstract class is enriched by concrete ones,
    such as :class:`.OTDistribution` interfacing the OpenTURNS probability distributions
    and :class:`.SPDistribution` interfacing the SciPy probability distributions.

Construction
------------

The :class:`.Distribution` of a given uncertain variable is built
from a recognized distribution name (e.g. 'Normal' for OpenTURNS or 'norm' for SciPy),
a variable dimension, a set of parameters
and optionally a standard representation of these parameters.

Capabilities
------------

From a :class:`.Distribution`, we can easily get statistics,
such as :attr:`.Distribution.mean`,
:attr:`.Distribution.standard_deviation`. We can also get the
numerical :attr:`.Distribution.range` and
mathematical :attr:`.Distribution.support`.

.. note::

    We call mathematical *support* the set of values that the random variable
    can take in theory, e.g. :math:`]-\infty,+\infty[` for a Gaussian variable,
    and numerical *range* the set of values that it can take in practice,
    taking into account the values rounded to zero double precision.
    Both support and range are described in terms of lower and upper bounds

We can also evaluate the cumulative density function
(:meth:`.Distribution.compute_cdf`)
for the different marginals of the random variable,
as well as the inverse cumulative density function
(:meth:`.Distribution.compute_inverse_cdf`). We can plot them,
either for a given marginal (:meth:`.Distribution.plot`)
or for all marginals (:meth:`.Distribution.plot_all`).

Lastly, we can compute realizations of the random variable
by means of the :meth:`.Distribution.compute_samples` method.
"""
from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Mapping
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import arange
from numpy import array
from numpy import ndarray

from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.file_path_manager import FileType
from gemseo.utils.matplotlib_figure import save_show_figure
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

LOGGER = logging.getLogger(__name__)

StandardParametersType = Mapping[str, Union[str, int, float]]
ParametersType = Union[Tuple[str, int, float], StandardParametersType]


class Distribution(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Probability distribution related to a random variable.

    The dimension of the random variable can be greater than 1. In this case,
    the same distribution is applied to all components of the random variable
    under the hypothesis that these components are stochastically independent.

    The string representation of a distribution
    interfacing a distribution called :code:`'MyDistribution'`
    with parameters :code:`(2,3)` is 'MyDistribution(2, 3)`
    if no standard parameters are passed.
    If the standard parameters are :code:`{a: 2, b: 3}`
    (resp. :code:`{a_inv: 2, b: 3}`),
    then the standard representation is: 'MyDistribution(a=2, b=3)`
    (resp. 'MyDistribution(a_inv=0.5, b=3)`)
    Standard parameters are useful to redefine the name of the parameters.
    For example, some exponential distributions consider the notion of rate
    while other ones consider the notion of scale, which is the inverse of the rate...
    even in the background, the distribution is the same!
    """

    math_lower_bound: ndarray
    """The mathematical lower bound of the random variable."""

    math_upper_bound: ndarray
    """The mathematical upper bound of the random variable."""

    num_lower_bound: ndarray
    """The numerical lower bound of the random variable."""

    num_upper_bound: ndarray
    """The numerical upper bound of the random variable."""

    distribution: type
    """The probability distribution of the random variable."""

    marginals: list[type]
    """The marginal distributions of the components of the random variable."""

    dimension: int
    """The number of dimensions of the random variable."""

    variable_name: str
    """The name of the random variable."""

    distribution_name: str
    """The name of the probability distribution."""

    transformation: str
    """The transformation applied to the random variable, e.g. 'sin(x)'."""

    parameters: tuple[Any] | dict[str, Any]
    """The parameters of the probability distribution."""

    standard_parameters: dict[str, str] | None
    """The standard representation of the parameters of the distribution, used for its
    string representation."""

    _MU = "mu"
    _SIGMA = "sigma"
    _LOWER = "lower"
    _UPPER = "upper"
    _MODE = "mode"
    _RATE = "rate"
    _LOC = "loc"

    _COMPOSED_DISTRIBUTION = None

    def __init__(
        self,
        variable: str,
        interfaced_distribution: str,
        parameters: ParametersType,
        dimension: int = 1,
        standard_parameters: StandardParametersType | None = None,
    ) -> None:
        """
        Args:
            variable: The name of the random variable.
            interfaced_distribution: The name of the probability distribution,
                typically the name of a class wrapped from an external library,
                such as 'Normal' for OpenTURNS or 'norm' for SciPy.
            parameters: The parameters of the class
                related to distribution.
            dimension: The dimension of the random variable.
            standard_parameters: The standard representation
                of the parameters of the probability distribution.
        """  # noqa: D205,D212,D415
        self.math_lower_bound = None
        self.math_upper_bound = None
        self.num_lower_bound = None
        self.num_upper_bound = None
        self.distribution = None
        self.marginals = None
        self.dimension = dimension
        self.variable_name = variable
        self.distribution_name = interfaced_distribution
        self.transformation = variable
        self.parameters = parameters
        if standard_parameters is None:
            self.standard_parameters = self.parameters
        else:
            self.standard_parameters = standard_parameters
        self.__file_path_manager = FilePathManager(
            FileType.FIGURE, default_name=f"distribution_{self.variable_name}"
        )
        msg = MultiLineString()
        msg.add("Define the random variable: {}", variable)
        msg.indent()
        msg.add("Distribution: {}", self)
        msg.add("Dimension: {}", dimension)
        LOGGER.debug("%s", msg)

    def __str__(self) -> str:
        return f"{self.distribution_name}({pretty_str(self.standard_parameters)})"

    @abstractmethod
    def compute_samples(
        self,
        n_samples: int = 1,
    ) -> ndarray:
        """Sample the random variable.

        Args:
            n_samples: The number of samples.

        Returns:
            The samples of the random variable,

            The number of columns is equal to the dimension of the variable
            and the number of lines is equal to the number of samples.
        """

    @abstractmethod
    def compute_cdf(
        self,
        vector: Iterable[float],
    ) -> ndarray:
        """Evaluate the cumulative density function (CDF).

        Evaluate the CDF of the components of the random variable
        for a given realization of this random variable.

        Args:
            vector: A realization of the random variable.

        Returns:
            The CDF values of the components of the random variable.
        """

    @abstractmethod
    def compute_inverse_cdf(
        self,
        vector: Iterable[float],
    ) -> ndarray:
        """Evaluate the inverse of the cumulative density function (ICDF).

        Args:
            vector: A vector of values comprised between 0 and 1
                whose length is equal to the dimension of the random variable.

        Returns:
            The ICDF values of the components of the random variable.
        """

    @property
    @abstractmethod
    def mean(self) -> ndarray:
        """The analytical mean of the random variable."""

    @property
    @abstractmethod
    def standard_deviation(self) -> ndarray:
        """The analytical standard deviation of the random variable."""

    @property
    def range(self) -> list[ndarray]:
        """The numerical range.

        The numerical range is the interval defined by
        the lower and upper bounds numerically reachable by the random variable.

        Here, the numerical range of the random variable is defined
        by one array for each component of the random variable,
        whose first element is the lower bound of this component
        while the second one is its upper bound.
        """
        value = [
            array([l_b, u_b])
            for l_b, u_b in zip(self.num_lower_bound, self.num_upper_bound)
        ]
        return value

    @property
    def support(self) -> list[ndarray]:
        """The mathematical support.

        The mathematical support is the interval defined by
        the theoretical lower and upper bounds of the random variable.

        Here, the mathematical range of the random variable is defined
        by one array for each component of the random variable,
        whose first element is the lower bound of this component
        while the second one is its upper bound.
        """
        value = [
            array([l_b, u_b])
            for l_b, u_b in zip(self.math_lower_bound, self.math_upper_bound)
        ]
        return value

    def plot_all(
        self,
        show: bool = True,
        save: bool = False,
        file_path: str | Path | None = None,
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_extension: str | None = None,
    ) -> list[Figure]:
        """Plot both probability and cumulative density functions for all components.

        Args:
            save: If True, save the figure.
            show: If True, display the figure.
            file_path: The path of the file to save the figures.
                If the extension is missing, use ``file_extension``.
                If ``None``,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_extension``.
            directory_path: The path of the directory to save the figures.
                If ``None``, use the current working directory.
            file_name: The name of the file to save the figures.
                If ``None``, use a default one generated by the post-processing.
            file_extension: A file extension, e.g. ``'png'``, ``'pdf'``, ``'svg'``, ...
                If ``None``, use a default file extension.

        Returns:
            The figures.
        """
        figures = []
        for index in range(self.dimension):
            figures.append(
                self.plot(
                    index=index,
                    show=show,
                    save=save,
                    file_path=file_path,
                    file_name=file_name,
                    file_extension=file_extension,
                    directory_path=directory_path,
                )
            )
        return figures

    def plot(
        self,
        index: int = 0,
        show: bool = True,
        save: bool = False,
        file_path: str | Path | None = None,
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_extension: str | None = None,
    ) -> Figure:
        """Plot both probability and cumulative density functions for a given component.

        Args:
            index: The index of a component of the random variable.
            save: If True, save the figure.
            show: If True, display the figure.
            file_path: The path of the file to save the figures.
                If the extension is missing, use ``file_extension``.
                If ``None``,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_extension``.
            directory_path: The path of the directory to save the figures.
                If ``None``, use the current working directory.
            file_name: The name of the file to save the figures.
                If ``None``, use a default one generated by the post-processing.
            file_extension: A file extension, e.g. ``'png'``, ``'pdf'``, ``'svg'``, ...
                If ``None``, use a default file extension.

        Returns:
            The figure.
        """
        variable_name = self.variable_name
        if self.dimension > 1:
            variable_name = f"{variable_name}[{index}]"

        l_b = self.num_lower_bound[index]
        u_b = self.num_upper_bound[index]
        x_values = arange(l_b, u_b, (u_b - l_b) / 100)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.2))
        fig.suptitle(f"Probability distribution of {variable_name}")
        ax1.plot(x_values, [self._pdf(index)(x_value) for x_value in x_values])
        ax1.grid()
        ax1.set_xlabel(variable_name)
        ax1.set_ylabel("Probability density function")
        ax1.set_box_aspect(1)
        ax2.plot(x_values, [self._cdf(index)(x_value) for x_value in x_values])
        ax2.grid()
        ax2.set_xlabel(variable_name)
        ax2.set_ylabel("Cumulative distribution function")
        ax2.yaxis.tick_right()
        ax2.set_box_aspect(1)
        if save:
            file_path = self.__file_path_manager.create_file_path(
                file_path=file_path,
                file_name=file_name,
                directory_path=directory_path,
                file_extension=file_extension,
            )
            if self.dimension > 1:
                file_path = self.__file_path_manager.add_suffix(file_path, str(index))
        else:
            file_path = None

        save_show_figure(fig, show, file_path)
        return fig

    def _pdf(
        self,
        index: int,
    ) -> Callable:
        """Get the probability density function of a marginal.

        Args:
            index: The index of a component of the random variable.

        Return:
            The probability density function
                of the given component of the random variable.
        """

        def pdf(
            point: float,
        ) -> float:
            """Probability Density Function (PDF).

            Args:
                point: An evaluation point.

            Returns:
                The PDF value at the evaluation point.
            """
            raise NotImplementedError

        return pdf

    def _cdf(
        self,
        index: int,
    ) -> Callable:
        """Get the cumulative density function of a marginal.

        Args:
            index: The index of a component of the random variable.

        Return:
            The cumulative density function
                of the given component of the random variable.
        """

        def cdf(
            level: float,
        ) -> float:
            """Cumulative Density Function (CDF).

            Args:
                level: A probability level.

            Returns:
                The CDF value for the probability level.
            """
            raise NotImplementedError

        return cdf
