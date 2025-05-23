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
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Abstract class for the estimation of statistics from a dataset.

Overview
--------

The abstract :class:`.BaseStatistics` class implements t
he concept of statistics library.
It is enriched by the :class:`.EmpiricalStatistics` and :class:`.ParametricStatistics`.

Construction
------------

A :class:`.BaseStatistics` object is built from a :class:`.Dataset`
and optionally variables names.
In this case,
statistics are only computed for these variables.
Otherwise,
statistics are computed for all the variable available in the dataset.
Lastly,
the user can give a name to its :class:`.BaseStatistics` object.
By default,
this name is the concatenation of the name
of the class overloading :class:`.BaseStatistics`
and the name of the :class:`.Dataset`.

Capabilities
------------

A :class:`.BaseStatistics` returns standard descriptive and statistical measures
for the different variables:

- :meth:`.BaseStatistics.compute_minimum`: the minimum value,
- :meth:`.BaseStatistics.compute_maximum`: the maximum value,
- :meth:`.BaseStatistics.compute_range`: the difference
  between minimum and maximum values,
- :meth:`.BaseStatistics.compute_mean`: the expectation (a.k.a. mean value),
- :meth:`.BaseStatistics.compute_moment`: a central moment,
  which is the expected value
  of a specified integer power
  of the deviation from the mean,
- :meth:`.BaseStatistics.compute_variance`: the variance,
  which is the mean squared variation around the mean value,
- :meth:`.BaseStatistics.compute_standard_deviation`: the standard deviation,
  which is the square root of the variance,
- :meth:`.BaseStatistics.compute_variation_coefficient`: the coefficient of variation,
  which is the standard deviation normalized by the mean,
- :meth:`.BaseStatistics.compute_quantile`: the quantile associated with a probability,
  which is the cut point diving the range into a first continuous interval
  with this given probability and a second continuous interval
  with the complementary probability; common *q*-quantiles dividing
  the range into *q* continuous interval with equal probabilities are also implemented:

    - :meth:`.BaseStatistics.compute_median`
      which implements the 2-quantile (50%).
    - :meth:`.BaseStatistics.compute_quartile`
      whose order (1, 2 or 3) implements the 4-quantiles (25%, 50% and 75%),
    - :meth:`.BaseStatistics.compute_percentile`
      whose order (1, 2, ..., 99) implements the 100-quantiles (1%, 2%, ..., 99%),

- :meth:`.BaseStatistics.compute_probability`:
  the probability that the random variable is larger or smaller
  than a certain threshold,
- :meth:`.BaseStatistics.compute_tolerance_interval`:
  the left-sided, right-sided or both-sided tolerance interval
  associated with a given coverage level and a given confidence level,
  which is a statistical interval within which,
  with some confidence level,
  a specified proportion of the random variable realizations falls
  (this proportion is the coverage level)

    - :meth:`.BaseStatistics.compute_a_value`:
      the A-value, which is the lower bound of the left-sided tolerance interval
      associated with a coverage level equal to 99% and a confidence level equal to 95%,
    - :meth:`.BaseStatistics.compute_b_value`:
      the B-value, which is the lower bound of the left-sided tolerance interval
      associated with a coverage level equal to 90% and a confidence level equal to 95%,
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from numpy import array

from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    BaseToleranceInterval,
)
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.datasets.dataset import Dataset
    from gemseo.typing import RealArray


class BaseStatistics(metaclass=ABCGoogleDocstringInheritanceMeta):
    """A toolbox to compute statistics.

    Note:
        Unless otherwise stated,
        the statistics are computed
        *variable-wise* and *component-wise*,
        i.e. variable-by-variable and component-by-component.
        So, for the sake of readability,
        the methods named as :meth:`compute_statistic`
        return ``dict[str, RealArray]`` objects
        whose values are the names of the variables
        and the values are the statistic estimated for the different component.
    """

    dataset: Dataset
    """The dataset."""

    n_samples: int
    """The number of samples."""

    n_variables: int
    """The number of variables."""

    name: str
    """The name of the object."""

    SYMBOLS: ClassVar[dict[str, str]] = {}

    __QUARTILE_LEVELS: Final[list[float]] = [0.25, 0.5, 0.75]
    __QUARTILE_ORDERS: Final[list[int]] = [1, 2, 3]

    def __init__(
        self,
        dataset: Dataset,
        variable_names: Iterable[str] = (),
        name: str = "",
    ) -> None:
        """
        Args:
            dataset: A dataset.
            variable_names: The names of the variables for which to compute statistics.
                If empty, consider all the variables of the dataset.
            name: A name for the toolbox computing statistics.
                If empty,
                concatenate the names of the dataset and the name of the class.
        """  # noqa: D205,D212,D415
        class_name = self.__class__.__name__
        self.name = name or f"{class_name}({dataset.name})"
        self.dataset = dataset
        self.n_samples = len(dataset)
        self.names = variable_names or dataset.variable_names
        self.n_variables = len(self.names)

    @property
    def __string_representation(self) -> MultiLineString:
        """The string representation of the object."""
        mls = MultiLineString()
        mls.add(self.name)
        mls.indent()
        mls.add("n_samples: {}", self.n_samples)
        mls.add("n_variables: {}", self.n_variables)
        mls.add("variables: {}", pretty_str(self.names))
        return mls

    def __repr__(self) -> str:
        return str(self.__string_representation)

    def _repr_html_(self) -> str:
        return self.__string_representation._repr_html_()

    @abstractmethod
    def compute_tolerance_interval(
        self,
        coverage: float,
        confidence: float = 0.95,
        side: BaseToleranceInterval.ToleranceIntervalSide = BaseToleranceInterval.ToleranceIntervalSide.BOTH,  # noqa:E501
    ) -> dict[str, list[BaseToleranceInterval.Bounds]]:  # noqa: D102
        r"""Compute a :math:`(p,1-\alpha)` tolerance interval :math:`\text{TI}[X]`.

        The tolerance interval :math:`\text{TI}[X]` is defined
        to contain at least a proportion :math:`p` of the values of :math:`X`
        with a level of confidence :math:`1-\alpha`.
        :math:`p` is also called the *coverage level* of the TI.

        Typically, :math:`\alpha=0.05` or equivalently :math:`1-\alpha=0.95`.

        The tolerance interval can be either

        - lower-sided (``side="LOWER"``: :math:`[L, +\infty[`),
        - upper-sided  (``side="UPPER"``: :math:`]-\infty, U]`) or
        - both-sided (``side="BOTH"``: :math:`[L, U]`).

        Args:
            coverage: A minimum proportion :math:`p\in[0,1]` of belonging to the TI.
            confidence: A level of confidence :math:`1-\alpha\in[0,1]`.
            side: The type of the tolerance interval.

        Returns:
            The component-wise tolerance intervals of the different variables,
            expressed as
            ``{variable_name: [(lower_bound, upper_bound), ...], ... }``
            where ``[(lower_bound, upper_bound), ...]``
            are the lower and upper bounds of the tolerance interval
            of the different components of ``variable_name``.

        See Also:
            :meth:`.compute_a_value`
            :meth:`.compute_b_value`
        """

    SYMBOLS["tolerance_interval"] = "TI"

    def compute_a_value(self) -> dict[str, RealArray]:
        r"""Compute the A-value :math:`\text{Aval}[X]`.

        The A-value is the lower bound of the left-sided tolerance interval
        associated with a coverage level equal to 99%
        and a confidence level equal to 95%.

        Returns:
            The component-wise A-value of the different variables.

        See Also:
            :meth:`.compute_tolerance_interval`
            :meth:`.compute_b_value`
        """
        return {
            name: array([t_i.lower for t_i in tolerance_intervals])
            for name, tolerance_intervals in self.compute_tolerance_interval(
                0.99, side=BaseToleranceInterval.ToleranceIntervalSide.LOWER
            ).items()
        }

    SYMBOLS["a_value"] = "Aval"

    def compute_b_value(self) -> dict[str, RealArray]:
        r"""Compute the B-value :math:`\text{Bval}[X]`.

        The B-value is the lower bound of the left-sided tolerance interval
        associated with a coverage level equal to 90%
        and a confidence level equal to 95%.

        Returns:
            The component-wise B-value of the different variables.

        See Also:
            :meth:`.compute_tolerance_interval`
            :meth:`.compute_a_value`
        """
        return {
            name: array([t_i.lower for t_i in tolerance_intervals])
            for name, tolerance_intervals in self.compute_tolerance_interval(
                0.9, side=BaseToleranceInterval.ToleranceIntervalSide.LOWER
            ).items()
        }

    SYMBOLS["b_value"] = "Bval"

    @abstractmethod
    def compute_maximum(self) -> dict[str, RealArray]:
        r"""Compute the maximum :math:`\text{Max}[X]`.

        Returns:
            The component-wise maximum of the different variables.
        """

    SYMBOLS["maximum"] = "Max"

    @abstractmethod
    def compute_mean(self) -> dict[str, RealArray]:
        r"""Compute the mean :math:`\mathbb{E}[X]`.

        Returns:
            The component-wise mean of the different variables.
        """

    SYMBOLS["mean"] = "E"

    def compute_margin(self, std_factor: float) -> dict[str, RealArray]:
        r"""Compute a margin :math:`\text{Margin}[X]=\mathbb{E}[X]+\kappa\mathbb{S}[X]`.

        Args:
            std_factor: The weight :math:`\kappa` of the standard deviation.

        Returns:
            The component-wise margin for the different variables.
        """
        mean = self.compute_mean()
        return {
            name: mean[name] + std_factor * standard_deviation
            for name, standard_deviation in self.compute_standard_deviation().items()
        }

    compute_mean_std = compute_margin

    SYMBOLS["mean_std"] = "E_StD"
    SYMBOLS["margin"] = "Margin"

    @abstractmethod
    def compute_minimum(self) -> dict[str, RealArray]:
        r"""Compute the :math:`\text{Min}[X]`.

        Returns:
            The component-wise minimum of the different variables.
        """

    SYMBOLS["minimum"] = "Min"

    def compute_median(self) -> dict[str, RealArray]:
        r"""Compute the median :math:`\text{Med}[X]`.

        Returns:
            The component-wise median of the different variables.
        """
        return self.compute_quantile(0.5)

    SYMBOLS["median"] = "Med"

    def compute_percentile(self, order: int) -> dict[str, RealArray]:
        r"""Compute the n-th percentile :math:`\text{p}[X; n]`.

        Args:
            order: The order :math:`n\in\{0,1,2,...100\}` of the percentile.

        Returns:
            The component-wise percentile of the different variables.

        Raises:
            ValueError: When :math:`n\notin\{0,1,2,...100\}`.
        """
        if not isinstance(order, int) or order > 100 or order < 0:
            msg = "Percentile order must be in {0, 1, 2, ..., 100}."
            raise TypeError(msg)
        return self.compute_quantile(order / 100.0)

    SYMBOLS["percentile"] = "p"

    @abstractmethod
    def compute_probability(
        self, thresh: Mapping[str, float | RealArray], greater: bool = True
    ) -> dict[str, RealArray]:
        r"""Compute the probability related to a threshold.

        Either :math:`\mathbb{P}[X \geq x]` or :math:`\mathbb{P}[X \leq x]`.

        Args:
            thresh: A threshold :math:`x` per variable.
            greater: The type of probability.
                If ``True``,
                compute the probability of exceeding the threshold.
                Otherwise,
                compute the opposite.

        Returns:
            The component-wise probability of the different variables.
        """

    @abstractmethod
    def compute_joint_probability(
        self, thresh: Mapping[str, float | RealArray], greater: bool = True
    ) -> dict[str, float]:
        r"""Compute the joint probability related to a threshold.

        Either :math:`\mathbb{P}[X \geq x]` or :math:`\mathbb{P}[X \leq x]`.

        Args:
            thresh: A threshold :math:`x` per variable.
            greater: The type of probability.
                If ``True``,
                compute the probability of exceeding the threshold.
                Otherwise,
                compute the opposite.

        Returns:
            The joint probability of the different variables
            (by definition of the joint probability,
            this statistics is not computed component-wise).
        """

    SYMBOLS["probability"] = "P"

    @abstractmethod
    def compute_quantile(self, prob: float) -> dict[str, RealArray]:
        r"""Compute the quantile :math:`\mathbb{Q}[X; \alpha]` related to a probability.

        Args:
            prob: A probability :math:`\alpha` between 0 and 1.

        Returns:
            The component-wise quantile of the different variables.
        """

    SYMBOLS["quantile"] = "Q"

    def compute_quartile(self, order: int) -> dict[str, RealArray]:
        r"""Compute the n-th quartile :math:`q[X; n]`.

        Args:
            order: The order :math:`n\in\{1,2,3\}` of the quartile.

        Returns:
            The component-wise quartile of the different variables.

        Raises:
            ValueError: When :math:`n\notin\{1,2,3\}`.
        """
        if order not in self.__QUARTILE_ORDERS:
            msg = "Quartile order must be in {1, 2, 3}."
            raise ValueError(msg)

        return self.compute_quantile(self.__QUARTILE_LEVELS[order - 1])

    SYMBOLS["quartile"] = "q"

    @abstractmethod
    def compute_range(self) -> dict[str, RealArray]:
        r"""Compute the range :math:`R[X]`.

        Returns:
            The component-wise range of the different variables.
        """

    SYMBOLS["range"] = "R"

    @abstractmethod
    def compute_standard_deviation(self) -> dict[str, RealArray]:
        r"""Compute the standard deviation :math:`\mathbb{S}[X]`.

        Returns:
            The component-wise standard deviation of the different variables.
        """

    SYMBOLS["standard_deviation"] = "StD"

    def compute_variation_coefficient(self) -> dict[str, RealArray]:
        r"""Compute the coefficient of variation :math:`CoV[X]`.

        This is the standard deviation normalized by the expectation:
        :math:`CoV[X]=\mathbb{E}[S]/\mathbb{E}[X]`.

        Returns:
            The component-wise coefficient of variation of the different variables.
        """
        mean = self.compute_mean()
        standard_deviation = self.compute_standard_deviation()
        return {name: standard_deviation[name] / mean[name] for name in mean}

    SYMBOLS["variation_coefficient"] = "CoV"

    @abstractmethod
    def compute_variance(self) -> dict[str, RealArray]:
        r"""Compute the variance :math:`\mathbb{V}[X]`.

        Returns:
            The component-wise variance of the different variables.
        """

    SYMBOLS["variance"] = "V"

    @abstractmethod
    def compute_moment(self, order: int) -> dict[str, RealArray]:
        r"""Compute the n-th moment :math:`M[X; n]`.

        Args:
            order: The order :math:`n` of the moment.

        Returns:
            The component-wise moment of the different variables.
        """

    SYMBOLS["moment"] = "M"

    @classmethod
    def compute_expression(
        cls,
        variable_name: str,
        statistic_name: str,
        show_name: bool = False,
        **options: bool | float,
    ) -> str:
        """Return the expression of a statistical function applied to a variable.

        E.g. "P[X >= 1.0]" for the probability that X exceeds 1.0.

        Args:
            variable_name: The name of the variable, e.g. ``"X"``.
            statistic_name: The name of the statistic, e.g. ``"probability"``.
            show_name: If ``True``, show option names.
                Otherwise, only show option values.
            **options: The options passed to the statistical function,
                e.g. ``{"greater": True, "thresh": 1.0}``.

        Returns:
            The expression of the statistical function applied to the variable.
        """
        if "greater" in options:
            separator = " >= " if options["greater"] else " <= "
            options.pop("greater")
        elif statistic_name == "probability":
            separator = " >= "
        else:
            separator = ""

        values = options if show_name else options.values()
        value = pretty_str(values)

        if value and not separator:
            separator = "; "

        return (
            f"{cls.SYMBOLS.get(statistic_name, statistic_name)}"
            f"[{variable_name}{separator}{value}]"
        )
