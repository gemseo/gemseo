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

The abstract :class:`.Statistics` class implements the concept of statistics library.
It is enriched by the :class:`.EmpiricalStatistics` and :class:`.ParametricStatistics`.

Construction
------------

A :class:`.Statistics` object is built from a :class:`.Dataset`
and optionally variables names.
In this case,
statistics are only computed for these variables.
Otherwise,
statistics are computed for all the variable available in the dataset.
Lastly,
the user can give a name to its :class:`.Statistics` object.
By default,
this name is the concatenation of the name of the class overloading :class:`.Statistics`
and the name of the :class:`.Dataset`.

Capabilities
------------

A :class:`.Statistics` returns standard descriptive and statistical measures
for the different variables:

- :meth:`.Statistics.compute_minimum`: the minimum value,
- :meth:`.Statistics.compute_maximum`: the maximum value,
- :meth:`.Statistics.compute_range`: the difference between minimum and maximum values,
- :meth:`.Statistics.compute_mean`: the expectation (a.k.a. mean value),
- :meth:`.Statistics.compute_moment`: a central moment,
  which is the expected value
  of a specified integer power
  of the deviation from the mean,
- :meth:`.Statistics.compute_variance`: the variance,
  which is the mean squared variation around the mean value,
- :meth:`.Statistics.compute_standard_deviation`: the standard deviation,
  which is the square root of the variance,
- :meth:`.Statistics.compute_variation_coefficient`: the coefficient of variation,
  which is the standard deviation normalized by the mean,
- :meth:`.Statistics.compute_quantile`: the quantile associated with a probability,
  which is the cut point diving the range into a first continuous interval
  with this given probability and a second continuous interval
  with the complementary probability; common *q*-quantiles dividing
  the range into *q* continuous interval with equal probabilities are also implemented:

    - :meth:`.Statistics.compute_median`
      which implements the 2-quantile (50%).
    - :meth:`.Statistics.compute_quartile`
      whose order (1, 2 or 3) implements the 4-quantiles (25%, 50% and 75%),
    - :meth:`.Statistics.compute_percentile`
      whose order (1, 2, ..., 99) implements the 100-quantiles (1%, 2%, ..., 99%),

- :meth:`.Statistics.compute_probability`:
  the probability that the random variable is larger or smaller
  than a certain threshold,
- :meth:`.Statistics.compute_tolerance_interval`:
  the left-sided, right-sided or both-sided tolerance interval
  associated with a given coverage level and a given confidence level,
  which is a statistical interval within which,
  with some confidence level,
  a specified proportion of the random variable realizations falls
  (this proportion is the coverage level)

    - :meth:`.Statistics.compute_a_value`:
      the A-value, which is the lower bound of the left-sided tolerance interval
      associated with a coverage level equal to 99% and a confidence level equal to 95%,
    - :meth:`.Statistics.compute_b_value`:
      the B-value, which is the lower bound of the left-sided tolerance interval
      associated with a coverage level equal to 90% and a confidence level equal to 95%,
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Iterable

from docstring_inheritance import GoogleDocstringInheritanceMeta
from numpy import ndarray

from gemseo.core.dataset import Dataset
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceIntervalSide,
)
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

LOGGER = logging.getLogger(__name__)


class Statistics(metaclass=GoogleDocstringInheritanceMeta):
    """Abstract class to interface a statistics library."""

    dataset: Dataset
    """The dataset."""

    n_samples: int
    """The number of samples."""

    n_variables: int
    """The number of variables."""

    name: str
    """The name of the object."""

    SYMBOLS = {}

    def __init__(
        self,
        dataset: Dataset,
        variables_names: Iterable[str] | None = None,
        name: str | None = None,
    ) -> None:
        """
        Args:
            dataset: A dataset.
            variables_names: The variables of interest.
                Default: consider all the variables available in the dataset.
            name: A name for the object.
                Default: use the concatenation of the class and dataset names.
        """  # noqa: D205,D212,D415
        class_name = self.__class__.__name__
        default_name = f"{class_name}_{dataset.name}"
        self.name = name or default_name
        msg = f"Create {self.name}, a {class_name} library."
        LOGGER.info(msg)
        self.dataset = dataset.get_all_data(by_group=False, as_dict=True)
        self.n_samples = dataset.n_samples
        self.names = variables_names or dataset.variables
        self.n_variables = dataset.n_variables

    def __str__(self) -> str:
        msg = MultiLineString()
        msg.add(self.name)
        msg.indent()
        msg.add("n_samples: {}", self.n_samples)
        msg.add("n_variables: {}", self.n_variables)
        msg.add("variables: {}", pretty_str(self.names))
        return str(msg)

    def compute_tolerance_interval(
        self,
        coverage: float,
        confidence: float = 0.95,
        side: ToleranceIntervalSide = ToleranceIntervalSide.BOTH,
    ) -> dict[str, tuple[ndarray, ndarray]]:  # noqa: D102
        r"""Compute a tolerance interval :math:`\text{TI}[X]`.

        This coverage level is the minimum percentage of belonging to the TI.
        The tolerance interval is computed with a confidence level
        and can be either lower-sided, upper-sided or both-sided.

        Args:
            coverage: A minimum percentage of belonging to the TI.
            confidence: A level of confidence in [0,1].
            side: The type of the tolerance interval
                characterized by its *sides* of interest,
                either a lower-sided tolerance interval :math:`[a, +\infty[`,
                an upper-sided tolerance interval :math:`]-\infty, b]`,
                or a two-sided tolerance interval :math:`[c, d]`.

        Returns:
            The tolerance limits of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["tolerance_interval"] = "TI"

    def compute_a_value(self) -> dict[str, ndarray]:
        r"""Compute the A-value :math:`\text{Aval}[X]`.

        Returns:
            The A-value of the different variables.
        """
        result = self.compute_tolerance_interval(
            1 - 0.1, 0.99, ToleranceIntervalSide.LOWER
        )
        result = {name: value[0] for name, value in result.items()}
        return result

    SYMBOLS["a_value"] = "Aval"

    def compute_b_value(self) -> dict[str, ndarray]:
        r"""Compute the B-value :math:`\text{Bval}[X]`.

        Returns:
            The B-value of the different variables.
        """
        result = self.compute_tolerance_interval(
            1 - 0.1, 0.95, ToleranceIntervalSide.LOWER
        )
        result = {name: value[0] for name, value in result.items()}
        return result

    SYMBOLS["b_value"] = "Bval"

    def compute_maximum(self) -> dict[str, ndarray]:
        r"""Compute the maximum :math:`\text{Max}[X]`.

        Returns:
            The maximum of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["maximum"] = "Max"

    def compute_mean(self) -> dict[str, ndarray]:
        r"""Compute the mean :math:`\mathbb{E}[X]`.

        Returns:
            The mean of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["mean"] = "E"

    def compute_mean_std(
        self,
        std_factor: float,
    ) -> dict[str, ndarray]:
        r"""Compute a margin :math:`\text{Margin}[X]=\mathbb{E}[X]+\kappa\mathbb{S}[X]`.

        Args:
            std_factor: The weight :math:`\kappa` of the standard deviation.

        Returns:
            The margin for the different variables.
        """
        result = self.compute_mean()
        for name, value in self.compute_standard_deviation().items():
            result[name] += std_factor * value
        return result

    compute_margin = compute_mean_std

    SYMBOLS["mean_std"] = "E_StD"
    SYMBOLS["margin"] = "Margin"

    def compute_minimum(self) -> dict[str, ndarray]:
        r"""Compute the :math:`\text{Min}[X]`.

        Returns:
            The minimum of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["minimum"] = "Min"

    def compute_median(self) -> dict[str, ndarray]:
        r"""Compute the median :math:`\text{Med}[X]`.

        Returns:
            The median of the different variables.
        """
        result = self.compute_quantile(0.5)
        return result

    SYMBOLS["median"] = "Med"

    def compute_percentile(
        self,
        order: int,
    ) -> dict[str, ndarray]:
        r"""Compute the n-th percentile :math:`\text{p}[X; n]`.

        Args:
            order: The order :math:`n` of the percentile.
                Either 0, 1, 2, ... or 100.

        Returns:
            The percentile of the different variables.
        """
        if not isinstance(order, int) or order > 100 or order < 0:
            raise TypeError(
                "Percentile order must be an integer between 0 and 100 inclusive."
            )
        prob = order / 100.0
        result = self.compute_quantile(prob)
        return result

    SYMBOLS["percentile"] = "p"

    def compute_probability(
        self,
        thresh: float,
        greater: bool = True,
    ) -> dict[str, ndarray]:
        r"""Compute the probability related to a threshold.

        Either :math:`\mathbb{P}[X \geq x]` or :math:`\mathbb{P}[X \leq x]`.

        Args:
            thresh: A threshold :math:`x`.
            greater: The type of probability.
                If True,
                compute the probability of exceeding the threshold.
                Otherwise,
                compute the opposite.

        Returns:
            The probability of the different variables
        """
        raise NotImplementedError

    SYMBOLS["probability"] = "P"

    def compute_quantile(
        self,
        prob: float,
    ) -> dict[str, ndarray]:
        r"""Compute the quantile :math:`\mathbb{Q}[X; \alpha]` related to a probability.

        Args:
            prob: A probability :math:`\alpha` between 0 and 1.

        Returns:
            The quantile of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["quantile"] = "Q"

    def compute_quartile(
        self,
        order: int,
    ) -> dict[str, ndarray]:
        r"""Compute the n-th quartile :math:`q[X; n]`.

        Args:
            order: The order :math:`n` of the quartile. Either 1, 2 or 3.

        Returns:
            The quartile of the different variables.
        """
        quartiles = [0.25, 0.5, 0.75]
        if order not in [1, 2, 3]:
            raise ValueError("Quartile order must be in [1,2,3]")
        prob = quartiles[order - 1]
        result = self.compute_quantile(prob)
        return result

    SYMBOLS["quartile"] = "q"

    def compute_range(self) -> dict[str, ndarray]:
        r"""Compute the range :math:`R[X]`.

        Returns:
            The range of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["range"] = "R"

    def compute_standard_deviation(self) -> dict[str, ndarray]:
        r"""Compute the standard deviation :math:`\mathbb{S}[X]`.

        Returns:
            The standard deviation of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["standard_deviation"] = "StD"

    def compute_variation_coefficient(self) -> dict[str, ndarray]:
        r"""Compute the coefficient of variation :math:`CoV[X]`.

        This is the standard deviation normalized by the expectation:
        :math:`CoV[X]=\mathbb{E}[S]/\mathbb{E}[X]`.

        Returns:
            The coefficient of variation of the different variables.
        """
        mean = self.compute_mean()
        standard_deviation = self.compute_standard_deviation()
        return {k: standard_deviation[k] / mean[k] for k in mean}

    SYMBOLS["variation_coefficient"] = "CoV"

    def compute_variance(self) -> dict[str, ndarray]:
        r"""Compute the variance :math:`\mathbb{V}[X]`.

        Returns:
            The variance of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["variance"] = "V"

    def compute_moment(
        self,
        order: int,
    ) -> dict[str, ndarray]:
        r"""Compute the n-th moment :math:`M[X; n]`.

        Args:
            order: The order :math:`n` of the moment.

        Returns:
            The moment of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["moment"] = "M"

    @classmethod
    def compute_expression(
        cls,
        variable_name: str,
        statistic_name: str,
        show_name: bool = False,
        **options: bool | float | int,
    ) -> str:
        """Return the expression of a statistical function applied to a variable.

        E.g. "P[X >= 1.0]" for the probability that X exceeds 1.0.

        Args:
            variable_name: The name of the variable, e.g. ``"X"``.
            statistic_name: The name of the statistic, e.g. ``"probability"``.
            show_name: If True, show option names.
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

        if show_name:
            values = []
            for name in sorted(options):
                values.append(f"{name}={options[name]}")
        else:
            values = []
            for name in sorted(options):
                if isinstance(options[name], Enum):
                    values.append(str(options[name].name))
                else:
                    values.append(str(options[name]))

        value = ", ".join(values)
        if value and not separator:
            separator = "; "

        return (
            f"{cls.SYMBOLS.get(statistic_name, statistic_name)}"
            f"[{variable_name}{separator}{value}]"
        )
