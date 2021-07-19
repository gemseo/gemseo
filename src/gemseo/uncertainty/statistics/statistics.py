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
  which is a the expected value
  of a specified integer power
  of the deviation from the mean,
- :meth:`.Statistics.compute_variance`: the variance,
  which is the mean squared variation around the mean value,
- :meth:`.Statistics.compute_standard_deviation`: the standard deviation,
  which is the square root of the variance,
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

from __future__ import division, unicode_literals

import logging
from enum import Enum
from typing import Dict, Iterable, Optional, Tuple

import six
from custom_inherit import DocInheritMeta
from numpy import ndarray

from gemseo.core.dataset import Dataset
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceIntervalSide,
)
from gemseo.utils.string_tools import MultiLineString, pretty_repr

LOGGER = logging.getLogger(__name__)


@six.add_metaclass(
    DocInheritMeta(
        abstract_base_class=True,
        style="google_with_merge",
        include_special_methods=True,
    )
)
class Statistics(object):
    """Abstract class to interface a statistics library.

    Attributes:
        dataset (Dataset): The dataset.
        n_samples (int): The number of samples.
        n_variables (int): The number of variables.
        name (str): The name of the object.
    """

    SYMBOLS = {}

    def __init__(
        self,
        dataset,  # type: Dataset,
        variables_names=None,  # type: Optional[Iterable[str]]
        name=None,  # type: Optional[str]
    ):  # type: (...) -> None # noqa: D205,D212,D415
        """
        Args:
            dataset: A dataset.
            variables_names: The variables of interest.
                Default: consider all the variables available in the dataset.
            name: A name for the object.
                Default: use the concatenation of the class and dataset names.
        """
        class_name = self.__class__.__name__
        default_name = "{}_{}".format(class_name, dataset.name)
        self.name = name or default_name
        msg = "Create {}, a {} library.".format(self.name, class_name)
        LOGGER.info(msg)
        self.dataset = dataset.get_all_data(by_group=False, as_dict=True)
        self.n_samples = dataset.n_samples
        self.names = variables_names or dataset.variables
        self.n_variables = dataset.n_variables

    def __str__(self):  # type: (...) -> str
        msg = MultiLineString()
        msg.add(self.name)
        msg.indent()
        msg.add("n_samples: {}", self.n_samples)
        msg.add("n_variables: {}", self.n_variables)
        msg.add("variables: {}", pretty_repr(self.names))
        return str(msg)

    def compute_tolerance_interval(
        self,
        coverage,  # type: float
        confidence=0.95,  # type: float
        side=ToleranceIntervalSide.BOTH,  # type: ToleranceIntervalSide
    ):  # type: (...) -> Dict[str, Tuple[ndarray,ndarray]]# noqa: D102
        r"""Compute a tolerance interval (TI) for a given coverage level.

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

    def compute_a_value(self):  # type: (...) -> Dict[str,ndarray]
        """Compute the A-value.

        Returns:
            The A-value of the different variables.
        """
        result = self.compute_tolerance_interval(
            1 - 0.1, 0.99, ToleranceIntervalSide.LOWER
        )
        result = {name: value[0] for name, value in result.items()}
        return result

    SYMBOLS["a_value"] = "Aval"

    def compute_b_value(self):  # type: (...) -> Dict[str,ndarray]
        """Compute the B-value.

        Returns:
            The B-value of the different variables.
        """
        result = self.compute_tolerance_interval(
            1 - 0.1, 0.95, ToleranceIntervalSide.LOWER
        )
        result = {name: value[0] for name, value in result.items()}
        return result

    SYMBOLS["b_value"] = "Bval"

    def compute_maximum(self):  # type: (...) -> Dict[str,ndarray]
        """Compute the maximum.

        Returns:
            The maximum of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["maximum"] = "Max"

    def compute_mean(self):  # type: (...) -> Dict[str,ndarray]
        """Compute the mean.

        Returns:
            The mean of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["mean"] = "E"

    def compute_mean_std(
        self,
        std_factor,  # type: float
    ):  # type: (...) -> Dict[str,ndarray]
        """Compute mean + std_factor * std.

        Returns:
            mean + std_factor * std for the different variables.
        """
        result = self.compute_mean()
        for name, value in self.compute_standard_deviation().items():
            result[name] += std_factor * value
        return result

    SYMBOLS["mean_std"] = "E_StD"

    def compute_minimum(self):  # type: (...) -> Dict[str,ndarray]
        """Compute the minimum.

        Returns:
            The minimum of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["minimum"] = "Min"

    def compute_median(self):  # type: (...) -> Dict[str,ndarray]
        """Compute the median.

        Returns:
            The median of the different variables.
        """
        result = self.compute_quantile(0.5)
        return result

    SYMBOLS["median"] = "Med"

    def compute_percentile(
        self,
        order,  # type: int
    ):  # type: (...) -> Dict[str,ndarray]
        """Compute the n-th percentile.

        Args:
            order: The order of the percentile.
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
        thresh,  # type: float
        greater=True,  # type: bool
    ):  # type: (...) -> Dict[str,ndarray]
        """Compute the probability related to a threshold.

        Args:
            thresh: A threshold.
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
        prob,  # type:float
    ):  # type: (...) -> Dict[str,ndarray]
        """Compute the quantile related to a probability.

        Args:
            prob: A probability between 0 and 1.

        Returns:
            The quantile of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["quantile"] = "Q"

    def compute_quartile(
        self,
        order,  # type:int
    ):  # type: (...) -> Dict[str,ndarray]
        """Compute the n-th quartile.

        Args:
            order: The order of the quartile. Either 1, 2 or 3.

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

    def compute_range(self):  # type: (...) -> Dict[str,ndarray]
        """Compute the range.

        Returns:
            The range of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["range"] = "R"

    def compute_standard_deviation(self):  # type: (...) -> Dict[str,ndarray]
        """Compute the standard deviation.

        Returns:
            The standard deviation of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["standard_deviation"] = "StD"

    def compute_variance(self):  # type: (...) -> Dict[str,ndarray]
        """Compute the variance.

        Returns:
            The variance of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["variance"] = "V"

    def compute_moment(
        self,
        order,  # type:int
    ):  # type: (...) -> Dict[str,ndarray]
        """Compute the n-th moment.

        Args:
            order: The order of a moment.

        Returns:
            The moment of the different variables.
        """
        raise NotImplementedError

    SYMBOLS["moment"] = "M"

    @classmethod
    def compute_expression(
        cls,
        variable,  # type:str
        function,  # type:str
        show_name=False,  # type:bool
        **options
    ):  # type: (...) -> str
        """Return the expression of a statistical function applied to a variable.

        Args:
            variable: The name of the variable.
            function: The name of the function.
            show_name: If True, show name. Otherwise, only show value.
            **options: The options passed to the statistical function.

        Returns:
            The expression of the statistical function applied to the variable.
        """
        middle = ""
        if "greater" in options:
            middle = ">=" if options["greater"] else "<="
            options.pop("greater")
        elif function == "probability":
            middle = ">="
        if show_name:
            value = []
            for name in sorted(options):
                value.append("{}={}".format(name, options[name]))
        else:
            value = []
            for name in sorted(options):
                if isinstance(options[name], Enum):
                    value.append(str(options[name].name))
                else:
                    value.append(str(options[name]))
        value = ", ".join(value)
        if value != "" and middle == "":
            middle = "; "
        return "{}[{}{}{}]".format(cls.SYMBOLS[function], variable, middle, value)
