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
"""
Estimation of statistics from a dataset
=======================================

Overview
--------

The abstract :class:`.Statistics` class implements the concept of
statistics library. It is enriched by concrete classes
such as :class:`.EmpiricalStatistics` and :class:`.ParametricStatistics`.

Construction
------------

A :class:`.Statistics` is built from a :class:`.Dataset` and optionally
a list of variables names. In this case, statistics are only computed
for these variables. Otherwise, statistics are computed for all variables.
Lastly, the user can name its :class:`.Statistics`. By default,
the name is the concatenation of the name of the class
overloading :class:`.Statistics`
and the name of the :class:`.Dataset`.

Capabilities
------------

A :class:`.Statistics` returns standard descriptive and statistical measures
for the different variables:

- :meth:`.Statistics.minimum`: the minimum value,
- :meth:`.Statistics.maximum`: the maximum value,
- :meth:`.Statistics.range`: the difference between minimum and maximum values,
- :meth:`.Statistics.mean`: the expectation, a.k.a. mean value,
- :meth:`.Statistics.moment`: the central moment which is a the expected value
  of a specified integer power of the deviation from the mean,
- :meth:`.Statistics.variance`: the variance, which is the mean squared
  variation around the mean value,
- :meth:`.Statistics.standard_deviation`: the standard deviation, which is the
  square root of the variance,
- :meth:`.Statistics.quantile`: the quantile associated with a probability,
  which is the cut point diving the range into a first continuous interval
  with this given probability and a second continuous interval
  with the complementary probability; common *q*-quantiles dividing
  the range into *q* continuous interval with equal probabilities are also
  implemented:

    - :meth:`.Statistics.median` which implements the 2-quantile (50%).
    - :meth:`.Statistics.quartile` whose order (1, 2 or 3) implements
      the 4-quantiles (respectively 25%, 50% and 75%),
    - :meth:`.Statistics.percentile` whose order (1, 2, ..., 99) implements
      the 100-quantiles (1%, 2%, ..., 99%),

- :meth:`.Statistics.probability`: the probability that the random variable
  is larger or smaller than a certain threshold,
- :meth:`.Statistics.tolerance_interval`: the left-sided, right-sided or
  both-sided tolerance interval associated with a given coverage level
  and a given confidence level, which is
  a statistical interval within which, with some confidence level,
  a specified proportion of the random variable realizations falls
  (this proportion is the coverage level)

    - :meth:`.Statistics.a_value`: the A-value, which is the lower bound of the
      left-sided tolerance interval associated with a coverage level
      equal to 99% and a confidence level equal to 95%,
    - :meth:`.Statistics.b_value`: the B-value, which is the lower bound of the
      left-sided tolerance interval associated with a coverage level equal
      to 90% and a confidence level equal to 95%,

"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library

standard_library.install_aliases()

from gemseo import LOGGER


class Statistics(object):
    """Abstract class for Statistics library interface."""

    def __init__(self, dataset, variables_names=None, name=None):
        """
        Constructor

        :param Dataset dataset: dataset
        :param list(str) variables_names: list of variables names.
            If None, the method considers
            all variables from loaded dataset. Default: None.
        :param str name: name of the object.
            If None, use the concatenation of class and dataset names.
            Default: None.
        """
        default_name = self.__class__.__name__ + "_" + dataset.name
        self.name = name or default_name
        msg = 'Create "' + self.name + '"'
        msg += ", a " + self.__class__.__name__ + " library."
        LOGGER.info(msg)
        self.dataset = dataset.get_all_data(by_group=False, as_dict=True)
        self.n_samples = dataset.n_samples
        self.names = variables_names or dataset.variables
        self.n_variables = dataset.n_variables

    def __str__(self):
        msg = self.name + "\n"
        msg += "| n_samples: " + str(self.n_samples) + "\n"
        msg += "| n_variables: " + str(self.n_variables) + "\n"
        msg += "| variables: " + str(self.names) + "\n"
        return msg

    def tolerance_interval(self, coverage, confidence=0.95, side="both"):
        """Compute the tolerance interval (TI) for a given minimum percentage
        of the population and a given confidence level.

        :param float coverage: minimum percentage of belonging to the TI.
        :param float confidence: level of confidence in [0,1]. Default: 0.95.
        :param str side: kind of interval: 'lower' for lower-sided TI,
            'upper' for upper-sided TI and 'both for both-sided TI.
        :return: tolerance limits
        """
        raise NotImplementedError

    def a_value(self):
        """Compute the b-value.

        :return: b-value
        """
        result = self.tolerance_interval(1 - 0.1, 0.99, "lower")
        result = {name: value[0] for name, value in result.items()}
        return result

    def b_value(self):
        """Compute the b-value.

        :return: b-value
        """
        result = self.tolerance_interval(1 - 0.1, 0.95, "lower")
        result = {name: value[0] for name, value in result.items()}
        return result

    def maximum(self):
        """Compute the maximum.

        :return: maximum
        """
        raise NotImplementedError

    def mean(self):
        """Compute the mean.

        :return: mean
        """
        raise NotImplementedError

    def minimum(self):
        """Compute the minimum.

        :return: minimum
        """
        raise NotImplementedError

    def median(self):
        """Compute the median.

        :param options: options
        :return: median
        """
        result = self.quantile(0.5)
        return result

    def percentile(self, order):
        """Compute the percentile.

        :param int order: percentile order, e.g. 4.
        :return: percentile
        """
        if not isinstance(order, int) or order > 100 or order < 0:
            raise TypeError(
                "Percentile order must be an integer " "between 0 and 100 inclusive."
            )
        prob = order / 100.0
        result = self.quantile(prob)
        return result

    def probability(self, thresh, greater):
        """Compute a probability associated to a threshold.

        :param float thresh: threshold
        :param bool greater: if True, compute the probability the probability
            of exceeding the threshold, if False, compute the reverse.
        :return: probability
        """
        raise NotImplementedError

    def quantile(self, prob):
        """Compute a quantile associated to a probability.

        :param float prob: probability between 0 and 1
        :return: quantile
        """
        raise NotImplementedError

    def quartile(self, order):
        """Compute a quartile.

        :param int order: quartile order in [1,2,3]
        :return: quartile
        """
        quartiles = [0.25, 0.5, 0.75]
        if order not in [1, 2, 3]:
            raise ValueError("Quartile order must be in [1,2,3]")
        prob = quartiles[order - 1]
        result = self.quantile(prob)
        return result

    def range(self):
        """Compute the range

        :return: range
        """
        raise NotImplementedError

    def standard_deviation(self):
        """Compute a standard_deviation.

        :return: standard deviation
        """
        raise NotImplementedError

    def variance(self):
        """Compute a variance.

        :return: variance
        """
        raise NotImplementedError

    def moment(self, order):
        """Compute the moment for a given order.

        :param int order: moment index
        :return: moment
        """
        raise NotImplementedError
