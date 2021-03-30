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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Empirical estimation of statistics from a dataset
=================================================

Overview
--------

The :class:`.EmpiricalStatistics` class inherits from the
abstract :class:`.Statistics` class and aims to estimate statistics
from a :class:`.Dataset`, based on empirical estimators.

Construction
------------

A :class:`.EmpiricalStatistics` is built from a :class:`.Dataset` and
optionally a list of variables names.
In this case, statistics are only computed for these variables.
Otherwise, statistics are computed for all variables.
Lastly, the user can name its :class:`.Statistics`. By default,
the name is the concatenation of 'EmpiricalStatistics' and
and the name of the :class:`.Dataset`.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import all as npall
from numpy import max as npmax
from numpy import mean
from numpy import min as npmin
from numpy import quantile, std, var
from scipy.stats import moment

from gemseo.uncertainty.statistics.statistics import Statistics

standard_library.install_aliases()

from gemseo import LOGGER


class EmpiricalStatistics(Statistics):
    """ Empirical estimation of statistics. """

    def __init__(self, dataset, variables_names=None, name=None):
        """Constructor

        :param Dataset dataset: dataset
        :param list(str) variables_names: list of variables names
            or list of variables names. If None, the method considers
            all variables from dataset. Default: None.
        :param str name: name of the object.
            If None, use the concatenation of class and dataset names.
            Default: None."""
        name = name or dataset.name
        super(EmpiricalStatistics, self).__init__(dataset, variables_names, name)

    def maximum(self):
        """Get the maximum.

        :return: maximum
        :rtype: dict
        """
        result = {name: npmax(self.dataset[name], 0) for name in self.names}
        return result

    def mean(self):
        """Get the mean.

        :return: mean
        :rtype: dict
        """
        result = {name: mean(self.dataset[name], 0) for name in self.names}
        return result

    def minimum(self):
        """Get the minimum.

        :return: minimum
        :rtype: dict
        """
        result = {name: npmin(self.dataset[name], 0) for name in self.names}
        return result

    def probability(self, thresh, greater=True):
        """Compute a probability associated to a threshold. This threshold
        is a dictionary of arrays indexed by variables names.
        For a multidimensional variable, the probability to be greater
        (or lower) than the threshold is defined as the probability that all
        variables components are greater (respectively lower) than their
        counterparts in the threshold.

        :param dict thresh: threshold
        :param bool greater: if True, compute the probability the probability
            of exceeding the threshold, if False, compute the reverse.
            Default: True.
        :return: probability
        """
        if greater:
            result = {
                name: mean(npall(self.dataset[name] >= thresh[name], 1))
                for name in self.names
            }
        else:
            result = {
                name: mean(npall(self.dataset[name] <= thresh[name], 1))
                for name in self.names
            }
        return result

    def quantile(self, prob):
        """Get the quantile associated to a given probability.

        :param int merge: if True, merge variables. Default: True.
        :return: quantile
        :rtype: dict
        """
        result = {name: quantile(self.dataset[name], prob, 0) for name in self.names}
        return result

    def standard_deviation(self):
        """Get the standard deviation.

        :return: standard deviation
        :rtype: dict
        """
        result = {name: std(self.dataset[name], 0) for name in self.names}
        return result

    def variance(self):
        """Get the variance.

        :return: variance
        :rtype: dict
        """
        result = {name: var(self.dataset[name], 0) for name in self.names}
        return result

    def moment(self, order):
        """Compute the central moment for a given order.

        :param int order: moment order.
        :return: moment
        :rtype: dict
        """
        result = {name: moment(self.dataset[name], order, 0) for name in self.names}
        return result

    def range(self):
        """Get the range of variables.

        :return: range of variables
        """
        lower = self.minimum()
        upper = self.maximum()
        result = {name: upper[name] - lower[name] for name in self.names}
        return result
