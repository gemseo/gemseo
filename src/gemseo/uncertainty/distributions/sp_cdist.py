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
Composed distribution based on scipy
====================================

Overview
--------

The :class:`.SPComposedDistribution` class is a concrete class
inheriting from :class:`.ComposedDistribution` which is an abstract one.

Construction
------------

The :class:`.SPComposedDistribution` of a list of given uncertain variables
is built from a list of :class:`.SPDistribution` objects
implementing the probability distributions of these variables
based on the OpenTURNS library and from a copula name.

.. note::

   A copula is a mathematical function used to define the dependence
   between random variables from their cumulative density functions.
   `See more <https://en.wikipedia.org/wiki/Copula_(probability_theory)>`__.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import array

from gemseo.uncertainty.distributions.distribution import ComposedDistribution

standard_library.install_aliases()


class SPComposedDistribution(ComposedDistribution):
    """ Scipy composed distribution. """

    COPULA = {ComposedDistribution.INDEPENDENT_COPULA: None}

    def __init__(self, distributions, copula=ComposedDistribution.INDEPENDENT_COPULA):
        """Constructor.

        :param list(SPDistribution) distributions: list of OT distributions.
        :param str copula: copula name.
        """
        super(SPComposedDistribution, self).__init__(distributions, copula)
        self.distribution = distributions
        self._mapping = {}
        index = 0
        for id_marg, marginal in enumerate(self.distribution):
            for id_submarg in range(marginal.dimension):
                self._mapping[index] = (id_marg, id_submarg)
                index += 1
        self._set_bounds(distributions)

    def cdf(self, vector):
        """Evaluate the cumulative density functions of the marginals
        of a random variable for a given instance.

        :param array vector: instance of the random variable.
        """
        tmp = []
        for index, value in enumerate(vector):
            id1 = self._mapping[index][0]
            id2 = self._mapping[index][1]
            tmp.append(self.distribution[id1].marginals[id2].cdf(value))
        return array(tmp)

    def inverse_cdf(self, vector):
        """Evaluate the inverses of the cumulative density functions of the
        marginals of a random variable for a given vector .

        :param array vector: vector of values comprised between 0 and 1 with
            same dimension as the random variable.
        """
        tmp = []
        for index, value in enumerate(vector):
            id1 = self._mapping[index][0]
            id2 = self._mapping[index][1]
            tmp.append(self.distribution[id1].marginals[id2].ppf(value))
        return array(tmp)

    def _pdf(self, index):
        id1 = self._mapping[index][0]
        id2 = self._mapping[index][1]

        def pdf(level):
            return self.distribution[id1].marginals[id2].pdf(level)

        return pdf

    def _cdf(self, index):
        id1 = self._mapping[index][0]
        id2 = self._mapping[index][1]

        def cdf(level):
            return self.distribution[id1].marginals[id2].cdf(level)

        return cdf
