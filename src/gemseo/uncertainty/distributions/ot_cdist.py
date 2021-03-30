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
Composed distribution based on OpenTURNS
========================================

Overview
--------

The :class:`.OTComposedDistribution` class is a concrete class
inheriting from :class:`.ComposedDistribution` which is an abstract one.
OT stands for `OpenTURNS <http://www.openturns.org/>`_
which is the library it relies on.

Construction
------------

The :class:`.OTComposedDistribution` of a list of given uncertain variables
is built from a list of :class:`.OTDistribution` objects
implementing the probability distributions of these variables
based on the OpenTURNS library and from a copula name.

.. note::

   A copula is a mathematical function used to define the dependence
   between random variables from their cumulative density functions.
   `See more <https://en.wikipedia.org/wiki/Copula_(probability_theory)>`__.
"""
from __future__ import absolute_import, division, unicode_literals

import openturns as ots
from future import standard_library
from numpy import array

from gemseo.uncertainty.distributions.distribution import ComposedDistribution

standard_library.install_aliases()


class OTComposedDistribution(ComposedDistribution):
    """ OpenTURNS composed distribution. """

    COPULA = {ComposedDistribution.INDEPENDENT_COPULA: ots.IndependentCopula}

    def __init__(self, distributions, copula=ComposedDistribution.INDEPENDENT_COPULA):
        """Constructor.

        :param list(OTDistribution) distributions: list of OT distributions.
        :param str copula: copula name.
        """
        super(OTComposedDistribution, self).__init__(distributions, copula)
        marginals = [
            marginal
            for distribution in distributions
            for marginal in distribution.marginals
        ]
        ot_copula = self.COPULA[copula](len(marginals))
        self.distribution = ots.ComposedDistribution(marginals, ot_copula)
        self._mapping = {}
        index = 0
        for id_dist, distribution in enumerate(distributions):
            for id_marg in range(distribution.dimension):
                self._mapping[index] = (id_dist, id_marg)
                index += 1
        self._set_bounds(distributions)

    def get_sample(self, n_samples=1):
        """Get sample.

        :param n_samples: number of samples.
        :type n_samples: int
        :return: samples
        :rtype: list(array)
        """
        sample = array(self.distribution.getSample(n_samples))
        return sample

    def cdf(self, vector):
        """Evaluate the cumulative density functions of the marginals
        of a random variable for a given instance.

        :param array vector: instance of the random variable.
        """
        tmp = []
        for index, value in enumerate(vector):
            id1 = self._mapping[index][0]
            id2 = self._mapping[index][1]
            value = ots.Point([value])
            tmp.append(self.marginals[id1].marginals[id2].computeCDF(value))
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
            tmp.append(self.marginals[id1].marginals[id2].computeQuantile(value)[0])
        return array(tmp)

    def _pdf(self, index):
        id1 = self._mapping[index][0]
        id2 = self._mapping[index][1]

        def pdf(level):
            return self.marginals[id1].marginals[id2].computePDF(level)

        return pdf

    def _cdf(self, index):
        id1 = self._mapping[index][0]
        id2 = self._mapping[index][1]

        def cdf(level):
            return self.marginals[id1].marginals[id2].computeCDF(level)

        return cdf
