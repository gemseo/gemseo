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
Distributions based on scipy
============================

Overview
--------

The :class:`.SPDistribution` class is a concrete class
inheriting from :class:`.Distribution` which is an abstract one.
SP stands for `scipy <https://docs.scipy.org/doc/scipy/reference/
tutorial/stats.html>`_ which is the library it relies on.

Construction
------------

The :class:`.SPDistribution` of a given uncertain variable is built
from mandatory arguments:

- a variable name,
- a distribution name recognized by OpenTURNS,
- a set of parameters provided as a dictionary
  of keyword arguments named as the arguments of the scipy constructor
  of this distribution.

.. warning::

    The distribution parameters must be provided according to the signature
    of the scipy classes. `Access the scipy documentation
    <https://docs.scipy.org/doc/scipy/reference/stats.html>`_.

The constructor has also optional arguments:

- a variable dimension (default: 1),
- a standard representation of these parameters
  (default: use :code:`parameters`).

Classical distributions
-----------------------

This module also implements a deliberately limited selection
of classical probability distributions
in a user-friendly way: :class:`.SPExponentialDistribution`,
:class:`.SPNormalDistribution`, :class:`.SPTriangularDistribution`,
and :class:`.SPUniformDistribution`. More precisely,
the argument whose nature is a dictionary of keyword parameters
is replaced with several user-defined keyword arguments.
In this way, the use writes :code:`SPUniformDistribution('x', -1., 3.)`
or :code:`SPUniformDistribution('x', lower=-1., upper=3.)`
instead of :code:`SPDistribution('x', 'Normal', (-1., 3.))`.
Furthermore, these classes inheriting from :class:`.SPDistribution`
are documented in such a way that a newbie could easily apprehend them.
"""
from __future__ import absolute_import, division, unicode_literals

import scipy.stats as sp_stats
from future import standard_library
from numpy import array, vstack

from gemseo.uncertainty.distributions.distribution import Distribution
from gemseo.uncertainty.distributions.sp_cdist import SPComposedDistribution

standard_library.install_aliases()

from gemseo import LOGGER

SP_WEBSITE = "https://docs.scipy.org/doc/scipy/reference/stats.html"


class SPDistribution(Distribution):

    """The SPDistribution inherits from Distribution. It relies on the
    probabilistic modeling features of the scipy library.
    """

    COMPOSED_DISTRIBUTION = SPComposedDistribution

    def __init__(
        self, variable, distribution, parameters, dimension=1, standard_parameters=None
    ):
        """Constructor

        :param str variable: variable name.
        :param str distribution: distribution name.
        :param tuple parameters: distribution parameters.
        :param int dimension: variable dimension.
        :param dict standard_parameters: standard parameters.
        """
        super(SPDistribution, self).__init__(
            variable, distribution, parameters, dimension, standard_parameters
        )
        self.marginals = self.__create_distributions(
            self.distribution_name, self.parameters
        )
        math_support = str(self.support)
        LOGGER.info("|_ Mathematical support: %s ", math_support)
        num_range = str(self.range)
        LOGGER.info("|_ Numerical range: %s", num_range)

    def get_sample(self, n_samples=1):
        """Get several samples.

        :param int n_samples: number of samples.
        :return: samples
        :rtype: array
        """
        return vstack([marginal.rvs(n_samples) for marginal in self.marginals]).T

    def cdf(self, vector):
        """Evaluate the cumulative density function of the random variable
        marginals for a given instance.

        :param array vector: instance of the random variable.
        :return: cdf values
        :rtype: array
        """
        return array(
            [self.marginals[index].cdf(value) for index, value in enumerate(vector)]
        )

    def inverse_cdf(self, vector):
        """Evaluate the inverse of the cumulative density function of the
        random variable marginals for a given unit vector .

        :param array vector: vector of values comprised between 0 and 1 with
            same dimension as the random variable.
        :return: inverse cdf values
        :rtype: array
        """
        return array(
            [self.marginals[index].ppf(value) for index, value in enumerate(vector)]
        )

    @property
    def mean(self):
        """Get the mean of the random variable.

        :return: mean of the random variable.
        :rtype: array
        """
        mean = [marginal.mean() for marginal in self.marginals]
        return array(mean)

    @property
    def standard_deviation(self):
        """Get the standard deviation of the random variable.

        :return: standard deviation of the random variable.
        :rtype: array
        """
        std = [marginal.std() for marginal in self.marginals]
        return array(std)

    def __create_distributions(self, distribution, parameters):
        """For each variable dimension, instantiate an openturns distribution
        from its class name and parameters.

        :param str distribution: distribution name,
            ie its openturns class name.
        :param parameters: parameters of the scipy distribution.
        :return: scipy distribution.
        """
        try:
            sp_dist = getattr(sp_stats, distribution)
        except Exception:
            raise ValueError("%s is an unknown scipy " "distribution. " % distribution)
        try:
            sp_dist(**parameters)
        except Exception:
            raise ValueError(
                "%s does not take these arguments. "
                "More details on: %s" % (distribution, SP_WEBSITE)
            )
        sp_dist = [sp_dist(**parameters)] * self.dimension
        self.__set_bounds(sp_dist)
        return sp_dist

    def __set_bounds(self, distributions, extrema_level=1e-12):
        """Set mathematical and numerical bounds (= support and range).

        :param distributions: list of scipy distributions.
        :param float extremal_level: quantile level corresponding to the lower
            bound of the numerical random variable range. Default: 1e-12.
        """
        self.math_lower_bound = []
        self.math_upper_bound = []
        self.num_lower_bound = []
        self.num_upper_bound = []
        for distribution in distributions:
            dist_range = distribution.interval(1.0)
            self.math_lower_bound.append(dist_range[0])
            self.math_upper_bound.append(dist_range[1])
            l_b = distribution.ppf(extrema_level)
            u_b = distribution.ppf(1 - extrema_level)
            self.num_lower_bound.append(l_b)
            self.num_upper_bound.append(u_b)
        self.math_lower_bound = array(self.math_lower_bound)
        self.math_upper_bound = array(self.math_upper_bound)
        self.num_lower_bound = array(self.num_lower_bound)
        self.num_upper_bound = array(self.num_upper_bound)

    def _pdf(self, index):
        """Get the probability density function of a marginal.

        :param int index: marginal index.
        :return: probability density function
        :rtype: function
        """

        def pdf(point):
            """Probability Density Function (PDF).

            :param float point: point value.
            :return: PDF value.
            :rtype: float
            """
            return self.marginals[index].pdf(point)

        return pdf

    def _cdf(self, index):
        """Get the cumulative density function of a marginal.

        :param int index: marginal index.
        :return: cumulative density function
        :rtype: function
        """

        def cdf(level):
            """Cumulative Density Function (CDF).

            :param float point: point value.
            :return: CDF value.
            :rtype: float
            """
            return self.marginals[index].cdf(level)

        return cdf


class SPNormalDistribution(SPDistribution):
    """ Create a normal distribution. """

    def __init__(self, variable, mu=0.0, sigma=1.0, dimension=1):
        """Constructor.

        :param str variable: variable name.
        :param float mu: mean.
        :param float sigma: standard deviation.
        :param int dimension: dimension.
        """
        standard_parameters = {self.MU: mu, self.SIGMA: sigma}
        parameters = {"loc": mu, "scale": sigma}
        super(SPNormalDistribution, self).__init__(
            variable, "norm", parameters, dimension, standard_parameters
        )


class SPUniformDistribution(SPDistribution):
    """ Create a uniform distribution. """

    def __init__(self, variable, lower=0.0, upper=1.0, dimension=1):
        """Constructor.

        :param str variable: variable name.
        :param float lower: lower bound.
        :param float upper: upper bound.
        :param int dimension: dimension.
        """
        parameters = {"loc": lower, "scale": upper - lower}
        standard_parameters = {self.LOWER: lower, self.UPPER: upper}
        super(SPUniformDistribution, self).__init__(
            variable, "uniform", parameters, dimension, standard_parameters
        )


class SPTriangularDistribution(SPDistribution):
    """ Create a triangular distribution. """

    def __init__(self, variable, lower=0.0, mode=0.5, upper=1.0, dimension=1):
        """Constructor.

        :param str variable: variable name.
        :param float lower: lower bound.
        :param float mode: mode.
        :param float upper: upper bound.
        :param int dimension: dimension.
        """
        parameters = {
            "loc": lower,
            "scale": upper - lower,
            "c": (mode - lower) / float(upper - lower),
        }
        standard_parameters = {self.LOWER: lower, self.MODE: mode, self.UPPER: upper}
        super(SPTriangularDistribution, self).__init__(
            variable, "triang", parameters, dimension, standard_parameters
        )


class SPExponentialDistribution(SPDistribution):
    """ Create a exponential distribution. """

    def __init__(self, variable, rate=1.0, loc=0.0, dimension=1):
        """Constructor.

        :param str variable: variable name.
        :param float rate: rate parameter.
        :param float loc: location parameter.
        :param int dimension: dimension.
        """
        parameters = {"loc": loc, "scale": 1 / float(rate)}
        super(SPExponentialDistribution, self).__init__(
            variable, "expon", parameters, dimension
        )
