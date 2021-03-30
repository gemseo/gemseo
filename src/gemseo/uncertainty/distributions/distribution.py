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
The notions of distribution and composed distributions
======================================================

Overview
--------

The abstract :class:`.Distribution` class implements the concept of
probability distribution. It is enriched by concrete classes
such as :class:`.OTDistribution` and :class:`.SPDistribution`.

Similarly, the abstract :class:`.ComposedDistribution` class implements
the concept of composed probability distribution from a list
of probability distributions. It inherits from :class:`.Distribution`.

Construction
------------

The :class:`.Distribution` of a given uncertain variable is built
from a recognized distribution name,
a variable dimension, a set of parameters
and optionally a standard representation of these parameters.

The :class:`.ComposedDistribution` of a list of given uncertain variables
is built from a list of :class:`.Distribution` objects
implementing the probability distributions of these variables
and from a copula name.

.. note::

   A copula is a mathematical function used to define the dependence
   between random variables from their cumulative density functions.
   `See more <https://en.wikipedia.org/wiki/Copula_(probability_theory)>`_.

Capabilities
------------

From a :class:`.Distribution`, we can easily get statistics,
such as :attr:`.Distribution.mean`,
:attr:`.Distribution.standard_deviation`,
numerical :attr:`.Distribution.range` and
mathematical :attr:`.Distribution.support`,
or plot the cumulative and probability density functions,
either for a given marginal (:meth:`.Distribution.plot`)
or for all marginals (:meth:`.Distribution.plot_all`).

We can also compute the cumulative density function
(:meth:`.Distribution.cdf`)
for the different marginals of the random variable,
as well as the inverse cumulative density function
(:meth:`.Distribution.inverse_cdf`).

Lastly, we can get realizations of the random variable
by means of the :meth:`.Distribution.get_sample` method.

.. note::

   As :class:`.ComposedDistribution` inherits from :class:`.Distribution`,
   it has the same capabilities.
"""
from __future__ import absolute_import, division, unicode_literals

import matplotlib.pyplot as plt
from future import standard_library
from numpy import arange, array, concatenate
from past.utils import old_div

standard_library.install_aliases()

from gemseo import LOGGER


class Distribution(object):
    """ Distribution. """

    MATHEMATICAL_LOWER_BOUND = "math_l_b"
    MATHEMATICAL_UPPER_BOUND = "math_u_b"
    NUMERICAL_LOWER_BOUND = "num_l_b"
    NUMERICAL_UPPER_BOUND = "num_u_b"
    MU = "mu"
    SIGMA = "sigma"
    LOWER = "lower"
    UPPER = "upper"
    MODE = "mode"
    RATE = "rate"
    LOC = "loc"

    def __init__(
        self, variable, distribution, parameters, dimension, standard_parameters=None
    ):
        """Constructor

        :param str variable: variable name.
        :param str distribution: distribution name.
        :param parameters: distribution parameters.
        :type parameters: tuple or dict
        :param int dimension: variable dimension.
        :param dict standard_parameters: standard parameters.
        """
        self.math_lower_bound = None
        self.math_upper_bound = None
        self.num_lower_bound = None
        self.num_upper_bound = None
        self.distribution = None
        self.marginals = None
        self.dimension = dimension
        self.variable_name = variable
        self.distribution_name = distribution
        self.transformation = variable
        self.parameters = parameters
        if standard_parameters is None:
            self.standard_parameters = self.parameters
        else:
            self.standard_parameters = standard_parameters
        LOGGER.info("Define the random variable: %s ", variable)
        LOGGER.info("|_ Distribution: %s ", str(self))
        LOGGER.info("|_ Dimension: %s ", str(dimension))

    def __str__(self):
        """String representation of the object.

        :return: string representation.
        :rtype: str
        """
        label = self.distribution_name + "("
        if isinstance(self.standard_parameters, dict):
            param = iter(self.standard_parameters.items())
            param = sorted([name + "=" + str(val) for name, val in param])
            label += ", ".join(param)
        else:
            param = [str(option) for option in self.standard_parameters]
            label += ", ".join(param)
        label += ")"
        return label

    def get_sample(self, n_samples=1):
        """Get several samples.

        :param int n_samples: number of samples.
        :return: samples
        :rtype: array
        """
        raise NotImplementedError

    def cdf(self, vector):
        """Evaluate the cumulative density function of the random variable
        marginals for a given instance.

        :param array vector: instance of the random variable.
        :return: cdf values
        :rtype: array
        """
        raise NotImplementedError

    def inverse_cdf(self, vector):
        """Evaluate the inverse of the cumulative density function of the
        random variable marginals for a given unit vector .

        :param array vector: vector of values comprised between 0 and 1 with
            same dimension as the random variable.
        :return: inverse cdf values
        :rtype: array
        """
        raise NotImplementedError

    @property
    def mean(self):
        """Get the mean of the random variable.

        :return: mean of the random variable.
        :rtype: array
        """
        raise NotImplementedError

    @property
    def standard_deviation(self):
        """Get the standard deviation of the random variable.

        :return: standard deviation of the random variable.
        :rtype: array
        """
        raise NotImplementedError

    @property
    def range(self):
        """Get the numerical range for the different components.

        :return: numerical range.
        :rtype: list(array)
        """
        value = [
            array([l_b, u_b])
            for l_b, u_b in zip(self.num_lower_bound, self.num_upper_bound)
        ]
        return value

    @property
    def support(self):
        """Get the mathematical support for the different components.

        :return: mathematical support.
        :rtype: list(array)
        """
        value = [
            array([l_b, u_b])
            for l_b, u_b in zip(self.math_lower_bound, self.math_upper_bound)
        ]
        return value

    def plot_all(self, show=True, save=False, prefix=None):
        """Plot the probability density function and the cumulative density
        function of the random variable.

        :param str prefix: If not None, start the filename with a prefix.
            Default: None.
        """
        for index in range(self.dimension):
            self.plot(index, show, save, prefix)

    def plot(self, index=0, show=True, save=False, prefix=None):
        """Plot the probability density function and the cumulative density
        function of the random variable.

        :param str prefix: If not None, start the filename with a prefix.
            Default: None.
        """
        variable_name = self.variable_name + "(" + str(index) + ")"
        l_b = self.num_lower_bound[index]
        u_b = self.num_upper_bound[index]
        xvals = arange(l_b, u_b, old_div((u_b - l_b), 100))
        y1vals = [self._pdf(index)(xval) for xval in xvals]
        ax1 = plt.subplot(121)
        ax1.plot(xvals, y1vals)
        ax1.set_xlabel(variable_name)
        ax1.set_title("PDF")
        y2vals = [self._cdf(index)(xval) for xval in xvals]
        ax2 = plt.subplot(122)
        ax2.plot(xvals, y2vals)
        ax2.set_xlabel(variable_name)
        ax2.set_title("CDF")
        fname = "distribution_" + self.variable_name + "_" + str(index)
        if show:
            plt.show()
        if save:
            if prefix is not None:
                fname = prefix + "_" + fname
            plt.savefig(fname + ".pdf")

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
            raise NotImplementedError

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
            raise NotImplementedError

        return cdf


class ComposedDistribution(Distribution):
    """ Composed Distribution. """

    INDEPENDENT_COPULA = "independent_copula"
    AVAILABLE_COPULA = [INDEPENDENT_COPULA]
    COMPOSED = "Composed"

    def __init__(self, distributions, copula=INDEPENDENT_COPULA):
        """Constructor.

        :param list(Distribution) distributions: list of distributions.
        :param str copula: copula name. Default: INDEPENDENT_COPULA.
        """
        dimension = sum([distribution.dimension for distribution in distributions])
        self._marginal_variables = [
            distribution.variable_name for distribution in distributions
        ]
        variable = "_".join(self._marginal_variables)
        super(ComposedDistribution, self).__init__(
            variable, self.COMPOSED, (copula,), dimension
        )
        self.marginals = distributions
        LOGGER.info("|_ Marginals:")
        for distribution in distributions:
            LOGGER.info(
                "   - %s(%s): %s",
                distribution.variable_name,
                distribution.dimension,
                distribution,
            )

    def _set_bounds(self, distributions):
        """Set mathematical and numerical bounds (= support and range).

        :param list(Distribution) distributions: list of distributions.
        """
        self.math_lower_bound = array([])
        self.math_upper_bound = array([])
        self.num_lower_bound = array([])
        self.num_upper_bound = array([])
        for dist in distributions:
            self.math_lower_bound = concatenate(
                (self.math_lower_bound, dist.math_lower_bound)
            )
            self.num_lower_bound = concatenate(
                (self.num_lower_bound, dist.num_lower_bound)
            )
            self.math_upper_bound = concatenate(
                (self.math_upper_bound, dist.math_upper_bound)
            )
            self.num_upper_bound = concatenate(
                (self.num_upper_bound, dist.num_upper_bound)
            )

    @property
    def mean(self):
        """Get the mean of the random variable.

        :return: mean of the random variable.
        :rtype: array
        """
        mean = [marginal.mean for marginal in self.marginals]
        return array(mean).flatten()

    @property
    def standard_deviation(self):
        """Get the standard deviation of the random variable.

        :return: standard deviation of the random variable.
        :rtype: array
        """
        std = [marginal.standard_deviation for marginal in self.marginals]
        return array(std).flatten()

    def get_sample(self, n_samples=1):
        """Get sample.

        :param int n_samples: number of samples.
        :return: samples
        :rtype: list(array)
        """
        sample = self.marginals[0].get_sample(n_samples)
        for marginal in self.marginals[1:]:
            sample = concatenate((sample, marginal.get_sample(n_samples)), axis=1)
        return sample
