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
Distributions based on OpenTURNS
================================

Overview
--------

The :class:`.OTDistribution` class is a concrete class
inheriting from :class:`.Distribution` which is an abstract one.
OT stands for `OpenTURNS <http://www.openturns.org/>`_
which is the library it relies on.

Similarly, the :class:`.OTComposedDistribution` class is a concrete class
inheriting from :class:`.ComposedDistribution` which is an abstract one.

Construction
------------

The :class:`.OTDistribution` of a given uncertain variable is built
from mandatory arguments:

- a variable name,
- a distribution name recognized by OpenTURNS,
- a set of parameters provided as a tuple
  of positional arguments filled in the order
  specified by the OpenTURNS constructor of this distribution.

.. warning::

    The distribution parameters must be provided according to the signature
    of the openTURNS classes. `Access the openTURNS documentation
    <http://openturns.github.io/openturns/latest/user_manual/
    probabilistic_modelling.html>`_.

The constructor has also optional arguments:

- a variable dimension (default: 1),
- a standard representation of these parameters
  (default: use the parameters provided in the tuple),
- a transformation of the variable (default: no transformation),
- lower and upper bounds for truncation (default: no truncation),
- a threshold for the OpenTURNS truncation tool
  (`more details <http://openturns.github.io/openturns/latest/user_manual/
  _generated/openturns.TruncatedDistribution.html>`_).

Classical distributions
-----------------------

This module also implements a deliberately limited selection
of standard probability distributions
in a user-friendly way: :class:`.OTExponentialDistribution`,
:class:`.OTNormalDistribution`, :class:`.OTTriangularDistribution`,
and :class:`.OTUniformDistribution`. More precisely,
the argument whose nature is a tuple of positional parameters
is replaced with several user-defined keyword arguments.
In this way, the use writes :code:`OTUniformDistribution('x', -1., 3.)`
or :code:`OTUniformDistribution('x', lower=-1., upper=3.)`
instead of :code:`OTDistribution('x', 'Normal', (-1., 3.))`.
Furthermore, these classes inheriting from :class:`.OTDistribution`
are documented in such a way that a newbie could easily apprehend them.
"""
from __future__ import absolute_import, division, unicode_literals

import openturns as ots
from future import standard_library
from numpy import array, inf

from gemseo.uncertainty.distributions.distribution import Distribution
from gemseo.uncertainty.distributions.ot_cdist import OTComposedDistribution

standard_library.install_aliases()

OT_WEBSITE = "http://openturns.github.io/openturns/latest/"
OT_WEBSITE += "user_manual/probabilistic_modelling.html"

from gemseo import LOGGER


class OTDistribution(Distribution):
    """ Interface to an OpenTURNS distribution. """

    COMPOSED_DISTRIBUTION = OTComposedDistribution

    def __init__(
        self,
        variable,
        distribution,
        parameters,
        dimension=1,
        standard_parameters=None,
        transformation=None,
        l_b=None,
        u_b=None,
        threshold=0.5,
    ):
        """Constructor

        :param str variable: variable name.
        :param str distribution: distribution name.
        :param tuple parameters: distribution parameters.
        :param int dimension: variable dimension.
        :param dict standard_parameters: standard parameters.
        :param str transformation: standard variable transformation,
            e.g. 'sin(x)'. If None, no transformation.  Default: None.
        :param float l_b: lower bound for truncation. If None, no lower
            truncation. Default: None.
        :param float u_b: upper bound for truncation. If None, no upper
            truncation. Default: None.
        :param float threshold: threshold value in [0,1].
        """
        super(OTDistribution, self).__init__(
            variable, distribution, parameters, dimension, standard_parameters
        )
        self.marginals = self.__create_distributions(
            self.distribution_name, self.parameters, transformation, l_b, u_b, threshold
        )
        self.distribution = ots.ComposedDistribution(self.marginals)
        math_support = str(self.support)
        LOGGER.info("|_ Mathematical support: %s", math_support)
        num_range = str(self.range)
        LOGGER.info("|_ Numerical range: %s", num_range)
        LOGGER.info("|_ Transformation: %s", self.transformation)

    def get_sample(self, n_samples=1):
        """Get several samples.

        :param int n_samples: number of samples.
        :return: samples
        :rtype: list(array)
        """
        sample = array(self.distribution.getSample(n_samples))
        return sample

    def cdf(self, vector):
        """Evaluate the cumulative density function of the random variable
        marginals for a given instance.

        :param array vector: instance of the random variable.
        :return: cdf values
        :rtype: array
        """
        return array(
            [
                self.marginals[index].computeCDF(ots.Point([value]))
                for index, value in enumerate(vector)
            ]
        )

    def inverse_cdf(self, vector):
        """Evaluate the inverses of the cumulative density functions of the
        random variable marginals for a given unit vector .

        :param array vector: vector of values comprised between 0 and 1 with
            same dimension as the random variable.
        :return: inverse cdf values
        :rtype: array
        """
        return array(
            [
                self.marginals[index].computeQuantile(value)[0]
                for index, value in enumerate(vector)
            ]
        )

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
            return self.marginals[index].computePDF(point)

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
            return self.marginals[index].computeCDF(level)

        return cdf

    @property
    def mean(self):
        """Get the mean of the random variable.

        :return: mean of the random variable.
        :rtype: array
        """
        return array(self.distribution.getMean())

    @property
    def standard_deviation(self):
        """Get the standard deviation of the random variable.

        :return: standard deviation of the random variable.
        :rtype: array
        """
        return array(self.distribution.getStandardDeviation())

    def __create_distributions(
        self, distribution, parameters, transformation, l_b, u_b, threshold
    ):
        """For each variable dimension, instantiate an openturns distribution
        from its class name and parameters.

        :param str distribution: distribution name,
            ie its openturns class name.
        :param parameters: parameters of the openturns distribution
        :param str transformation: standard variable transformation,
            e.g. 'sin(x)'. If None, no transformation.  Default: None.
        :param float l_b: lower bound for truncation. If None, no lower
            truncation. Default: None.
        :param float u_b: upper bound for truncation. If None, no upper
            truncation. Default: None.
        :param float threshold: threshold value in [0,1].
        """
        try:
            ot_dist = getattr(ots, distribution)
        except Exception:
            raise ValueError(
                distribution + " is an unknown openturns" " distribution. "
            )
        try:
            ot_dist = [ot_dist(*parameters)] * self.dimension
        except Exception:
            raise ValueError(
                distribution + " does not take these arguments. "
                "More details on: " + OT_WEBSITE
            )
        self.__set_bounds(ot_dist)
        if transformation is not None:
            ot_dist = self.__transform_marginal_dist(ot_dist, transformation)
        self.__set_bounds(ot_dist)
        if l_b is not None or u_b is not None:
            ot_dist = self.__truncate_marginal_dist(ot_dist, l_b, u_b, threshold)
        self.__set_bounds(ot_dist)
        return ot_dist

    def __transform_marginal_dist(self, marginals, transformation):
        """Apply standard transformations on the marginals, e.g. -, +, *,
        **, sin, exp, log, ...

        :param marginals: marginal distributions.
        :param str transformation: mathematical expression, e.g. 'sin(2*x)'
        """
        variable_name = self.variable_name
        transformation = transformation.replace(" ", "")
        symbolic_function = ots.SymbolicFunction([self.variable_name], [transformation])
        marginals = [
            ots.CompositeDistribution(symbolic_function, marginal)
            for marginal in marginals
        ]
        prev = self.transformation
        transf = transformation.replace(variable_name, "(" + prev + ")")
        self.transformation = transf
        return marginals

    def __truncate_marginal_dist(self, distributions, l_b, u_b, threshold=0.5):
        """Truncate the distribution of a random variable.

        :param distributions: openturns distributions.
        :param float l_b: lower bound for truncation. If None, no lower
            truncation. Default: None.
        :param float u_b: upper bound for truncation. If None, no upper
            truncation. Default: None.
        :param float threshold: threshold value in [0,1].
        :return: transformed openturns distributions.
        """
        marginals = [
            self.__truncate_distribution(dist, index, l_b, u_b, threshold)
            for index, dist in enumerate(distributions)
        ]
        prev = self.transformation
        self.transformation = "Trunc(" + prev + ")"
        return marginals

    def __truncate_distribution(self, distributions, index, l_b, u_b, threshold=0.5):
        """Truncate a distribution with lower bound, upper bound or both.

        :param ot_dist: openturns distributions
        :param float l_b: lower bound value.
        :param float u_b: upper bound value.
        :param float threshold: threshold value in [0,1].
        :return: truncated openturns distributions.
        """
        if l_b is None:
            LOGGER.info(
                "Truncate distribution of component %s" " above %s.", index, u_b
            )
            upper = ots.TruncatedDistribution.UPPER
            current_u_b = self.math_upper_bound[index]
            if u_b > current_u_b:
                raise ValueError("u_b is greater " "than the current upper bound.")
            distributions = ots.TruncatedDistribution(
                distributions, u_b, upper, threshold
            )
        elif u_b is None:
            LOGGER.info(
                "Truncate distribution of component %s" " below %s.", index, l_b
            )
            lower = ots.TruncatedDistribution.LOWER
            current_l_b = self.math_lower_bound[index]
            if l_b < current_l_b:
                raise ValueError("l_b is lower " "than the current lower bound.")
            distributions = ots.TruncatedDistribution(
                distributions, l_b, lower, threshold
            )
        else:
            LOGGER.info(
                "Truncate distribution of component %s" " below %s and above %s.",
                index,
                l_b,
                u_b,
            )
            current_l_b = self.math_lower_bound[index]
            current_u_b = self.math_upper_bound[index]
            if l_b < current_l_b:
                raise ValueError("l_b is lower " "than the current lower bound.")
            if u_b > current_u_b:
                raise ValueError("u_b is greater " "than the current upper bound.")
            distributions = ots.TruncatedDistribution(
                distributions, l_b, u_b, threshold
            )
        return distributions

    def __set_bounds(self, distributions):
        """Set mathematical and numerical bounds (= support and range).

        :param distributions: list of openturns distributions.
        """
        self.math_lower_bound = []
        self.math_upper_bound = []
        self.num_lower_bound = []
        self.num_upper_bound = []
        for distribution in distributions:
            dist_range = distribution.getRange()
            l_b = dist_range.getLowerBound()[0]
            u_b = dist_range.getUpperBound()[0]
            self.num_lower_bound.append(l_b)
            self.num_upper_bound.append(u_b)
            if not dist_range.getFiniteLowerBound()[0]:
                l_b = -inf
            if not dist_range.getFiniteUpperBound()[0]:
                u_b = inf
            self.math_lower_bound.append(l_b)
            self.math_upper_bound.append(u_b)
        self.math_lower_bound = array(self.math_lower_bound)
        self.math_upper_bound = array(self.math_upper_bound)
        self.num_lower_bound = array(self.num_lower_bound)
        self.num_upper_bound = array(self.num_upper_bound)


class OTNormalDistribution(OTDistribution):
    """ Create a normal distribution. """

    def __init__(
        self,
        variable,
        mu=0.0,
        sigma=1.0,
        dimension=1,
        transformation=None,
        l_b=None,
        u_b=None,
        threshold=0.5,
    ):
        """Constructor.

        :param str variable: variable name.
        :param float mu: mean.
        :param float sigma: standard deviation.
        :param int dimension: dimension.
        :param str transformation: standard variable transformation,
            e.g. 'sin(x)'. If None, no transformation.  Default: None.
        :param float l_b: lower bound for truncation. If None, no lower
            truncation. Default: None.
        :param float u_b: upper bound for truncation. If None, no upper
            truncation. Default: None.
        :param float threshold: threshold value in [0,1].
        """
        standard_parameters = {self.MU: mu, self.SIGMA: sigma}
        super(OTNormalDistribution, self).__init__(
            variable,
            "Normal",
            (mu, sigma),
            dimension,
            standard_parameters,
            transformation,
            l_b,
            u_b,
            threshold,
        )


class OTUniformDistribution(OTDistribution):
    """ Create a uniform distribution. """

    def __init__(
        self,
        variable,
        lower=0.0,
        upper=1.0,
        dimension=1,
        transformation=None,
        l_b=None,
        u_b=None,
        threshold=0.5,
    ):
        """Constructor.

        :param str variable: variable name.
        :param float lower: lower bound.
        :param float upper: upper bound.
        :param int dimension: dimension.
        :param str transformation: standard variable transformation,
            e.g. 'sin(x)'. If None, no transformation.  Default: None.
        :param float l_b: lower bound for truncation. If None, no lower
            truncation. Default: None.
        :param float u_b: upper bound for truncation. If None, no upper
            truncation. Default: None.
        :param float threshold: threshold value in [0,1].
        """
        standard_parameters = {self.LOWER: lower, self.UPPER: upper}
        super(OTUniformDistribution, self).__init__(
            variable,
            "Uniform",
            (lower, upper),
            dimension,
            standard_parameters,
            transformation,
            l_b,
            u_b,
            threshold,
        )


class OTTriangularDistribution(OTDistribution):
    """ Create a triangular distribution. """

    def __init__(
        self,
        variable,
        lower=0.0,
        mode=0.5,
        upper=1.0,
        dimension=1,
        transformation=None,
        l_b=None,
        u_b=None,
        threshold=0.5,
    ):
        """Constructor.

        :param str variable: variable name.
        :param float lower: lower bound.
        :param float mode: mode.
        :param float upper: upper bound.
        :param int dimension: dimension.
        :param str transformation: standard variable transformation,
            e.g. 'sin(x)'. If None, no transformation.  Default: None.
        :param float l_b: lower bound for truncation. If None, no lower
            truncation. Default: None.
        :param float u_b: upper bound for truncation. If None, no upper
            truncation. Default: None.
        :param float threshold: threshold value in [0,1].
        """
        standard_parameters = {self.LOWER: lower, self.MODE: mode, self.UPPER: upper}
        super(OTTriangularDistribution, self).__init__(
            variable,
            "Triangular",
            (lower, mode, upper),
            dimension,
            standard_parameters,
            transformation,
            l_b,
            u_b,
            threshold,
        )


class OTExponentialDistribution(OTDistribution):
    """ Create a exponential distribution. """

    def __init__(
        self,
        variable,
        rate=1.0,
        loc=0.0,
        dimension=1,
        transformation=None,
        l_b=None,
        u_b=None,
        threshold=0.5,
    ):
        """Constructor.

        :param str variable: variable name.
        :param float rate: rate parameter.
        :param float loc: location parameter.
        :param int dimension: dimension.
        :param str transformation: standard variable transformation,
            e.g. 'sin(x)'. If None, no transformation.  Default: None.
        :param float l_b: lower bound for truncation. If None, no lower
            truncation. Default: None.
        :param float u_b: upper bound for truncation. If None, no upper
            truncation. Default: None.
        :param float threshold: threshold value in [0,1].
        """
        standard_parameters = {self.RATE: rate, self.LOC: loc}
        super(OTExponentialDistribution, self).__init__(
            variable,
            "Exponential",
            (rate, loc),
            dimension,
            standard_parameters,
            transformation,
            l_b,
            u_b,
            threshold,
        )
