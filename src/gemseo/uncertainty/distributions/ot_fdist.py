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
Fitting a distribution from data based on OpenTURNS
===================================================

Overview
--------

The :class:`.OTDistributionFitter` class considers several samples
of an uncertain variable, fits a user-defined probability distribution
from this dataset and returns a :class:`.OTDistribution`.
It can also return a goodness-of-fit measure
associated with this distribution,
e.g. Bayesian Information Criterion, Kolmogorov test or Chi Squared test,
or select an optimal distribution among a collection according to
a criterion with a threshold.

Construction
------------

The :class:`.OTDistributionFitter` of a given uncertain variable is built
from only two arguments:

- a variable name,
- a one-dimensional numpy array.

Capabilities
------------

Fit a distribution
~~~~~~~~~~~~~~~~~~

The :meth:`.OTDistributionFitter.fit` method takes a distribution name
recognized by OpenTURNS as argument (e.g. 'Normal', 'Uniform', 'Exponential',
...) as argument and returns an :class:`.OTDistribution`
whose underlying OpenTURNS distribution is the specified one fitted
from the dataset passed to the constructor.

Measure the goodness-of-fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`.OTDistributionFitter.measure` method has two mandatory arguments:

- a distribution which is a either a :class:`.OTDistribution`
  or a distribution name from which :meth:`!fit` method
  builds a :class:`.OTDistribution`,
- a fitting criterion name.

.. note::

   Use the :meth:`.OTDistributionFitter.get_available_criteria` method to get
   the complete list of available criteria
   and the :meth:`.OTDistributionFitter.get_significance_tests` method
   to get the list of available criteria which are significance tests.

The :meth:`.OTDistributionFitter.measure` method can also use a level
associated with the criterion.

The :meth:`.OTDistributionFitter.measure` methods returns a goodness-of-fit
measure whose nature is either a scalar
when the criterion is not a significance test
or a tuple when the criterion is a significance test. In that case,
the first component of the tuple is a boolean indicating if the measured
distribution is acceptable to model the data and the second one is
a dictionary containing the test statistics, the p-value and
the significance level.

Select an optimal distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`.OTDistributionFitter.select` method select aims to select an
optimal distribution among a collection. It uses two mandatory arguments:

- a list of distribution, either a list of distributions names
  or a list of :class:`.OTDistribution`,
- a fitting criterion name.

The :meth:`.OTDistributionFitter.select` method can also use a level
associated with the criterion and a criterion selection:

- 'best': select the distribution minimizing (or maximizing, depending
  on the criterion) the criterion,
- 'first': Select the first distribution for which the criterion is
  greater (or lower, depending on the criterion) than the level.
"""
from __future__ import absolute_import, division, unicode_literals

import openturns as ots
from future import standard_library
from numpy import ndarray
from past.builtins import basestring

from gemseo.uncertainty.distributions.ot_dist import OTDistribution

standard_library.install_aliases()

from gemseo import LOGGER


class OTDistributionFitter(object):
    """ OpenTURNS distribution fitter. """

    AVAILABLE_FACTORIES = {}
    for factory in ots.DistributionFactory.GetContinuousUniVariateFactories():
        factory_class_name = factory.getImplementation().getClassName()
        dist_name = factory_class_name.split("Factory")[0]
        AVAILABLE_FACTORIES[dist_name] = getattr(ots, factory_class_name)

    AVAILABLE_FITTING_TESTS = {
        "BIC": ots.FittingTest.BIC,
        "Kolmogorov": ots.FittingTest.Kolmogorov,
        "ChiSquared": ots.FittingTest.ChiSquared,
    }

    SIGNIFICANCE_TESTS = ["Kolmogorov", "ChiSquared"]

    CRITERIA_TO_MAXIMIZE = []
    CRITERIA_TO_MINIMIZE = ["BIC"]

    def __init__(self, variable, data):
        """Constructor.

        :param str variable: variable name.
        :param array data: data.
        """
        self.variable = variable
        try:
            isinstance(data, ndarray)
            self.data = ots.Sample(data.reshape((-1, 1)))
        except AttributeError:
            raise TypeError("data must be a numpy array")

    def _get_factory(self, distribution):
        """Get distribution factory.

        :param str distribution: distribution name.
        """
        try:
            distribution_factory = self.AVAILABLE_FACTORIES[distribution]
        except KeyError:
            distributions = ", ".join(list(self.AVAILABLE_FACTORIES.keys()))
            raise ValueError(
                "{} is not a name of "
                "distribution available for fitting. "
                "Available ones are: {}.".format(distribution, distributions)
            )
        return distribution_factory

    def _get_fitting_test(self, criterion):
        """Get fitting test.

        :param str criterion: criterion name.
        """
        try:
            fitting_test = self.AVAILABLE_FITTING_TESTS[criterion]
        except KeyError:
            tests = ", ".join(list(self.AVAILABLE_FITTING_TESTS.keys()))
            raise ValueError(
                criterion + " is not a name of fitting test. "
                "Available ones are: " + tests + "."
            )
        return fitting_test

    def fit(self, distribution):
        """Fit a distribution.

        :param str distribution: distribution name.
        """
        factory = self._get_factory(distribution)
        fitted_distribution = factory().build(self.data)
        parameters = fitted_distribution.getParameter()
        distribution = OTDistribution(self.variable, distribution, parameters)
        return distribution

    def measure(self, distribution, criterion, level=0.05):
        """Measure the goodness-of-fit of a distribution to data.

        :param distribution: distribution.
        :type distribution: OTDistribution or str
        :param str fitting_criterion: goodness-of-fit criterion.
        :param float level: risk of committing a Type 1 error,
            that is an incorrect rejection of a true null hypothesis,
            for criteria based on test hypothesis.
            Default: 0.05.
        """
        if isinstance(distribution, basestring):
            distribution = self.fit(distribution)
        if distribution.dimension > 1:
            raise TypeError("A 1D distribution is required.")
        distribution = distribution.marginals[0]
        fitting_test = self._get_fitting_test(criterion)
        if criterion in self.SIGNIFICANCE_TESTS:
            result = fitting_test(self.data, distribution, level)
            details = {
                "p-value": result.getPValue(),
                "statistics": result.getStatistic(),
                "level": level,
            }
            result = (result.getBinaryQualityMeasure(), details)
        else:
            result = fitting_test(self.data, distribution)
        return result

    def select(
        self, distributions, fitting_criterion, level=0.05, selection_criterion="best"
    ):
        """Select the best distribution.

        :param distributions: list of distributions.
        :type distribution: list(OTDistribution) or list(str)
        :param str fitting_criterion: goodness-of-fit criterion.
        :param float level: significance level. For hypothesis tests,
            this is the risk of committing a Type 1 error,
            that is an incorrect rejection of a true null hypothesis.
            For other tests, this is a threshold.
            Default: 0.05.
        :param str selection_criterion: selection criterion
        """
        results = []
        for index, distribution in enumerate(distributions):
            if isinstance(distribution, basestring):
                distribution = self.fit(distribution)
            results.append(self.measure(distribution, fitting_criterion, level))
            distributions[index] = distribution
        index = self.select_from_results(
            results, fitting_criterion, level, selection_criterion
        )
        return distributions[index]

    @classmethod
    def select_from_results(
        cls, results, fitting_criterion, level=0.05, selection_criterion="best"
    ):
        """Select the best distribution from results

        :param list results: results
        :param str fitting_criterion: goodness-of-fit criterion.
        :param float level: significance level. For hypothesis tests,
            this is the risk of committing a Type 1 error,
            that is an incorrect rejection of a true null hypothesis.
            For other tests, this is a threshold.
            Default: 0.05.
        :param str selection_criterion: selection criterion
        """
        if fitting_criterion in cls.SIGNIFICANCE_TESTS:
            for index, _ in enumerate(results):
                results[index] = results[index][1]["p-value"]
            if sum([pval > level for pval in results]) == 0:
                LOGGER.warning(
                    "All criteria values are lower than the " "significance level %s.",
                    level,
                )
        if selection_criterion == "best" or level is None:
            index = cls.__find_opt_distribution(results, fitting_criterion)
        else:
            index = cls.__apply_first_strategy(results, fitting_criterion, level)
        return index

    @classmethod
    def __apply_first_strategy(cls, results, fitting_criterion, level=0.05):
        """Select the index of the best distribution from results
        by applying the "first" strategy.

        :param dict results: results
        :param str fitting_criterion: goodness-of-fit criterion.
        :param float level: significance level. For hypothesis tests,
            this is the risk of committing a Type 1 error,
            that is an incorrect rejection of a true null hypothesis.
            For other tests, this is a threshold.
            Default: 0.05.
        """
        select = False
        index = 0
        for result in results:
            select = result >= level
            if select:
                break
            index += 1
        if not select:
            index = cls.__find_opt_distribution(results, fitting_criterion)
        return index

    @classmethod
    def __find_opt_distribution(cls, results, fitting_criterion):
        """ Returns the index of the optimum distribution."""
        if fitting_criterion in cls.CRITERIA_TO_MINIMIZE:
            index = results.index(min(results))
        else:
            index = results.index(max(results))
        return index

    def get_available_distributions(self):
        """ Get available distributions. """
        return sorted(self.AVAILABLE_FACTORIES.keys())

    def get_available_criteria(self):
        """ Get available goodness-of-fit criteria. """
        return sorted(self.AVAILABLE_FITTING_TESTS.keys())

    def get_significance_tests(self):
        """ Get significance tests. """
        return sorted(self.SIGNIFICANCE_TESTS)
