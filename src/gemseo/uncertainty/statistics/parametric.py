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
Parametric estimation of statistics from a dataset
==================================================

Overview
--------

The :class:`.ParametricStatistics` class inherits from the
abstract :class:`.Statistics` class and aims to estimate statistics
from a :class:`.Dataset`, based on a collection of
candidate parametric distribution calibrated from this :class:`.Dataset`.
For each variable, parameters of these distributions are calibrated
from the :class:`.Dataset`
and the fitted parametric :class:`.Distribution` which is optimal
in the sense of a goodness-of-fit criterion and a selection criterion
is selected to estimate :class:`.Statistics` associated with this variable.
The :class:`.ParametricStatistics` relies on the OpenTURNS library through
the :class:`.OTDistribution` and :class:`.OTDistributionFitter` classes.

Construction
------------

The :class:`.ParametricStatistics` is built from two mandatory arguments:

- a dataset,
- a list of distributions names,

and can consider optional arguments:

- a subset of variables names (by default, statistics are computed
  for all variables),
- a fitting criterion name (by default, BIC is used;
  see :meth:`.ParametricStatistics.get_available_criteria`
  and :meth:`.ParametricStatistics.get_significance_tests`
  for more information),
- a level associated with the fitting criterion,
- a selection criterion:

  - 'best': select the distribution minimizing (or maximizing, depending
    on the criterion) the criterion,
  - 'first': Select the first distribution for which the criterion is
    greater (or lower, depending on the criterion) than the level,

- a name for the :class:`.ParametricStatistics` object (by default,
  the name is the concatenation of 'ParametricStatistics' and
  and the name of the :class:`.Dataset`).

Capabilities
------------

By inheritance, a :class:`.ParametricStatistics` object has the
same capabilities as :class:`.Statistics`. Additional ones are:

- :meth:`.get_fitting_matrix`: this method shows the values
  of the fitting criterion for the different variables and
  candidate probability distributions
  as well as the select probability distribution,
- :meth:`.plot_criteria`: this method plots the criterion values
  for a given variable.
"""
from __future__ import absolute_import, division, unicode_literals

import os

import matplotlib.pyplot as plt
import openturns as ot
from future import standard_library
from numpy import array, exp, inf, linspace, log
from past.utils import old_div

from gemseo.third_party.prettytable.prettytable import PrettyTable
from gemseo.uncertainty.distributions.ot_dist import OTNormalDistribution
from gemseo.uncertainty.distributions.ot_fdist import OTDistributionFitter
from gemseo.uncertainty.statistics.statistics import Statistics

standard_library.install_aliases()

from gemseo import LOGGER


class ParametricStatistics(Statistics):
    """ Parametric estimation of statistics. """

    def __init__(
        self,
        dataset,
        distributions,
        variables_names=None,
        fitting_criterion="BIC",
        level=0.05,
        selection_criterion="best",
        name=None,
    ):
        """Constructor

        :param Dataset dataset: dataset
        :param list(str) distributions: list of distributions names
        :param list(str) variables_names: list of variables names
            or list of variables names. If None, the method considers
            all variables from loaded dataset. Default: None.
        :param str fitting_criterion: goodness-of-fit criterion.
            Default: 'BIC'.
        :param float level: risk of committing a Type 1 error,
            that is an incorrect rejection of a true null hypothesis,
            for criteria based on test hypothesis.
            Default: 0.05.
        :param str selection_criterion: selection criterion. Default: 'best'
        :param str name: name of the object.
            If None, use the concatenation of class and dataset names.
            Default: None.
        """
        super(ParametricStatistics, self).__init__(dataset, variables_names, name)
        significance_tests = OTDistributionFitter.SIGNIFICANCE_TESTS
        self.fitting_criterion = fitting_criterion
        self.selection_criterion = selection_criterion
        LOGGER.info("| Set goodness-of-fit criterion: %s.", fitting_criterion)
        if self.fitting_criterion in significance_tests:
            self.level = level
            LOGGER.info("| Set significance level of hypothesis test: %s.", level)
        else:
            self.level = None
        self.build_distributions(distributions)

    def build_distributions(self, distributions):
        """Build distributions from a list of distributions names,
        a test level and the stored dataset.

        :param list(str) distributions: list of distributions names.
        """
        self._all_distributions = self._fit_distributions(distributions)
        self.distributions = self._select_best_distributions(distributions)

    def get_fitting_matrix(self):
        """Get the fitting matrix. This matrix contains goodness-of-fit
        measures for each pair < variable, distribution >."""
        rownames = sorted(self._all_distributions.keys())
        colnames = list(self._all_distributions[rownames[0]].keys())
        table = PrettyTable(["Variable"] + colnames + ["Selection"])
        for varname in rownames:
            row, _ = self.get_criteria(varname)
            row = [varname] + [row[distribution] for distribution in colnames]
            row = row + [self.distributions[varname]["name"]]
            table.add_row(row)
        return str(table)

    def get_criteria(self, varname):
        """Get criteria for a given variable name.

        :param str varname: variable name.
        """
        varname_dist = self._all_distributions[varname]
        criteria = {
            distribution: result["criterion"]
            for distribution, result in varname_dist.items()
        }
        is_pvalue = False
        significance_tests = OTDistributionFitter.SIGNIFICANCE_TESTS
        if self.fitting_criterion in significance_tests:
            criteria = {
                distribution: result[1]["p-value"]
                for distribution, result in criteria.items()
            }
            is_pvalue = True
        return criteria, is_pvalue

    def plot_criteria(
        self, varname, title=None, save=False, show=True, n_legend_cols=4, directory="."
    ):
        """Plot criteria for a given variable name

        :param str varname: name of the variable
        :param str title: title. Default: None.
        :param bool save: save the plot into a file. Default: False.
        :param bool show: show the plot. Default: True.
        :param int n_legend_cols: number of text columns in the upper legend.
            Default: 4.
        :param str directory: directory absolute or relative path.
            Default: '.'.
        """
        if varname not in self.names:
            raise ValueError(
                varname + " is not a variable of the dataset."
                "Available ones are:" + ", ".join(self.names)
            )
        criteria, is_pvalue = self.get_criteria(varname)
        xvals = []
        yvals = []
        labels = []
        xval = 0
        for distribution, criterion in criteria.items():
            xval += 1
            xvals.append(xval)
            yvals.append(criterion)
            labels.append(distribution)
        plt.subplot(121)
        plt.bar(xvals, yvals, tick_label=labels, align="center")
        if is_pvalue:
            plt.ylabel("p-value from " + self.fitting_criterion + " test")
            plt.axhline(self.level, color="r", linewidth=2.0)
        plt.grid(True, "both")
        plt.subplot(122)
        data = array(self.dataset[varname])
        data_min = min(data)
        data_max = max(data)
        xvals = linspace(data_min, data_max, 1000)
        distributions = self._all_distributions[varname]
        try:
            plt.hist(data, density=True)
        except AttributeError:
            plt.hist(data, normed=True)
        for dist_name, dist_value in distributions.items():
            pdf = dist_value["fitted_distribution"].distribution.computePDF
            yvals = [pdf([xval])[0] for xval in xvals]
            plt.plot(xvals, yvals, label=dist_name, linewidth=2.0)
        plt.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncol=n_legend_cols,
            mode="expand",
            borderaxespad=0.0,
        )
        plt.grid(True, "both")
        if title is not None:
            plt.suptitle(title)
        filename = os.path.join(directory, "criteria.pdf")
        if save:
            plt.savefig(filename)
        if show:
            plt.show()
        plt.close()

    def _select_best_distributions(self, distributions_names):
        """ Select the best distributions for the different variables."""
        LOGGER.info("Select the best distribution for each variable.")
        distributions = {}
        for varname in self.names:
            varname_dist = self._all_distributions[varname]
            criteria = [
                varname_dist[distribution]["criterion"]
                for distribution in distributions_names
            ]
            select_from_results = OTDistributionFitter.select_from_results
            index = select_from_results(
                criteria, self.fitting_criterion, self.level, self.selection_criterion
            )
            name = distributions_names[index]
            value = varname_dist[name]["fitted_distribution"]
            distributions[varname] = {"name": name, "value": value}
            LOGGER.info("| The best distribution for %s is %s.", varname, value)
        return distributions

    def _fit_distributions(self, distributions):
        """Fit distributions for the different dataset marginals
        among a given collection of distributions names.

        :param list(str) distributions: list of distributions names
        """
        dist_list = ", ".join(distributions)
        LOGGER.info(
            "Fit different distributions (%s) per variable "
            "and compute the goodness-of-fit criterion.",
            dist_list,
        )
        results = {}
        for varname in self.names:
            LOGGER.info("| Fit different distributions for %s.", varname)
            dataset = self.dataset[varname]
            results[varname] = self._fit_marginal_distributions(
                varname, dataset, distributions
            )
        return results

    def _fit_marginal_distributions(self, variable, sample, distributions):
        """Fit distributions for a given dataset marginal
        among a given collection of distributions names

        :param str variable: variable name
        :param array dataset: sample
        :param list(str) distributions: list of distributions names
        """
        result = {}
        factory = OTDistributionFitter(variable, sample)
        for distribution in distributions:
            fitted_distribution = factory.fit(distribution)
            test_result = factory.measure(
                fitted_distribution, self.fitting_criterion, self.level
            )
            result[distribution] = {}
            result[distribution]["fitted_distribution"] = fitted_distribution
            result[distribution]["criterion"] = test_result
        return result

    def maximum(self):
        """Get the maximum.

        :return: maximum
        """
        result = {
            name: self.distributions[name]["value"].math_upper_bound
            for name in self.names
        }
        return result

    def mean(self):
        """Get the mean.

        :return: mean
        """
        result = {name: self.distributions[name]["value"].mean for name in self.names}
        return result

    def minimum(self):
        """Get the minimum.

        :return: minimum
        """
        result = {
            name: self.distributions[name]["value"].math_lower_bound
            for name in self.names
        }
        return result

    def probability(self, thresh, greater=False):
        """Compute a probability associated to a threshold.

        :param float thresh: threshold
        :param bool greater: if True, compute the probability the probability
            of exceeding the threshold, if False, compute the reverse.
            Default: True.
        :return: probability
        """
        dist = self.distributions
        if greater:
            result = {
                name: 1 - dist[name]["value"].cdf(thresh[name])[0]
                for name in self.names
            }
        else:
            result = {
                name: dist[name]["value"].cdf(thresh[name])[0] for name in self.names
            }
        return result

    def tolerance_interval(self, coverage, confidence=0.95, side="both"):
        """Compute the tolerance interval (TI) for a given minimum percentage
        of the population and a given confidence level.

        :param float coverage: minimum percentage of belonging to the TI.
        :param float confidence: level of confidence in [0,1]. Default: 0.95.
        :param str side: kind of interval: 'lower' for lower-sided TI,
            'upper' for upper-sided TI and 'both for both-sided TI.
        :return: tolerance limits
        """
        if side not in ["upper", "lower", "both"]:
            raise ValueError(
                "The argument 'side' represents the type"
                "of tolerance bounds. Available ones are: 'upper'"
                "'lower' and 'both'."
            )
        if not 0.0 <= coverage <= 1.0:
            raise ValueError("The argument 'coverage'" " must be number in [0,1].")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("The argument 'confidence'" " must be number in [0,1].")
        limits = {}
        for varname in self.names:
            dist_name = self.distributions[varname]["name"]
            dist = self.distributions[varname]["value"]
            if dist_name == "Normal":
                lower, upper = self._normal_tolerance_interval(
                    dist, coverage, confidence, side
                )
            elif dist_name == "Uniform":
                lower, upper = self._uniform_tolerance_interval(
                    dist, coverage, confidence, side
                )
            elif dist_name == "LogNormal":
                lower, upper = self._lognormal_tolerance_interval(
                    dist, coverage, confidence, side
                )
            elif dist_name == "WeibullMin":
                lower, upper = self._weibull_tolerance_interval(
                    dist, coverage, confidence, side
                )
            elif dist_name == "Exponential":
                lower, upper = self._exponential_tolerance_interval(
                    dist, coverage, confidence, side
                )
            else:
                raise ValueError(
                    "Tolerance interval is not implemented for"
                    'distribution "' + dist_name + '".'
                )
            limits[varname] = (lower, upper)
        return limits

    def _weibull_tolerance_interval(
        self, dist, coverage, confidence=0.95, side="lower"
    ):
        """Compute the tolerance interval (TI) for a given minimum percentage
        of the population and a given confidence level, when data are
        Weibull distributed

        :param WeibullMin dist: OT WeibullMin distribution
        :param float coverage: minimum percentage of belonging to the TI.
        :param float confidence: level of confidence in [0,1]. Default: 0.95.
        :param str side: kind of interval: 'lower' for one-sided lower TI,
            'upper' for one-sided upper TI or 'both' for two-sided TI.
        :return: tolerance limits
        """
        alpha = dist.marginals[0].getParameter()[0]
        beta = dist.marginals[0].getParameter()[1]
        gamma = dist.marginals[0].getParameter()[2]
        x_i = log(beta)
        delta = 1.0 / alpha
        if side == "upper":
            lbd = log(-log(1 - coverage))
            offset = -self.n_samples ** 0.5 * lbd
            dof = self.n_samples - 1
            student = ot.Student(dof, offset)
            upper = delta * student.computeQuantile(confidence)[0]
            upper /= (self.n_samples - 1) ** 0.5
            upper = x_i - upper
            lower = -inf
        elif side == "lower":
            lbd = log(-log(coverage))
            offset = -self.n_samples ** 0.5 * lbd
            dof = self.n_samples - 1
            student = ot.Student(dof, offset)
            lower = delta * student.computeQuantile(1 - confidence)[0]
            lower /= (self.n_samples - 1) ** 0.5
            lower = x_i - lower
            upper = inf
        else:
            coverage = (coverage + 1) / 2.0
            alpha = (1 - confidence) / 2.0
            lbd = log(-log(1 - coverage))
            offset = -self.n_samples ** 0.5 * lbd
            dof = self.n_samples - 1
            student = ot.Student(dof, offset)
            upper = delta * student.computeQuantile(1 - alpha)[0]
            upper /= (self.n_samples - 1) ** 0.5
            upper = x_i - upper
            lbd = log(-log(coverage))
            offset = -self.n_samples ** 0.5 * lbd
            dof = self.n_samples - 1
            student = ot.Student(dof, offset)
            lower = delta * student.computeQuantile(alpha)[0]
            lower /= (self.n_samples - 1) ** 0.5
            lower = x_i - lower
        limits = (array([exp(lower) + gamma]), array([exp(upper) + gamma]))
        return limits

    def _uniform_tolerance_interval(
        self, dist, coverage, confidence=0.95, side="lower"
    ):
        """Compute the tolerance interval (TI) for a given minimum percentage
        of the population and a given confidence level, when data are
        uniformly distributed

        :param Distribution dist: OT uniform distribution
        :param float coverage: minimum percentage of belonging to the TI.
        :param float confidence: level of confidence in [0,1]. Default: 0.95.
        :param str side: kind of interval: 'lower' for one-sided lower TI,
            'upper' for one-sided upper TI or 'both' for two-sided TI.
        :return: tolerance limits
        """
        minimum = dist.marginals[0].getParameter()[0]
        maximum = dist.marginals[0].getParameter()[1]
        if side == "upper":
            upper = (maximum - minimum) * coverage
            upper /= (1 - confidence) ** (1.0 / self.n_samples)
            upper += minimum
            limits = (array([-inf]), array([upper]))
        elif side == "lower":
            lower = (maximum - minimum) * (1 - coverage)
            lower /= confidence ** (1.0 / self.n_samples)
            lower += minimum
            limits = (array([lower]), array([inf]))
        else:
            upper = (maximum - minimum) * (coverage + 1) / 2.0
            upper /= ((1 - confidence) / 2.0) ** (1.0 / self.n_samples)
            upper += minimum
            lower = (maximum - minimum) * (1 - (coverage + 1) / 2.0)
            lower /= (1 - (1 - confidence) / 2.0) ** (1.0 / self.n_samples)
            lower += minimum
            limits = (array([lower]), array([upper]))
        return limits

    def _exponential_tolerance_interval(
        self, dist, coverage, confidence=0.95, side="lower"
    ):
        """Compute the tolerance interval (TI) for a given minimum percentage
        of the population and a given confidence level, when data are
        exponentially distributed

        :param Exponential dist: OT exponential distribution
        :param float coverage: minimum percentage of belonging to the TI.
        :param float confidence: level of confidence in [0,1]. Default: 0.95.
        :param str side: kind of interval: 'lower' for one-sided lower TI,
            'upper' for one-sided upper TI or 'both' for two-sided TI.
        :return: tolerance limits
        """
        lmbda = dist.marginals[0].getParameter()[0]
        gamma = dist.marginals[0].getParameter()[1]
        if side == "upper":
            chisq = ot.ChiSquare(2 * self.n_samples)
            chisq = chisq.computeQuantile(confidence)[0]
            upper = -2 * self.n_samples * log(coverage) * lmbda
            upper /= chisq
            lower = 0.0
        elif side == "lower":
            chisq = ot.ChiSquare(2 * self.n_samples)
            chisq = chisq.computeQuantile(confidence)[0]
            lower = -2 * self.n_samples * log(1 - coverage) * lmbda
            lower /= chisq
            upper = inf
        else:
            coverage = (coverage + 1) / 2.0
            alpha = (1 - confidence) / 2.0
            chisq = ot.ChiSquare(2 * self.n_samples)
            chisq = chisq.computeQuantile(1 - alpha)[0]
            upper = -2 * self.n_samples * log(1 - coverage) * lmbda
            upper /= chisq
            lower = -2 * self.n_samples * log(coverage) * lmbda
            lower /= chisq

        limits = (array([lower + gamma]), array([upper + gamma]))
        return limits

    def _normal_tolerance_interval(self, dist, coverage, confidence=0.95, side="lower"):
        """Compute the tolerance interval (TI) for a given minimum percentage
        of the population and a given confidence level, when data are
        normally distributed

        :param Normal dist: OT normal distribution
        :param float coverage: minimum percentage of belonging to the TI.
        :param float confidence: level of confidence in [0,1]. Default: 0.95.
        :param str side: kind of interval: 'lower' for one-sided lower TI,
            'upper' for one-sided upper TI or 'both' for two-sided TI.
        :return: tolerance limits
        """
        mean = dist.marginals[0].getParameter()[0]
        std = dist.marginals[0].getParameter()[1]
        if side in ["upper", "lower"]:
            z_p = ot.Normal().computeQuantile(coverage)[0]
            delta = z_p * self.n_samples ** 0.5
            dof = self.n_samples - 1
            student = ot.Student(dof, delta)
            student_quantile = student.computeQuantile(confidence)[0]
            tolerance_factor = old_div(student_quantile, self.n_samples ** 0.5)
            if side == "upper":
                upper = mean + tolerance_factor * std
                limits = (array([-inf]), array([upper]))
            else:
                lower = mean - tolerance_factor * std
                limits = (array([lower]), array([inf]))
        else:
            z_p = ot.Normal().computeQuantile((1 + coverage) / 2.0)[0]
            left = (1 + 1.0 / self.n_samples) ** 0.5 * z_p
            chisq = ot.ChiSquare(self.n_samples - 1)
            right = old_div(
                (self.n_samples - 1), chisq.computeQuantile(1 - confidence)[0]
            )
            right = right ** 0.5
            weight = self.n_samples - 3 - chisq.computeQuantile(1 - confidence)[0]
            weight /= 2 * (self.n_samples + 1) ** 2
            weight = (1 + weight) ** 0.5
            tolerance_factor = left * right * weight
            lower = mean - tolerance_factor * std
            upper = mean + tolerance_factor * std
            limits = (array([lower]), array([upper]))
        return limits

    def _lognormal_tolerance_interval(
        self, dist, coverage, confidence=0.95, side="lower"
    ):
        """Compute the tolerance interval (TI) for a given minimum percentage
        of the population and a given confidence level, when data are
        normally distributed

        :param LogNormal dist: OT LogNormal distribution
        :param float coverage: minimum percentage of belonging to the TI.
        :param float confidence: level of confidence in [0,1]. Default: 0.95.
        :param str side: kind of interval: 'lower' for one-sided lower TI,
            'upper' for one-sided upper TI or 'both' for two-sided TI.
        :return: tolerance limits
        """
        dist = OTNormalDistribution(
            "x",
            dist.marginals[0].getParameter()[0],
            dist.marginals[0].getParameter()[1],
        )
        lower, upper = self._normal_tolerance_interval(dist, coverage, confidence, side)
        limits = (exp(lower), exp(upper))
        return limits

    def quantile(self, prob):
        """Get the quantile associated to a given probability.

        :param float prob: probability
        :return: quantile
        :rtype: float or list(float)
        """
        prob = array([prob])
        result = {
            name: self.distributions[name]["value"].inverse_cdf(prob)
            for name in self.names
        }
        return result

    def standard_deviation(self):
        """Get the standard deviation.

        :return: standard deviation
        :rtype: float or list(float)
        """
        result = {
            name: self.distributions[name]["value"].standard_deviation
            for name in self.names
        }
        return result

    def variance(self):
        """Get the variance.

        :return: variance
        :rtype: float or list(float)
        """
        result = {
            name: self.distributions[name]["value"].standard_deviation ** 2
            for name in self.names
        }
        return result

    def moment(self, order):
        """Compute the moment for a given order, either centered or not.

        :param int order: moment index
        :return: moment
        :rtype: float or list(float)
        """
        dist = self.distributions
        result = [
            dist[varname]["value"].distribution.getMoment(order)[0]
            for varname in self.names
        ]
        return result

    def range(self):
        """Get the range of variables.

        :return: range of variables
        """
        result = {}
        for name in self.names:
            dist = self.distributions[name]["value"]
            result[name] = dist.math_upper_bound - dist.math_lower_bound
        return result

    @classmethod
    def get_available_distributions(cls):
        """ Get available distributions. """
        return sorted(OTDistributionFitter.AVAILABLE_FACTORIES.keys())

    @classmethod
    def get_available_criteria(cls):
        """ Get available goodness-of-fit criteria. """
        return sorted(OTDistributionFitter.AVAILABLE_FITTING_TESTS.keys())

    @classmethod
    def get_significance_tests(cls):
        """ Get significance tests. """
        return sorted(OTDistributionFitter.SIGNIFICANCE_TESTS)
