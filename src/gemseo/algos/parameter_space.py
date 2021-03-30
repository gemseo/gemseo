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
Parameter space including both deterministic and uncertain parameters
=====================================================================

Overview
--------

The :class:`.ParameterSpace` class describes a set of parameters of
interest which can be either deterministic or uncertain. This class
inherits from :class:`.DesignSpace`.

Capabilities
------------

The :meth:`.DesignSpace.add_variable` aims to add deterministic
variables from:

- a variable name,
- a variable size (default: 1),
- a variable type (default: float),
- a lower bound (default: - infinity),
- an upper bound (default: + infinity),
- a current value (default: None).

The :meth:`.add_random_variable` aims to add uncertain
variables (a.k.a. random variables) from:

- a variable name,
- a distribution name
  (see :meth:`~gemseo.uncertainty.api.get_available_distributions`),
- a variable size,
- distribution parameters (:code:`parameters` set as
  a tuple of positional arguments for :class:`.OTDistribution`
  or a dictionary of keyword arguments for :class:`.SPDistribution`,
  or keyword arguments for standard probability distribution such
  as :class:`.OTNormalDistribution` and :class:`.OTNormalDistribution`).

The :class:`.ParameterSpace` also provides the following methods:

- :meth:`.get_cdf`: evaluate the cumulative density function
  for the different variables and their different
- :meth:`.get_composed_distribution`: returns the probability distribution
  of an uncertain variable,
- :meth:`.get_marginal_distributions`: returns the marginal probability
  distributions of an uncertain variable,
- :meth:`.get_range` returns the numerical range
  of the different uncertain parameters,
- :meth:`.ParameterSpace.get_sample`: returns several sample
  of the uncertain variables,
- :meth:`.get_support`: returns the mathematical support
  of the different uncertain variables,
- :meth:`.is_uncertain`: checks if a parameter is uncertain,
- :meth:`.is_deterministic`: checks if a parameter is deterministic.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import array, ndarray
from sympy import simplify

from gemseo.algos.design_space import DesignSpace
from gemseo.uncertainty.distributions.distribution import ComposedDistribution
from gemseo.uncertainty.distributions.factory import DistributionFactory
from gemseo.utils.data_conversion import DataConversion

standard_library.install_aliases()

from gemseo import LOGGER


class ParameterSpace(DesignSpace):
    """ Parameter space. """

    INITIAL_DISTRIBUTION = "Initial distribution"
    TRANSFORMATION = "Transformation"
    SUPPORT = "Support"
    MEAN = "Mean"
    STANDARD_DEVIATION = "Standard deviation"
    RANGE = "Range"
    BLANK = ""
    PARAMETER_SPACE = "Parameter space"

    def __init__(
        self,
        print_decimals=2,
        shorten=True,
        copula=ComposedDistribution.INDEPENDENT_COPULA,
    ):
        """Constructor

        :param int print_decimals: number of decimals to print. Default: 2.
        :param bool shorten: if True, simplify the expressions
            of variable transformations. Default: True.
        :param str copula: copula name.
            Default: ComposedDistribution.INDEPENDENT_COPULA.
        """
        LOGGER.info("Create a new parameter space. ")
        super(ParameterSpace, self).__init__()
        self.uncertain_variables = []
        self.marginals = {}
        self.distribution = None
        self._ndecimals = print_decimals
        self._shorten = shorten
        if copula not in ComposedDistribution.AVAILABLE_COPULA:
            raise ValueError("%s is not an available copula." % (copula))
        self.copula = copula

    def is_uncertain(self, variable):
        """Check if a variable is uncertain.

        :param str variable: variable name.
        """
        return variable in self.uncertain_variables

    def is_deterministic(self, variable):
        """Check if a variable is deterministic.

        :param str variable: variable name.
        """
        determistic = set(self.variables_names) - set(self.uncertain_variables)
        return variable in determistic

    def __update_parameter_space(self, variable):
        """Update parameter space.

        :param variable: variable name.
        :type variable: str
        """
        if variable not in self.variables_names:
            l_b = self.marginals[variable].math_lower_bound
            u_b = self.marginals[variable].math_upper_bound
            value = self.marginals[variable].mean
            size = self.marginals[variable].dimension
            self.add_variable(variable, size, "float", l_b, u_b, value)
        else:
            l_b = self.marginals[variable].math_lower_bound
            u_b = self.marginals[variable].math_upper_bound
            value = self.marginals[variable].mean
            self.set_lower_bound(variable, l_b)
            self.set_upper_bound(variable, u_b)
            self.set_current_variable(variable, value)

    def add_random_variable(self, name, distribution, size=1, **parameters):
        """Add a random variable from a distribution

        :param str name: name of the random variable.
        :param str distribution: distribution name.
        :param int size: variable size.
        :param parameters: parameters of the distribution.
        """
        factory = DistributionFactory()
        distribution = factory.create(
            distribution, variable=name, dimension=size, **parameters
        )
        variable = distribution.variable_name
        LOGGER.info("Add the random variable: %s", variable)
        self.marginals[variable] = distribution
        self.uncertain_variables.append(variable)
        self._build_composed_distribution()
        self.__update_parameter_space(variable)

    def _build_composed_distribution(self):
        """ Build the composed distribution from the marginal ones. """
        tmp_marginal = self.marginals[self.uncertain_variables[0]]
        marginals = [self.marginals[name] for name in self.uncertain_variables]
        self.distribution = tmp_marginal.COMPOSED_DISTRIBUTION(marginals, self.copula)

    def get_composed_distribution(self, variable):
        """Get the composed distribution of a random variable.

        :param str variable: variable name.
        """
        return self.marginals[variable].distribution

    def get_marginal_distributions(self, variable):
        """Get the marginal distributions of a random variable.

        :param str variable: variable name.
        """
        return self.marginals[variable].marginals

    def get_range(self, variable):
        """Get the numerical range of a random variable.

        :param str variable: variable name.
        """
        return self.marginals[variable].range

    def get_support(self, variable):
        """Get the mathematical support of a random variable.

        :param str variable: variable name.
        """
        return self.marginals[variable].support

    def remove_variable(self, name):
        """Remove a variable from the probability space.

        :param str name: variable name.
        """
        if name in self.uncertain_variables:
            del self.marginals[name]
            self.uncertain_variables.remove(name)
            self._build_composed_distribution()
        super(ParameterSpace, self).remove_variable(name)

    def set_dependence(self, variables, copula, **options):
        """Set dependence relation between random variables.

        :param list(str) variables: list of variables names.
        :param str copula: copula name.
        :param options: copula options.
        """
        raise NotImplementedError

    def get_sample(self, n_samples=1, as_dict=False):
        """Get sample.

        :param int n_samples: number of samples.
        :param bool as_dict: return a dictionary.
        :return: samples
        :rtype: list(array) or list(dict)
        """
        sample = self.distribution.get_sample(n_samples)
        if as_dict:
            sample = [
                DataConversion.array_to_dict(
                    data_array, self.uncertain_variables, self.variables_sizes
                )
                for data_array in sample
            ]
        return sample

    def get_cdf(self, value, inverse=False):
        """Get the inverse Cumulative Density Function
         values of the different marginals.

        :param dict(array) value: values
        :return: (inverse) CDF values
        :rtype: dict(array)
        """
        if inverse:
            self.__check_dict_of_array(value)
        values = {}
        for name in self.uncertain_variables:
            val = value[name]
            distribution = self.marginals[name]
            if inverse:
                current_v = distribution.inverse_cdf(val)
            else:
                current_v = distribution.cdf(val)
            values[name] = array(current_v)

        return values

    def __check_dict_of_array(self, obj):
        """Check if the object is a dictionary of array.

        :param obj: object to test
        """
        error_msg = "obj must be a dictionary whose keys are the variables "
        error_msg += "names and values are arrays whose dimensions are the "
        error_msg += "variables ones and components are in [0, 1]."
        if not isinstance(obj, dict):
            raise TypeError(error_msg)
        for variable, value in obj.items():
            if variable not in self.uncertain_variables:
                LOGGER.debug(
                    "%s is not defined in the probability space."
                    " Available variables are [%s]."
                    " Use uniform distribution for %s.",
                    variable,
                    ", ".join(self.uncertain_variables),
                    variable,
                )
            else:
                if not isinstance(value, ndarray):
                    raise TypeError(error_msg)
                if len(value.flatten()) != self.variables_sizes[variable]:
                    raise ValueError(error_msg)
                if any(value.flatten() > 1.0) or any(value.flatten() < 0.0):
                    raise ValueError(error_msg)

    def __str__(self, *args, **kwargs):
        """String representation.

        :return: description
        :rtype: str
        """
        table = self.get_pretty_table()
        distribution = []
        transformation = []
        support = []
        mean = []
        std = []
        rnge = []
        for variable in self.variables_names:
            if variable in self.uncertain_variables:
                dist = self.marginals[variable]
                tmp_mean = dist.mean
                tmp_std = dist.standard_deviation
                tmp_range = dist.range
                tmp_support = dist.support
                for dim in range(dist.dimension):
                    distribution.append(str(dist))
                    transformation.append(dist.transformation)
                    if self._shorten:
                        transformation[-1] = str(simplify(transformation[-1]))
                    mean.append(tmp_mean[dim])
                    mean[-1] = round(mean[-1], self._ndecimals)
                    std.append(tmp_std[dim])
                    std[-1] = round(std[-1], self._ndecimals)
                    rnge.append(tmp_range[dim])
                    support.append(tmp_support[dim])
            else:
                for dim in range(self.variables_sizes[variable]):
                    distribution.append(self.BLANK)
                    transformation.append(self.BLANK)
                    mean.append(self.BLANK)
                    std.append(self.BLANK)
                    support.append(self.BLANK)
                    rnge.append(self.BLANK)

        table.add_column(self.INITIAL_DISTRIBUTION, distribution)
        table.add_column(self.TRANSFORMATION, transformation)
        table.add_column(self.SUPPORT, support)
        table.add_column(self.MEAN, mean)
        table.add_column(self.STANDARD_DEVIATION, std)
        table.add_column(self.RANGE, rnge)
        table.title = self.PARAMETER_SPACE
        desc = str(table)
        return desc

    def unnormalize_vect(self, x_vect, minus_lb=True, no_check=False, use_dist=True):
        """Inverse transformation from a unit design vector.
        Unnormalizes a normalized vector of the design space.

        :param array x_vect: design variables.
        :param bool minus_lb: if True, remove lower bounds at normalization.
        :param bool no_check: if True, don't check that values are in [0,1].
        :param bool use_dist: if True, rescale wrt the stats law.
        :return: normalized vector
        :rtype: array
        """
        if not use_dist:
            return super(ParameterSpace, self).unnormalize_vect(x_vect)

        data_names = self.variables_names
        data_sizes = self.variables_sizes
        dict_sample = DataConversion.array_to_dict(x_vect, data_names, data_sizes)
        x_u_geom = super(ParameterSpace, self).unnormalize_vect(x_vect)
        x_u = self.get_cdf(dict_sample, inverse=True)
        x_u_geom = DataConversion.array_to_dict(x_u_geom, data_names, data_sizes)
        missing_names = list(set(data_names) - set(x_u.keys()))
        for name in missing_names:
            x_u[name] = x_u_geom[name]
        x_u = DataConversion.dict_to_array(x_u, data_names)
        return x_u

    def normalize_vect(self, x_vect, minus_lb=True, use_dist=False):
        """Normalizes a vector of the design space.
        Unbounded variables are not normalized.

        :param array x_vect: design variables.
        :param bool minus_lb: if True, remove lower bounds at normalization.
        :param bool no_check: if True, don't check that values are in [0,1].
        :param bool use_dist: if True, rescale wrt the stats law.
        :return: normalized vector
        :rtype: array
        """
        if use_dist:
            return super(ParameterSpace, self).normalize_vect(x_vect)

        data_names = self.variables_names
        data_sizes = self.variables_sizes
        dict_sample = DataConversion.array_to_dict(x_vect, data_names, data_sizes)
        x_u_geom = super(ParameterSpace, self).normalize_vect(x_vect)
        x_u = self.get_cdf(dict_sample, inverse=False)
        x_u_geom = DataConversion.array_to_dict(x_u_geom, data_names, data_sizes)
        missing_names = list(set(data_names) - set(x_u.keys()))
        for name in missing_names:
            x_u[name] = x_u_geom[name]
        x_u = DataConversion.dict_to_array(x_u, data_names)
        return x_u
