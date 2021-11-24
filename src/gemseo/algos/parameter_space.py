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

"""Variable space defining both deterministic and uncertain variables.

Overview
--------

The :class:`.ParameterSpace` class describes a set of parameters of interest
which can be either deterministic or uncertain.
This class inherits from :class:`.DesignSpace`.

Capabilities
------------

The :meth:`.DesignSpace.add_variable` aims to add deterministic variables from:

- a variable name,
- an optional variable size (default: 1),
- an optional variable type (default: float),
- an optional lower bound (default: - infinity),
- an optional upper bound (default: + infinity),
- an optional current value (default: None).

The :meth:`.add_random_variable` aims to add uncertain
variables (a.k.a. random variables) from:

- a variable name,
- a distribution name
  (see :meth:`~gemseo.uncertainty.api.get_available_distributions`),
- an optional variable size,
- optional distribution parameters (:code:`parameters` set as
  a tuple of positional arguments for :class:`.OTDistribution`
  or a dictionary of keyword arguments for :class:`.SPDistribution`,
  or keyword arguments for standard probability distribution such
  as :class:`.OTNormalDistribution` and :class:`.SPNormalDistribution`).

The :class:`.ParameterSpace` also provides the following methods:

- :meth:`.compute_samples`: returns several samples
  of the uncertain variables,
- :meth:`.evaluate_cdf`: evaluate the cumulative density function
  for the different variables and their different
- :meth:`.get_range` returns the numerical range
  of the different uncertain parameters,
- :meth:`.get_support`: returns the mathematical support
  of the different uncertain variables,
- :meth:`.is_uncertain`: checks if a parameter is uncertain,
- :meth:`.is_deterministic`: checks if a parameter is deterministic.
"""
from __future__ import division, unicode_literals

import collections
import logging
import sys
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Union

if TYPE_CHECKING:
    from gemseo.core.dataset import Dataset

from numpy import array, ndarray

from gemseo.algos.design_space import DesignSpace, DesignVariable
from gemseo.uncertainty.distributions.composed import ComposedDistribution
from gemseo.uncertainty.distributions.factory import (
    DistributionFactory,
    DistributionParametersType,
)
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.py23_compat import Path

if sys.version_info < (3, 7, 0):
    RandomVariable = collections.namedtuple(
        "RandomVariable", ["distribution", "size", "parameters"]
    )
    RandomVariable.__new__.__defaults__ = (1, {})
else:
    RandomVariable = collections.namedtuple(
        "RandomVariable",
        ["distribution", "size", "parameters"],
        defaults=(1, {}),
    )

LOGGER = logging.getLogger(__name__)


class ParameterSpace(DesignSpace):
    """Parameter space.

    Attributes:
        uncertain_variables (List(str)): The names of the uncertain variables.
        distributions (Dict(str,Distribution)): The marginal probability distributions
            of the uncertain variables.
        distribution (ComposedDistribution): The joint probability distribution
            of the uncertain variables.
    """

    _INITIAL_DISTRIBUTION = "Initial distribution"
    _TRANSFORMATION = "Transformation"
    _SUPPORT = "Support"
    _MEAN = "Mean"
    _STANDARD_DEVIATION = "Standard deviation"
    _RANGE = "Range"
    _BLANK = ""
    _PARAMETER_SPACE = "Parameter space"

    def __init__(
        self,
        hdf_file=None,  # type: Optional[Union[str,Path]]
        copula=ComposedDistribution._INDEPENDENT_COPULA,  # type: str
        name=None,  # type: Optional[str]
    ):  # type: (...) -> None
        """
        Args:
            copula: A name of copula defining the dependency between random variables.
        """
        LOGGER.debug("*** Create a new parameter space ***")
        super(ParameterSpace, self).__init__(hdf_file=hdf_file, name=name)
        self.uncertain_variables = []
        self.distributions = {}
        self.distribution = None
        if copula not in ComposedDistribution.AVAILABLE_COPULA_MODELS:
            raise ValueError("{} is not a copula name.".format(copula))
        self._copula = copula
        self.__distributions_definitions = {}
        # To be defined as:
        # self.__distributions_definitions["u"] = ("SPNormalDistribution", {"mu": 1.})
        # where the first component of the tuple is a distribution name
        # and the second one a mapping of the distribution parameter.

    def is_uncertain(
        self,
        variable,  # type: str
    ):  # type: (...) -> bool
        """Check if a variable is uncertain.

        Args:
            variable: The name of the variable.

        Returns:
            True is the variable is uncertain.
        """
        return variable in self.uncertain_variables

    def is_deterministic(
        self,
        variable,  # type: str
    ):  # type: (...) -> bool
        """Check if a variable is deterministic.

        Args:
            variable: The name of the variable.

        Returns:
            True is the variable is deterministic.
        """
        deterministic = set(self.variables_names) - set(self.uncertain_variables)
        return variable in deterministic

    def __update_parameter_space(
        self,
        variable,  # type: str
    ):  # type: (...) -> None
        """Update the parameter space with a random variable.

        Args:
            variable: The name of the random variable.
        """
        if variable not in self.variables_names:
            l_b = self.distributions[variable].math_lower_bound
            u_b = self.distributions[variable].math_upper_bound
            value = self.distributions[variable].mean
            size = self.distributions[variable].dimension
            self.add_variable(variable, size, "float", l_b, u_b, value)
        else:
            l_b = self.distributions[variable].math_lower_bound
            u_b = self.distributions[variable].math_upper_bound
            value = self.distributions[variable].mean
            self.set_lower_bound(variable, l_b)
            self.set_upper_bound(variable, u_b)
            self.set_current_variable(variable, value)

    def add_random_variable(
        self,
        name,  # type: str
        distribution,  # type: str
        size=1,  # type: int
        **parameters  # type: DistributionParametersType
    ):  # type: (...) -> None
        """Add a random variable from a probability distribution.

        Args:
            name: The name of the random variable.
            distribution: The name of a class
                implementing a probability distribution,
                e.g. 'OTUniformDistribution' or 'SPDistribution'.
            size: The dimension of the random variable.
            **parameters: The parameters of the distribution.
        """
        self.__distributions_definitions[name] = (distribution, parameters)
        factory = DistributionFactory()
        distribution = factory.create(
            distribution, variable=name, dimension=size, **parameters
        )
        LOGGER.debug("Add the random variable: %s.", name)
        self.distributions[name] = distribution
        self.uncertain_variables.append(name)
        self._build_composed_distribution()
        self.__update_parameter_space(name)

    def _build_composed_distribution(self):  # type: (...) -> None
        """Build the composed distribution from the marginal ones."""
        tmp_marginal = self.distributions[self.uncertain_variables[0]]
        marginals = [self.distributions[name] for name in self.uncertain_variables]
        self.distribution = tmp_marginal._COMPOSED_DISTRIBUTION(marginals, self._copula)

    def get_range(
        self,
        variable,  # type: str
    ):  # type: (...) -> List[ndarray]
        """Return the numerical range of a random variable.

        Args:
            variable: The name of the random variable.

        Returns:
            The range of the components of the random variable.
        """
        return self.distributions[variable].range

    def get_support(
        self,
        variable,  # type: str
    ):  # type: (...) -> List[ndarray]
        """Return the mathematical support of a random variable.

        Args:
            variable: The name of the random variable.

        Returns:
            The support of the components of the random variable.
        """
        return self.distributions[variable].support

    def remove_variable(
        self,
        name,  # type: str
    ):  # type: (...) -> None
        """Remove a variable from the probability space.

        Args:
            name: The name of the variable.
        """
        if name in self.uncertain_variables:
            del self.distributions[name]
            self.uncertain_variables.remove(name)
            if self.uncertain_variables:
                self._build_composed_distribution()
        super(ParameterSpace, self).remove_variable(name)

    def compute_samples(
        self,
        n_samples=1,  # type: int
        as_dict=False,  # type: bool
    ):  # type: (...) -> Union[Dict[str,ndarray],ndarray]
        """Sample the random variables and return the realizations.

        Args:
            n_samples: A number of samples.
            as_dict: The type of the returned object.
                If True, return a dictionary.
                Otherwise, return an array.

        Returns:
            The realizations of the random variables,
                either stored in an array or in a dictionary
                whose values are the names of the random variables
                and the values are the evaluations.
        """
        sample = self.distribution.compute_samples(n_samples)
        if as_dict:
            sample = [
                DataConversion.array_to_dict(
                    data_array, self.uncertain_variables, self.variables_sizes
                )
                for data_array in sample
            ]
        return sample

    def evaluate_cdf(
        self,
        value,  # type: Dict[str,ndarray]
        inverse=False,  # type:bool
    ):  # type: (...) -> Dict[str,ndarray]
        """Evaluate the cumulative density function (or its inverse) of each marginal.

        Args:
            value: The values of the uncertain variables
                passed as a dictionary whose keys are the names of the variables.
            inverse: The type of function to evaluate.
                If True, compute the cumulative density function.
                Otherwise, compute the inverse cumulative density function.

        Returns:
            A dictionary where the keys are the names of the random variables
                and the values are the evaluations.
        """
        if inverse:
            self.__check_dict_of_array(value)
        values = {}
        for name in self.uncertain_variables:
            val = value[name]
            distribution = self.distributions[name]
            if inverse:
                current_v = distribution.compute_inverse_cdf(val)
            else:
                current_v = distribution.compute_cdf(val)
            values[name] = array(current_v)

        return values

    def __check_dict_of_array(
        self,
        obj,  # type: Any
    ):  # type: (...) -> None
        """Check if the object is a dictionary whose values are numpy arrays.

        Args:
            obj: The object to test.
        """
        error_msg = (
            "obj must be a dictionary whose keys are the variables "
            "names and values are arrays "
            "whose dimensions are the variables ones and components are in [0, 1]."
        )
        if not isinstance(obj, dict):
            raise TypeError(error_msg)
        for variable, value in obj.items():
            if variable not in self.uncertain_variables:
                LOGGER.debug(
                    "%s is not defined in the probability space; "
                    "available variables are [%s]; "
                    "use uniform distribution for %s.",
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

    def __str__(self):  # type: (...) -> str
        table = super(ParameterSpace, self).get_pretty_table()
        distribution = []
        for variable in self.variables_names:
            if variable in self.uncertain_variables:
                dist = self.distributions[variable]
                for _ in range(dist.dimension):
                    distribution.append(str(dist))
            else:
                for _ in range(self.variables_sizes[variable]):
                    distribution.append(self._BLANK)

        table.add_column(self._INITIAL_DISTRIBUTION, distribution)
        table.title = self._PARAMETER_SPACE
        desc = str(table)
        return desc

    def get_tabular_view(
        self,
        decimals=2,  # type: int
    ):  # type: (...) -> str
        """Return a tabular view of the parameter space.

        This view contains statistical information.

        Args:
            decimals: The number of decimals to print.

        Returns:
            The tabular view.
        """
        table = super(ParameterSpace, self).get_pretty_table()
        distribution = []
        transformation = []
        support = []
        mean = []
        std = []
        rnge = []
        for variable in self.variables_names:
            if variable in self.uncertain_variables:
                dist = self.distributions[variable]
                tmp_mean = dist.mean
                tmp_std = dist.standard_deviation
                tmp_range = dist.range
                tmp_support = dist.support
                for dim in range(dist.dimension):
                    distribution.append(str(dist))
                    transformation.append(dist.transformation)
                    mean.append(tmp_mean[dim])
                    mean[-1] = round(mean[-1], decimals)
                    std.append(tmp_std[dim])
                    std[-1] = round(std[-1], decimals)
                    rnge.append(tmp_range[dim])
                    support.append(tmp_support[dim])
            else:
                for _ in range(self.variables_sizes[variable]):
                    distribution.append(self._BLANK)
                    transformation.append(self._BLANK)
                    mean.append(self._BLANK)
                    std.append(self._BLANK)
                    support.append(self._BLANK)
                    rnge.append(self._BLANK)

        table.add_column(self._INITIAL_DISTRIBUTION, distribution)
        table.add_column(self._TRANSFORMATION, transformation)
        table.add_column(self._SUPPORT, support)
        table.add_column(self._MEAN, mean)
        table.add_column(self._STANDARD_DEVIATION, std)
        table.add_column(self._RANGE, rnge)
        table.title = self._PARAMETER_SPACE
        desc = str(table)
        return desc

    def unnormalize_vect(
        self,
        x_vect,  # type:ndarray
        minus_lb=True,  # type:bool
        no_check=False,  # type: bool
        use_dist=False,  # type:bool
    ):  # type: (...) ->ndarray
        """Unnormalize a normalized vector of the parameter space.

        If `use_dist` is True,
        use the inverse cumulative probability distributions of the random variables
        to unscale the components of the random variables.
        Otherwise,
        use the approach defined in :meth:`.DesignSpace.unnormalize_vect`
        with `minus_lb` and `no_check`.

        For the components of the deterministic variables,
        use the approach defined in :meth:`.DesignSpace.unnormalize_vect`
        with `minus_lb` and `no_check`.

        Args:
            x_vect: The values of the design variables.
            minus_lb: If True, remove the lower bounds at normalization.
            no_check: If True, do not check that the values are in [0,1].
            use_dist: If True, unnormalize the components of the random variables
                with their inverse cumulative probability distributions.

        Returns:
            The unnormalized vector.
        """
        if not use_dist:
            return super(ParameterSpace, self).unnormalize_vect(x_vect)

        data_names = self.variables_names
        data_sizes = self.variables_sizes
        dict_sample = DataConversion.array_to_dict(x_vect, data_names, data_sizes)
        x_u_geom = super(ParameterSpace, self).unnormalize_vect(x_vect)
        x_u = self.evaluate_cdf(dict_sample, inverse=True)
        x_u_geom = DataConversion.array_to_dict(x_u_geom, data_names, data_sizes)
        missing_names = list(set(data_names) - set(x_u.keys()))
        for name in missing_names:
            x_u[name] = x_u_geom[name]
        x_u = DataConversion.dict_to_array(x_u, data_names)
        return x_u

    def transform_vect(
        self, vector  # type: ndarray
    ):  # type:(...) -> ndarray
        return self.normalize_vect(vector, use_dist=True)

    def untransform_vect(
        self, vector  # type: ndarray
    ):  # type:(...) -> ndarray
        return self.unnormalize_vect(vector, use_dist=True)

    def normalize_vect(
        self,
        x_vect,  # type:ndarray
        minus_lb=True,  # type: bool
        use_dist=False,  # type: bool
    ):  # type: (...) ->ndarray
        """Normalize a vector of the parameter space.

        If `use_dist` is True,
        use the cumulative probability distributions of the random variables
        to scale the components of the random variables between 0 and 1.
        Otherwise,
        use the approach defined in :meth:`.DesignSpace.normalize_vect`
        with `minus_lb`.

        For the components of the deterministic variables,
        use the approach defined in :meth:`.DesignSpace.normalize_vect`
        with `minus_lb`.

        Args:
            x_vect: The values of the design variables.
            minus_lb: If True, remove the lower bounds at normalization.
            use_dist: If True, normalize the components of the random variables
                with their cumulative probability distributions.

        Returns:
            The normalized vector.
        """
        if not use_dist:
            return super(ParameterSpace, self).normalize_vect(x_vect)

        data_names = self.variables_names
        data_sizes = self.variables_sizes
        dict_sample = DataConversion.array_to_dict(x_vect, data_names, data_sizes)
        x_u_geom = super(ParameterSpace, self).normalize_vect(x_vect)
        x_u = self.evaluate_cdf(dict_sample, inverse=False)
        x_u_geom = DataConversion.array_to_dict(x_u_geom, data_names, data_sizes)
        missing_names = list(set(data_names) - set(x_u.keys()))
        for name in missing_names:
            x_u[name] = x_u_geom[name]
        x_u = DataConversion.dict_to_array(x_u, data_names)
        return x_u

    @property
    def deterministic_variables(self):  # type: (...) -> List[str]
        """The deterministic variables."""
        return [
            variable
            for variable in self.variables_names
            if variable not in self.uncertain_variables
        ]

    def extract_uncertain_space(
        self,
        as_design_space=False,  # type: bool
    ):  # type: (...) -> Union[DesignSpace,ParameterSpace]
        """Define a new :class:`.DesignSpace` from the uncertain variables only.

        Args:
            as_design_space: If False,
                return a :class:`.ParameterSpace`
                containing the original uncertain variables as is;
                otherwise,
                return a :class:`.DesignSpace`
                where the original uncertain variables are made deterministic.
                In that case,
                the bounds of a deterministic variable correspond
                to the limits of the support of the original probability distribution
                and the current value correspond to its mean.

        Return:
            A :class:`.ParameterSpace` defined by the uncertain variables only.
        """
        uncertain_space = deepcopy(self).filter(self.uncertain_variables)
        if as_design_space:
            return uncertain_space.to_design_space()

        return uncertain_space

    def extract_deterministic_space(self):  # type: (...) -> DesignSpace
        """Define a new :class:`.DesignSpace` from the deterministic variables only.

        Return:
            A :class:`.DesignSpace` defined by the deterministic variables only.
        """
        deterministic_space = DesignSpace()
        for name in self.deterministic_variables:
            deterministic_space.add_variable(
                name, self.get_size(name), self.get_type(name)
            )
            value = self._current_x.get(name)
            if value is not None:
                deterministic_space.set_current_variable(name, value)
            deterministic_space.set_lower_bound(name, self.get_lower_bound(name))
            deterministic_space.set_upper_bound(name, self.get_upper_bound(name))
        return deterministic_space

    @staticmethod
    def init_from_dataset(
        dataset,  # type: Dataset
        groups=None,  # type: Optional[Iterable[str]]
        uncertain=None,  # type: Optional[Mapping[str,bool]]
        copula=ComposedDistribution._INDEPENDENT_COPULA,  # type: str
    ):  # type: (...) -> ParameterSpace
        """Initialize the parameter space from a dataset.

        Args:
            dataset: The dataset used for the initialization.
            groups: The groups of the dataset to be considered.
                If None, consider all the groups.
            uncertain: Whether the variables should be uncertain or not.
            copula: A name of copula defining the dependency between random variables.
        """
        parameter_space = ParameterSpace(copula=copula)

        if uncertain is None:
            uncertain = {}

        if groups is None:
            groups = dataset.groups
        for group in groups:
            for name in dataset.get_names(group):
                data = dataset.get_data_by_names(name)[name]
                l_b = data.min(0)
                u_b = data.max(0)
                value = (l_b + u_b) / 2
                size = len(l_b)

                if uncertain.get(name, False):
                    for idx in range(size):
                        parameter_space.add_random_variable(
                            "{}_{}".format(name, idx),
                            "OTUniformDistribution",
                            1,
                            minimum=float(l_b[idx]),
                            maximum=float(u_b[idx]),
                        )
                else:
                    parameter_space.add_variable(name, size, "float", l_b, u_b, value)

        return parameter_space

    def to_design_space(self):  # type: (...) -> DesignSpace
        """Convert the parameter space into a :class:`.DesignSpace`.

        The original deterministic variables are kept as is
        while the original uncertain variables are made deterministic.
        In that case,
        the bounds of a deterministic variable correspond
        to the limits of the support of the original probability distribution
        and the current value correspond to its mean.

        Return:
            A :class:`.DesignSpace` where all original variables are made deterministic.
        """
        design_space = self.extract_deterministic_space()
        for name in self.uncertain_variables:
            design_space.add_variable(
                name,
                size=self.get_size(name),
                var_type=self.get_type(name),
                l_b=self.get_lower_bound(name),
                u_b=self.get_upper_bound(name),
                value=self.get_current_x([name]),
            )
        return design_space

    def __getitem__(
        self,
        name,  # type: str
    ):  # type: (...) -> Union[DesignVariable, RandomVariable]
        if name not in self.variables_names:
            raise KeyError("Variable '{}' is not known.".format(name))

        if self.is_uncertain(name):
            return RandomVariable(
                distribution=self.__distributions_definitions[name][0],
                size=self.get_size(name),
                parameters=self.__distributions_definitions[name][1],
            )
        else:
            try:
                value = self.get_current_x([name])
            except KeyError:
                value = None

            return DesignVariable(
                size=self.get_size(name),
                var_type=self.get_type(name),
                l_b=self.get_lower_bound(name),
                u_b=self.get_upper_bound(name),
                value=value,
            )

    def __setitem__(
        self,
        name,  # type: str
        item,  # type: Union[DesignVariable, RandomVariable]
    ):  # type: (...) -> None
        if isinstance(item, RandomVariable):
            self.add_random_variable(
                name, item.distribution, size=item.size, **item.parameters
            )
        else:
            self.add_variable(
                name,
                size=item.size,
                var_type=item.var_type,
                l_b=item.l_b,
                u_b=item.u_b,
                value=item.value,
            )
