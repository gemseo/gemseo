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

- :meth:`.ParameterSpace.compute_samples`: returns several samples
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
from __future__ import annotations

import collections
import logging
from copy import deepcopy
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import TYPE_CHECKING

from gemseo.uncertainty.distributions.distribution import Distribution

if TYPE_CHECKING:
    from gemseo.core.dataset import Dataset

from numpy import array, ndarray

from gemseo.algos.design_space import DesignSpace, DesignVariable
from gemseo.uncertainty.distributions.composed import ComposedDistribution
from gemseo.uncertainty.distributions.factory import (
    DistributionFactory,
    DistributionParametersType,
)
from gemseo.utils.data_conversion import (
    concatenate_dict_of_arrays_to_array,
    split_array_to_dict_of_arrays,
)
from pathlib import Path

RandomVariable = collections.namedtuple(
    "RandomVariable",
    ["distribution", "size", "parameters"],
    defaults=(1, {}),
)

LOGGER = logging.getLogger(__name__)


class ParameterSpace(DesignSpace):
    """Parameter space."""

    uncertain_variables: list[str]
    """The names of the uncertain variables."""

    distributions: dict[str, Distribution]
    """The marginal probability distributions of the uncertain variables."""

    distribution: ComposedDistribution
    """The joint probability distribution of the uncertain variables."""

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
        hdf_file: str | Path | None = None,
        copula: str = ComposedDistribution.CopulaModel.independent_copula.value,
        name: str | None = None,
    ) -> None:
        """
        Args:
            copula: A name of copula defining the dependency between random variables.
        """  # noqa: D205, D212, D415
        LOGGER.debug("*** Create a new parameter space ***")
        super().__init__(hdf_file=hdf_file, name=name)
        self.uncertain_variables = []
        self.distributions = {}
        self.distribution = None
        if copula not in ComposedDistribution.AVAILABLE_COPULA_MODELS:
            raise ValueError(f"{copula} is not a copula name.")
        self._copula = copula
        self.__distributions_definitions = {}
        # To be defined as:
        # self.__distributions_definitions["u"] = ("SPNormalDistribution", {"mu": 1.})
        # where the first component of the tuple is a distribution name
        # and the second one a mapping of the distribution parameter.
        self.__distribution_family_id = ""

    def is_uncertain(
        self,
        variable: str,
    ) -> bool:
        """Check if a variable is uncertain.

        Args:
            variable: The name of the variable.

        Returns:
            True is the variable is uncertain.
        """
        return variable in self.uncertain_variables

    def is_deterministic(
        self,
        variable: str,
    ) -> bool:
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
        variable: str,
    ) -> None:
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
        name: str,
        distribution: str,
        size: int = 1,
        **parameters: DistributionParametersType,
    ) -> None:
        """Add a random variable from a probability distribution.

        Warnings:
            The probability distributions must have
            the same :class:`~.Distribution.DISTRIBUTION_FAMILY_ID`.
            For instance,
            one cannot mix a random variable
            distributed as a :class:`.OTUniformDistribution` with identifier ``"OT"``
            and a random variable
            distributed as a :class:`.SPNormalDistribution` with identifier ``"SP"``.

        Args:
            name: The name of the random variable.
            distribution: The name of a class
                implementing a probability distribution,
                e.g. ``"OTUniformDistribution"`` or ``"SPDistribution"``.
            size: The dimension of the random variable.
            **parameters: The parameters of the distribution.

        Raises:
            ValueError: When mixing probability distributions from different families,
                e.g. an :class:`.OTDistribution` and a :class:`.SPDistribution`.
        """
        self.__distributions_definitions[name] = (distribution, parameters)
        distribution = DistributionFactory().create(
            distribution, variable=name, dimension=size, **parameters
        )
        distribution_family_id = distribution.__class__.__name__[0:2]
        if self.__distribution_family_id:
            if distribution_family_id != self.__distribution_family_id:
                raise ValueError(
                    f"A parameter space cannot mix {self.__distribution_family_id} "
                    f"and {distribution_family_id} distributions."
                )
        else:
            self.__distribution_family_id = distribution_family_id

        LOGGER.debug("Add the random variable: %s.", name)
        self.distributions[name] = distribution
        self.uncertain_variables.append(name)
        self._build_composed_distribution()
        self.__update_parameter_space(name)

    def _build_composed_distribution(self) -> None:
        """Build the composed distribution from the marginal ones."""
        tmp_marginal = self.distributions[self.uncertain_variables[0]]
        marginals = [self.distributions[name] for name in self.uncertain_variables]
        self.distribution = tmp_marginal._COMPOSED_DISTRIBUTION(marginals, self._copula)

    def get_range(
        self,
        variable: str,
    ) -> list[ndarray]:
        """Return the numerical range of a random variable.

        Args:
            variable: The name of the random variable.

        Returns:
            The range of the components of the random variable.
        """
        return self.distributions[variable].range

    def get_support(
        self,
        variable: str,
    ) -> list[ndarray]:
        """Return the mathematical support of a random variable.

        Args:
            variable: The name of the random variable.

        Returns:
            The support of the components of the random variable.
        """
        return self.distributions[variable].support

    def remove_variable(
        self,
        name: str,
    ) -> None:
        """Remove a variable from the probability space.

        Args:
            name: The name of the variable.
        """
        if name in self.uncertain_variables:
            del self.distributions[name]
            self.uncertain_variables.remove(name)
            if self.uncertain_variables:
                self._build_composed_distribution()
        super().remove_variable(name)

    def compute_samples(
        self,
        n_samples: int = 1,
        as_dict: bool = False,
    ) -> dict[str, ndarray] | ndarray:
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
                split_array_to_dict_of_arrays(
                    data_array, self.variables_sizes, self.uncertain_variables
                )
                for data_array in sample
            ]
        return sample

    def evaluate_cdf(
        self,
        value: dict[str, ndarray],
        inverse: bool = False,
    ) -> dict[str, ndarray]:
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
            if val.ndim == 1:
                if inverse:
                    current_v = distribution.compute_inverse_cdf(val)
                else:
                    current_v = distribution.compute_cdf(val)
            else:
                if inverse:
                    current_v = [
                        distribution.compute_inverse_cdf(sample) for sample in val
                    ]
                else:
                    current_v = [distribution.compute_cdf(sample) for sample in val]

            values[name] = array(current_v)

        return values

    def __check_dict_of_array(
        self,
        obj: Any,
    ) -> None:
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

                if value.shape[-1] != self.variables_sizes[variable]:
                    raise ValueError(error_msg)

                if (value > 1.0).any() or (value < 0.0).any():
                    raise ValueError(error_msg)

    def __str__(self) -> str:
        table = super().get_pretty_table()
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
        decimals: int = 2,
    ) -> str:
        """Return a tabular view of the parameter space.

        This view contains statistical information.

        Args:
            decimals: The number of decimals to print.

        Returns:
            The tabular view.
        """
        table = super().get_pretty_table()
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
        x_vect: ndarray,
        minus_lb: bool = True,
        no_check: bool = False,
        use_dist: bool = False,
        out: ndarray | None = None,
    ) -> ndarray:
        """Unnormalize a normalized vector of the parameter space.

        If ``use_dist`` is True,
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
            minus_lb: Whether to remove the lower bounds at normalization.
            no_check: Whether to check if the components are in :math:`[0,1]`.
            use_dist: Whether to unnormalize the components of the random variables
                with their inverse cumulative probability distributions.
            out: The array to store the unnormalized vector.
                If None, create a new array.

        Returns:
            The unnormalized vector.
        """
        if not use_dist:
            return super().unnormalize_vect(x_vect, no_check=no_check, out=out)

        if x_vect.ndim not in [1, 2]:
            raise ValueError("x_vect must be either a 1D or a 2D NumPy array.")

        return self.__unnormalize_vect(x_vect, no_check)

    def __unnormalize_vect(self, x_vect, no_check):
        data_names = self.variables_names
        data_sizes = self.variables_sizes
        x_u_geom = super().unnormalize_vect(x_vect, no_check=no_check)
        x_u = self.evaluate_cdf(
            split_array_to_dict_of_arrays(x_vect, data_sizes, data_names), inverse=True
        )
        x_u_geom = split_array_to_dict_of_arrays(x_u_geom, data_sizes, data_names)
        missing_names = [name for name in data_names if name not in x_u]
        for name in missing_names:
            x_u[name] = x_u_geom[name]

        return concatenate_dict_of_arrays_to_array(x_u, data_names)

    def transform_vect(  # noqa:D102
        self,
        vector: ndarray,
        out: ndarray | None = None,
    ) -> ndarray:
        return self.normalize_vect(vector, use_dist=True, out=out)

    def untransform_vect(  # noqa:D102
        self,
        vector: ndarray,
        no_check: bool = False,
        out: ndarray | None = None,
    ) -> ndarray:
        return self.unnormalize_vect(vector, use_dist=True, no_check=no_check, out=out)

    def normalize_vect(
        self,
        x_vect: ndarray,
        minus_lb: bool = True,
        use_dist: bool = False,
        out: ndarray | None = None,
    ) -> ndarray:
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
            out: The array to store the normalized vector.
                If None, create a new array.

        Returns:
            The normalized vector.
        """
        if not use_dist:
            return super().normalize_vect(x_vect, out=out)

        if x_vect.ndim not in [1, 2]:
            raise ValueError("x_vect must be either a 1D or a 2D NumPy array.")

        return self.__normalize_vect(x_vect)

    def __normalize_vect(self, x_vect):
        data_names = self.variables_names
        data_sizes = self.variables_sizes
        dict_sample = split_array_to_dict_of_arrays(x_vect, data_sizes, data_names)
        x_n_geom = super().normalize_vect(x_vect)
        x_n = self.evaluate_cdf(dict_sample)
        x_n_geom = split_array_to_dict_of_arrays(x_n_geom, data_sizes, data_names)
        missing_names = [name for name in data_names if name not in x_n]
        for name in missing_names:
            x_n[name] = x_n_geom[name]

        return concatenate_dict_of_arrays_to_array(x_n, data_names)

    @property
    def deterministic_variables(self) -> list[str]:
        """The deterministic variables."""
        return [
            variable
            for variable in self.variables_names
            if variable not in self.uncertain_variables
        ]

    def extract_uncertain_space(
        self,
        as_design_space: bool = False,
    ) -> DesignSpace | ParameterSpace:
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

    def extract_deterministic_space(self) -> DesignSpace:
        """Define a new :class:`.DesignSpace` from the deterministic variables only.

        Return:
            A :class:`.DesignSpace` defined by the deterministic variables only.
        """
        deterministic_space = DesignSpace()
        for name in self.deterministic_variables:
            deterministic_space.add_variable(
                name, self.get_size(name), self.get_type(name)
            )
            value = self._current_value.get(name)
            if value is not None:
                deterministic_space.set_current_variable(name, value)
            deterministic_space.set_lower_bound(name, self.get_lower_bound(name))
            deterministic_space.set_upper_bound(name, self.get_upper_bound(name))
        return deterministic_space

    @staticmethod
    def init_from_dataset(
        dataset: Dataset,
        groups: Iterable[str] | None = None,
        uncertain: Mapping[str, bool] | None = None,
        copula: str = ComposedDistribution.CopulaModel.independent_copula.value,
    ) -> ParameterSpace:
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
                            f"{name}_{idx}",
                            "OTUniformDistribution",
                            minimum=float(l_b[idx]),
                            maximum=float(u_b[idx]),
                        )
                else:
                    parameter_space.add_variable(name, size, "float", l_b, u_b, value)

        return parameter_space

    def to_design_space(self) -> DesignSpace:
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
                value=self.get_current_value([name]),
            )
        return design_space

    def __getitem__(
        self,
        name: str,
    ) -> DesignVariable | RandomVariable:
        if name not in self.variables_names:
            raise KeyError(f"Variable '{name}' is not known.")

        if self.is_uncertain(name):
            return RandomVariable(
                distribution=self.__distributions_definitions[name][0],
                size=self.get_size(name),
                parameters=self.__distributions_definitions[name][1],
            )
        else:
            try:
                value = self.get_current_value([name])
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
        name: str,
        item: DesignVariable | RandomVariable,
    ) -> None:
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

    def rename_variable(  # noqa:D102
        self,
        current_name: str,
        new_name: str,
    ) -> None:
        super().rename_variable(current_name, new_name)

        if current_name in self.uncertain_variables:
            self.uncertain_variables[
                self.uncertain_variables.index(current_name)
            ] = new_name
            self.__distributions_definitions[
                new_name
            ] = self.__distributions_definitions.pop(current_name)
