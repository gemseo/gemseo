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

## Overview

The [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace] class describes
a set of parameters of interest
which can be either deterministic or uncertain.
This class inherits from [DesignSpace][gemseo.algos.design_space.DesignSpace].

## Capabilities

The
[DesignSpace.add_variable()][gemseo.algos.design_space.DesignSpace.add_variable]
aims to add deterministic variables from:

- a variable name,
- an optional variable size (default: 1),
- an optional variable type (default: float),
- an optional lower bound (default: - infinity),
- an optional upper bound (default: + infinity),
- an optional current value (default: None).

The
[add_random_variable()][gemseo.algos.parameter_space.ParameterSpace.add_random_variable]
method
aims to add uncertain
variables (a.k.a. random variables) from:

- a variable name,
- a distribution name
  (see [get_available_distributions()][gemseo.uncertainty.get_available_distributions]),
- an optional variable size,
- optional distribution parameters (`parameters` set as
  a tuple of positional arguments for
  [OTDistribution][gemseo.uncertainty.distributions.openturns.distribution.OTDistribution]
  or a dictionary of keyword arguments for
  [SPDistribution][gemseo.uncertainty.distributions.scipy.distribution.SPDistribution],
  or keyword arguments for standard probability distribution such
  as
  [OTNormalDistribution][gemseo.uncertainty.distributions.openturns.normal.OTNormalDistribution]
  and
  [SPNormalDistribution][gemseo.uncertainty.distributions.scipy.normal.SPNormalDistribution]).

The [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace] also provides
the following methods:

- [compute_samples()][gemseo.algos.parameter_space.ParameterSpace.compute_samples]:
  returns several samples of the uncertain variables,
- [evaluate_cdf()][gemseo.algos.parameter_space.ParameterSpace.evaluate_cdf]:
  evaluate the cumulative density function for the different variables component-wise,
- [get_range()][gemseo.algos.parameter_space.ParameterSpace.get_range]:
  returns the numerical range of the different uncertain parameters,
- [get_support()][gemseo.algos.parameter_space.ParameterSpace.get_support]:
  returns the mathematical support of the different uncertain variables,
- [is_uncertain()][gemseo.algos.parameter_space.ParameterSpace.is_uncertain]:
  checks if a parameter is uncertain,
- [is_deterministic()][gemseo.algos.parameter_space.ParameterSpace.is_uncertain]:
  checks if a parameter is deterministic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

from prettytable import PrettyTable

from gemseo.uncertainty.distributions.factory import DISTRIBUTION_FACTORY
from gemseo.uncertainty.distributions.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.utils.string_tools import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.datasets.dataset import Dataset
    from gemseo.typing import RealArray
    from gemseo.uncertainty.distributions.base_joint import BaseJointDistribution
    from gemseo.uncertainty.distributions.base_settings import BaseDistributionSettings
    from gemseo.utils.pydantic import BaseSettings

from numpy import array
from numpy import ndarray

from gemseo.algos.design_space import DesignSpace
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.string_tools import _format_value_in_pretty_table_16

LOGGER = logging.getLogger(__name__)


class ParameterSpace(DesignSpace):
    """Parameter space."""

    uncertain_variables: list[str]
    """The names of the uncertain variables."""

    distributions: dict[str, BaseJointDistribution]
    """The marginal probability distributions of the uncertain variables.

    These variables are defined as random vectors with independent components.
    """

    distribution: BaseJointDistribution | None
    """The joint probability distribution of the uncertain variables, if any."""

    _INITIAL_DISTRIBUTION = "Initial distribution"
    _TRANSFORMATION = "Transformation"
    _SUPPORT = "Support"
    _MEAN = "Mean"
    _STANDARD_DEVIATION = "Standard deviation"
    _RANGE = "Range"
    _BLANK = ""
    _PARAMETER_SPACE = "Parameter space"

    __random_vector_name_to_settings: dict[str, tuple[BaseSettings, ...]]
    """The map from a random vector name
    to the settings of its marginal distributions."""

    __copulas: list[tuple[tuple[str, ...], Any]]
    """The independent copulas defined by blocks of random variables."""

    __supports_dependency: bool
    """Whether the wrapped UQ library supports dependent variables."""

    __distribution_library_name: str
    """The name of the library implementing the probability distributions."""

    def __init__(self, name: str = "") -> None:  # noqa:D107
        super().__init__(name=name)
        self.uncertain_variables = []
        self.distributions = {}
        self.distribution = None
        self.__random_vector_name_to_settings = {}
        self.__copulas = []
        self.__distribution_library_name = ""
        self.__supports_dependency = True

    def add_copula(self, copula: Any, *names: str) -> None:
        """Add a copula defining the dependency structure between random variables.

        This function can be called several times in order to add several copulas
        associated to different random variables.
        All the variables which are not linked through any copula will be
        independent.

        Args:
            copula: The copula.
            *names: The names of the random variables.

        Raises:
            ValueError: When the joint probability distribution does not support
                dependent random variables,
                i.e. when the joint distribution settings do not have a `copula` field,
                when there is no variable with that name,
                or when there is already a copula for one of the random variables.
        """
        for name in names:
            if name not in self.uncertain_variables:
                msg = f"There is no variable name {name!r}."
                raise ValueError(msg)

            for existing_names, _ in self.__copulas:
                if name in existing_names:
                    msg = f"The random variable {name!r} has already a copula."
                    raise ValueError(msg)

        if not self.__supports_dependency:
            msg = (
                f"{self.distribution.__class__.__name__} does not support "
                "dependent variables."
            )
            raise ValueError(msg)

        self.__copulas.append((names, copula))
        self.__set_joint_distribution()

    def __set_joint_distribution(self) -> None:
        """Set the joint probability distribution."""
        if not self.uncertain_variables:
            return

        marginal_settings = []
        for settings in self.__random_vector_name_to_settings.values():
            marginal_settings.extend(settings)

        marginal_class_name = marginal_settings[0].target_class_name
        marginal_class = DISTRIBUTION_FACTORY.get_class(marginal_class_name)
        joint_class = marginal_class.JOINT_DISTRIBUTION_CLASS
        settings_class = joint_class.settings_class
        if self.__copulas:
            new_copulas = []
            uncertain_names = self.uncertain_variables
            variable_sizes = self.variable_sizes
            for variable_names, copula in self.__copulas:
                indices = []
                for variable_name in variable_names:
                    pos = sum(
                        variable_sizes[uncertain_names[i]]
                        for i in range(uncertain_names.index(variable_name))
                    )
                    indices.extend(range(pos, pos + variable_sizes[variable_name]))

                new_copulas.append((indices, copula))

            settings = settings_class(
                marginal_settings=marginal_settings,
                copula=new_copulas,
            )
        else:
            settings = settings_class(marginal_settings=marginal_settings)

        self.distribution = joint_class(settings)

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
        deterministic = self._variables.keys() - set(self.uncertain_variables)
        return variable in deterministic

    def add_random_vector(
        self,
        name: str,
        *settings: BaseDistributionSettings,
    ) -> None:
        """Add a random vector.

        Args:
            name: The name of the vector.
            *settings: The settings of the marginal probability distributions.

        Raises:
            ValueError: When mixing probability distributions from different families,
                e.g. an
                [OTDistribution][gemseo.uncertainty.distributions.openturns.distribution.OTDistribution]
                and a
                [SPDistribution][gemseo.uncertainty.distributions.scipy.distribution.SPDistribution]
                or
                when the lengths of the distribution parameter collections
                are not consistent.
        """
        self._check_variable_name(name)
        distribution_library_names = {s._LIBRARY_NAME for s in settings}
        if self.__distribution_library_name:
            distribution_library_names.add(self.__distribution_library_name)
        if len(distribution_library_names) > 1:
            msg = (
                "A parameter space cannot mix probability distributions "
                "based on different libraries; "
                f"got {pretty_repr(distribution_library_names, use_and=True)}."
            )
            raise ValueError(msg)

        if len(self.__random_vector_name_to_settings) == 0:
            marginal_class_name = settings[0].target_class_name
            marginal_class = DISTRIBUTION_FACTORY.get_class(marginal_class_name)
            joint_class = marginal_class.JOINT_DISTRIBUTION_CLASS
            settings_class = joint_class.settings_class
            self.__supports_dependency = "copula" in settings_class.__fields__

        marginals = [DISTRIBUTION_FACTORY.create_from_settings(s) for s in settings]
        self.__distribution_library_name = next(iter(distribution_library_names))
        self.__random_vector_name_to_settings[name] = settings

        # Define the distribution of the random vector with a joint distribution.
        cls = marginals[0].JOINT_DISTRIBUTION_CLASS
        distribution = cls(cls.settings_class(marginal_settings=settings))
        self.distributions[name] = distribution

        # Update the uncertain variables.
        self.uncertain_variables.append(name)

        # Create the joint distribution.
        self.__set_joint_distribution()

        # Update the parameter space as subclass of a DesignSpace.
        self.add_variable(
            name,
            distribution.dimension,
            self.DesignVariableType.FLOAT,
            distribution.math_lower_bound,
            distribution.math_upper_bound,
            distribution.mean,
        )

    def add_random_variable(
        self,
        name: str,
        distribution_settings: BaseDistributionSettings,
        size: int = 1,
    ) -> None:
        """Add a random variable.

        Args:
            name: The name of the variable.
            distribution_settings: The settings
                of the marginal probability distribution.
                This distribution is common to all components of the variable.
            size: The dimension of the variable.
        """
        self.add_random_vector(name, *[distribution_settings] * size)

    def get_range(
        self,
        variable: str,
    ) -> RealArray:
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
    ) -> RealArray:
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

        This method also removes the copulas associated with this variable.

        Args:
            name: The name of the variable.
        """
        if name in self.uncertain_variables:
            del self.distributions[name]
            del self.__random_vector_name_to_settings[name]
            for copula in list(self.__copulas):
                if name in copula[0]:
                    self.__copulas.remove(copula)

            self.uncertain_variables.remove(name)
            self.__set_joint_distribution()
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
                If `True`, return a dictionary.
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
                    data_array, self.variable_sizes, self.uncertain_variables
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
                If `True`, compute the cumulative density function.
                Otherwise, compute the inverse cumulative density function.

        Returns:
            A dictionary where the keys are the names of the random variables
            and the values are the evaluations.
        """
        if inverse:
            self.__check_dict_of_array(value)

        method_name = "compute_inverse_cdf" if inverse else "compute_cdf"
        values = {}
        for name in self.uncertain_variables:
            input_samples = value[name]
            compute = getattr(self.distributions[name], method_name)
            if input_samples.ndim == 1:
                output_samples = compute(input_samples)
            else:
                output_samples = list(map(compute, input_samples))

            values[name] = array(output_samples)

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

                if value.shape[-1] != self._variables[variable].size:
                    raise ValueError(error_msg)

                if (value > 1.0).any() or (value < 0.0).any():
                    raise ValueError(error_msg)

    def get_pretty_table(  # noqa: D102
        self,
        fields: Sequence[str] = (),
        with_index: bool = False,
        capitalize: bool = False,
        simplify: bool = False,
    ) -> PrettyTable:
        if not simplify or self.deterministic_variables or fields:
            table = super().get_pretty_table(
                fields=fields, capitalize=capitalize, with_index=with_index
            )
        else:
            table = PrettyTable(["Name" if capitalize else "name"])
            table.custom_format = _format_value_in_pretty_table_16
            for name, variable in self._variables.items():
                name_template = f"{name}"
                if with_index and variable.size > 1:
                    name_template += "[{index}]"

                for i in range(variable.size):
                    table.add_row([name_template.format(name=name, index=i)])

        distributions = []
        transformations = []
        for variable in self:
            if variable in self.uncertain_variables:
                for marginal in self.distributions[variable].marginals:
                    distributions.append(repr(marginal))
                    transformations.append(marginal.transformation)
            else:
                empty = [self._BLANK] * self._variables[variable].size
                distributions.extend(empty)
                transformations.extend(empty)

        if self.uncertain_variables:
            default_variable_name = (
                self
                .distributions[self.uncertain_variables[0]]
                .marginals[0]
                .DEFAULT_VARIABLE_NAME
            )
            add_transformation = False
            for transformation in transformations:
                if transformation not in {default_variable_name, self._BLANK}:
                    add_transformation = True
                    break

            if add_transformation:
                table.add_column(
                    "Initial distribution" if capitalize else "initial distribution",
                    distributions,
                )
                table.add_column(
                    "Transformation(x)=" if capitalize else "transformation(x)=",
                    transformations,
                )
            else:
                table.add_column(
                    "Distribution" if capitalize else "distribution", distributions
                )

        return table

    def __str__(self) -> str:
        title = "" if self.deterministic_variables else "Uncertain space"
        return self._get_string_representation(False, simplify=True, title=title)

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
        for variable in self:
            if variable in self.uncertain_variables:
                joint_distribution = self.distributions[variable]
                joint_mean = joint_distribution.mean
                joint_std = joint_distribution.standard_deviation
                joint_range = joint_distribution.range
                joint_support = joint_distribution.support
                for i, marginal in enumerate(joint_distribution.marginals):
                    distribution.append(str(marginal))
                    transformation.append(marginal.transformation)
                    mean.append(round(joint_mean[i], decimals))
                    std.append(round(joint_std[i], decimals))
                    rnge.append(joint_range[i])
                    support.append(joint_support[i])
            else:
                for _ in range(self.variable_sizes[variable]):
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
        return str(table)

    def unnormalize_vect(
        self,
        x_vect: ndarray,
        minus_lb: bool = True,
        no_check: bool = False,
        use_dist: bool = False,
        out: ndarray | None = None,
    ) -> ndarray:
        """Unnormalize a normalized vector of the parameter space.

        If `use_dist` is True,
        use the inverse cumulative probability distributions of the random variables
        to unscale the components of the random variables.
        Otherwise,
        use the approach defined in
        [DesignSpace.unnormalize_vect()][gemseo.algos.design_space.DesignSpace.unnormalize_vect]
        with `minus_lb` and `no_check`.

        For the components of the deterministic variables,
        use the approach defined in
        [DesignSpace.unnormalize_vect()][gemseo.algos.design_space.DesignSpace.unnormalize_vect]
        with `minus_lb` and `no_check`.

        Args:
            x_vect: The values of the design variables.
            minus_lb: Whether to remove the lower bounds at normalization.
            no_check: Whether to check if the components are in $[0,1]$.
            use_dist: Whether to unnormalize the components of the random variables
                with their inverse cumulative probability distributions.
            out: The array to store the unnormalized vector.
                If `None`, create a new array.

        Returns:
            The unnormalized vector.
        """
        if not use_dist:
            return super().unnormalize_vect(x_vect, no_check=no_check, out=out)

        if x_vect.ndim not in {1, 2}:
            msg = "x_vect must be either a 1D or a 2D NumPy array."
            raise ValueError(msg)

        return self.__unnormalize_vect(x_vect, no_check)

    def __unnormalize_vect(self, x_vect, no_check):
        data_names = self._variables.keys()
        data_sizes = self.variable_sizes
        x_u_geom = super().unnormalize_vect(x_vect, no_check=no_check)
        x_u = self.evaluate_cdf(
            split_array_to_dict_of_arrays(x_vect, data_sizes, data_names), inverse=True
        )
        x_u_geom = split_array_to_dict_of_arrays(x_u_geom, data_sizes, data_names)
        missing_names = [name for name in self if name not in x_u]
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
        use the approach defined in
        [DesignSpace.normalize_vect()][gemseo.algos.design_space.DesignSpace.normalize_vect]
        with `minus_lb`.

        For the components of the deterministic variables,
        use the approach defined in
        [DesignSpace.normalize_vect()][gemseo.algos.design_space.DesignSpace.normalize_vect]
        with `minus_lb`.

        Args:
            x_vect: The values of the design variables.
            minus_lb: If `True`, remove the lower bounds at normalization.
            use_dist: If `True`, normalize the components of the random variables
                with their cumulative probability distributions.
            out: The array to store the normalized vector.
                If `None`, create a new array.

        Returns:
            The normalized vector.
        """
        if not use_dist:
            return super().normalize_vect(x_vect, out=out)

        if x_vect.ndim not in {1, 2}:
            msg = "x_vect must be either a 1D or a 2D NumPy array."
            raise ValueError(msg)

        return self.__normalize_vect(x_vect)

    def __normalize_vect(self, x_vect):
        data_names = self._variables.keys()
        data_sizes = self.variable_sizes
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
            variable for variable in self if variable not in self.uncertain_variables
        ]

    def extract_uncertain_space(
        self,
        as_design_space: bool = False,
    ) -> DesignSpace | ParameterSpace:
        """Define a new parameter space from the uncertain variables only.

        Args:
            as_design_space: If `False`,
                return a [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
                containing the original uncertain variables as is;
                otherwise,
                return a [DesignSpace][gemseo.algos.design_space.DesignSpace]
                where the original uncertain variables are made deterministic.
                In that case,
                the bounds of a deterministic variable correspond
                to the limits of the support of the original probability distribution
                and the current value correspond to its mean.

        Return:
            A parameter space defined by the uncertain variables only.
        """
        uncertain_space = self.filter(self.uncertain_variables, copy=True)
        if as_design_space:
            return uncertain_space.to_design_space()

        return uncertain_space

    def extract_deterministic_space(self) -> DesignSpace:
        """Define a design space from the deterministic variables only.

        Return:
            A design space defined by the deterministic variables only.
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
        copulas: Iterable[tuple[tuple[str, ...], Any]] = (),
    ) -> ParameterSpace:
        """Initialize the parameter space from a dataset.

        Args:
            dataset: The dataset used for the initialization.
            groups: The groups of the dataset to be considered.
                If empty, consider all the groups.
            uncertain: Whether the variables should be uncertain or not.
            copulas: The copula defined by independent blocks of random vectors.
                An element of `block_copulas` is
                a pair of random vector names and copula,
                where the copula must be from the same family
                as the marginal probability distributions,
                e.g. an OpenTURNS copula in the case of OpenTURNS marginals.
                The block must be passed as
                `set_joint_distribution((("b", "e"), copula_1), (("h", "d"), copula_2))`.
                Random vectors from different blocks are independent.
                A random vector can have only one copula.
                The components of a vector without a copula
                will be considered independent with all the other ones.
                In the absence of block copulas,
                all the components of the full random vector are independent.
        """  # noqa: E501
        parameter_space = ParameterSpace()

        if uncertain is None:
            uncertain = {}

        if not groups:
            groups = dataset.group_names
        for group in groups:
            for name in dataset.get_variable_names(group):
                data = dataset.get_view(variable_names=name).to_numpy()
                l_b = data.min(0)
                u_b = data.max(0)
                value = (l_b + u_b) / 2
                size = len(l_b)

                if uncertain.get(name, False):
                    for idx in range(size):
                        parameter_space.add_random_variable(
                            f"{name}_{idx}",
                            OTUniformDistribution_Settings(
                                minimum=float(l_b[idx]), maximum=float(u_b[idx])
                            ),
                        )
                else:
                    parameter_space.add_variable(name, size, "float", l_b, u_b, value)

        parameter_space.__set_joint_distribution()
        return parameter_space

    def to_design_space(self) -> DesignSpace:
        """Convert the parameter space into a design space.

        The original deterministic variables are kept as is
        while the original uncertain variables are made deterministic.
        In that case,
        the bounds of a deterministic variable correspond
        to the limits of the support of the original probability distribution
        and the current value correspond to its mean.

        Return:
            A design space where all original variables are made deterministic.
        """
        design_space = self.extract_deterministic_space()
        for name in self.uncertain_variables:
            design_space.add_variable(
                name,
                size=self.get_size(name),
                type_=self.get_type(name),
                lower_bound=self.get_lower_bound(name),
                upper_bound=self.get_upper_bound(name),
                value=self.get_current_value([name]),
            )
        return design_space

    def rename_variable(  # noqa:D102
        self,
        current_name: str,
        new_name: str,
    ) -> None:
        super().rename_variable(current_name, new_name)
        if current_name in self.uncertain_variables:
            position = self.uncertain_variables.index(current_name)
            self.uncertain_variables[position] = new_name
            dict_ = self.__random_vector_name_to_settings
            dict_[new_name] = dict_.pop(current_name)
            dict_ = self.distributions
            dict_[new_name] = dict_.pop(current_name)

    def add_variables_from(self, space: DesignSpace, *names: str) -> None:  # noqa: D102
        if not isinstance(space, ParameterSpace):
            super().add_variables_from(space, *names)
            return

        for name in names:
            if name in space.uncertain_variables:
                self.add_random_vector(
                    name,
                    *space.__random_vector_name_to_settings[name],
                )
                self.set_current_variable(name, space._current_value[name])
            else:
                self._add_variable_from(space, name)
