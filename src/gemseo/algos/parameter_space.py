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
  (see :func:`.uncertainty.get_available_distributions`),
- an optional variable size,
- optional distribution parameters (``parameters`` set as
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

import logging
from typing import TYPE_CHECKING
from typing import Any

from prettytable import PrettyTable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.datasets.dataset import Dataset
    from gemseo.typing import StrKeyMapping
    from gemseo.uncertainty.distributions.base_joint import BaseJointDistribution

from numpy import array
from numpy import ndarray

from gemseo.algos.design_space import DesignSpace
from gemseo.uncertainty.distributions.factory import DistributionFactory
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

    distribution: BaseJointDistribution
    """The joint probability distribution of the uncertain variables."""

    _INITIAL_DISTRIBUTION = "Initial distribution"
    _TRANSFORMATION = "Transformation"
    _SUPPORT = "Support"
    _MEAN = "Mean"
    _STANDARD_DEVIATION = "Standard deviation"
    _RANGE = "Range"
    _BLANK = ""
    _PARAMETER_SPACE = "Parameter space"

    __uncertain_variables_to_definitions: dict[str, tuple[str, dict[str, Any]]]
    """The uncertain variable names bound to their definition.

    The definition is a 2-tuple. The first component is the name of the class of the
    probability distribution and the second one is the dictionary of the keywords
    arguments of the add_random_vector method.
    """

    def __init__(self, name: str = "") -> None:  # noqa:D107
        LOGGER.debug("*** Create a new parameter space ***")
        super().__init__(name=name)
        self.uncertain_variables = []
        self.distributions = {}
        self.distribution = None
        self.__uncertain_variables_to_definitions = {}
        self.__distribution_family_id = ""

    def build_joint_distribution(self, copula: Any = None) -> None:
        """Build the joint probability distribution.

        Args:
            copula: A copula distribution
                defining the dependency structure between random variables;
                if ``None``, consider an independent copula.
        """
        if self.uncertain_variables:
            distributions = [
                marginal
                for name in self.uncertain_variables
                for marginal in self.distributions[name].marginals
            ]
            self.distribution = distributions[0].JOINT_DISTRIBUTION_CLASS(
                distributions, copula
            )

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
        distribution: str,
        size: int = 0,
        interfaced_distribution: str = "",
        interfaced_distribution_parameters: tuple[list[Any]]
        | Mapping[str, list[Any]] = (),
        **parameters: list[Any],
    ) -> None:
        """Add a *d*-length random vector from a probability distribution.

        Warnings:
            The probability distributions must have
            the same identifier. For instance,
            one cannot mix a random vector
            using a :class:`.OTUniformDistribution` with identifier ``"OT"``
            and a random vector
            using a :class:`.SPNormalDistribution` with identifier ``"SP"``.

        Args:
            name: The name of the random vector.
            distribution: The name of a class
                implementing a probability distribution,
                e.g. ``"OTUniformDistribution"`` or ``"SPUniformDistribution"``,
                or an interface to a library of probability distributions,
                e.g. ``"OTDistribution"`` or ``"SPDistribution"``.
            size: The length *d* of the random vector.
                If ``0``, deduce it from the parameters.
            interfaced_distribution: The name of the distribution
                in the library of probability distributions
                when ``distribution`` is the name of a class
                implementing an interface to this library.
            interfaced_distribution_parameters: The parameters of the distribution
                in the library of probability distributions
                when ``distribution`` is the name of a class
                implementing an interface to this library.
                The values of the data structure (mapping or tuple) must be set
                either as ``[p_1,...,p_d]``
                (one value per component of the random vector)
                or as ``[p]``
                (one value for all the components)
                If empty, use the default ones.
            **parameters: The parameters of the distribution,
                either as ``[p_1,...,p_d]``
                (one value per component of the random vector)
                or as ``[p]``
                (one value for all the components);
                otherwise, use the default ones.

        Raises:
            ValueError: When mixing probability distributions from different families,
                e.g. an :class:`.OTDistribution` and a :class:`.SPDistribution` or
                when the lengths of the distribution parameter collections
                are not consistent.
        """
        self._check_variable_name(name)
        distribution_class = DistributionFactory().get_class(distribution)
        parameters_as_tuple = isinstance(interfaced_distribution_parameters, tuple)

        # Check that the distribution belongs to the same library as the previous ones.
        distribution_family_id = distribution_class.__name__[0:2]
        if self.__distribution_family_id:
            if distribution_family_id != self.__distribution_family_id:
                msg = (
                    f"A parameter space cannot mix {self.__distribution_family_id} "
                    f"and {distribution_family_id} distributions."
                )
                raise ValueError(msg)
        else:
            self.__distribution_family_id = distribution_family_id

        # Set the size if undefined and check the consistency with the parameters.
        size = self.__get_random_vector_size(
            interfaced_distribution_parameters, parameters.values(), size
        )

        # Force the collections of the parameters to the same size.
        parameters = {
            name: self.__get_random_vector_parameter_value(size, value)
            for name, value in parameters.items()
        }
        if parameters_as_tuple:
            interfaced_distribution_parameters = tuple(
                self.__get_random_vector_parameter_value(size, value)
                for value in interfaced_distribution_parameters
            )
        else:
            interfaced_distribution_parameters = {
                name: self.__get_random_vector_parameter_value(size, value)
                for name, value in interfaced_distribution_parameters.items()
            }

        # Store the definitions of the uncertain variables
        # for use by RandomVariable and RandomVector (see __getitem__ and __setitem__).
        is_random_variable = size == 1
        if is_random_variable:
            data = {name: value[0] for name, value in parameters.items()}
            if parameters_as_tuple:
                idp_data = {value[0] for value in interfaced_distribution_parameters}
            else:
                idp_data = {
                    name: value[0]
                    for name, value in interfaced_distribution_parameters.items()
                }
        else:
            data = parameters.copy()
            idp_data = interfaced_distribution_parameters
        if idp_data:
            data["parameters"] = idp_data
        self.__uncertain_variables_to_definitions[name] = (
            distribution,
            {
                "size": size,
                "interfaced_distribution": interfaced_distribution,
                "interfaced_distribution_parameters": interfaced_distribution_parameters,  # noqa: E501
                **parameters,
            },
        )

        # Define the marginal distributions
        # (one marginal for each component of the random vector).
        marginals = []
        for i in range(size):
            kwargs = {k: v[i] for k, v in parameters.items()}
            if interfaced_distribution:
                kwargs["interfaced_distribution"] = interfaced_distribution
                if parameters_as_tuple:
                    kwargs["parameters"] = tuple(
                        v[i] for v in interfaced_distribution_parameters
                    )
                else:
                    kwargs["parameters"] = {
                        k: v[i] for k, v in interfaced_distribution_parameters.items()
                    }

            marginals.append(distribution_class(**kwargs))

        # Define the distribution of the random vector with a joint distribution.
        joint_distribution_class = distribution_class.JOINT_DISTRIBUTION_CLASS
        self.distributions[name] = joint_distribution_class(marginals)

        # Update the uncertain variables.
        self.uncertain_variables.append(name)

        # Update the full joint distribution,
        # i.e. the joint distribution of all the uncertain variables.
        self.build_joint_distribution()

        # Update the parameter space as subclass of a DesignSpace.
        l_b = self.distributions[name].math_lower_bound
        u_b = self.distributions[name].math_upper_bound
        value = self.distributions[name].mean
        self.add_variable(
            name,
            self.distributions[name].dimension,
            self.DesignVariableType.FLOAT,
            l_b,
            u_b,
            value,
        )

    @staticmethod
    def __get_random_vector_parameter_value(size: int, value: list[Any]) -> list[Any]:
        """Adapt the parameter value if its size is inconsistent with the random vector.

        When the ``size`` of the random vector greater than one
        and the length of the parameter ``value`` is 1,
        the parameter value is repeated ``size`` times.

        Args:
            size: The size of the random vector.
            value: The value of the parameter of the probability distribution,
                whose length is either ``1`` or ``size``.

        Returns:
            The value of the parameter of the probability distribution
            for each component of the random vector.
        """
        return value * size if len(value) == 1 and size != 1 else value

    @staticmethod
    def __get_random_vector_size(
        interfaced_distribution_parameters: tuple[list[Any]] | Mapping[str, list[Any]],
        parameter_values: Iterable[list[Any]],
        size: int,
    ) -> int:
        """Define the random vector size if undefined.

        Args:
            interfaced_distribution_parameters: The parameters of the distribution
                in the library of probability distributions
                when ``distribution`` is the name of a class
                implementing an interface to this library;
                if empty, use the default ones.
            parameter_values: The parameters of the distribution,
                either as ``(p_1,...,p_d)``
                (one value per component of the random vector)
                or as ``(p)``
                (one value for all the components);
                otherwise, use the default ones.
            size: The length *d* of the random vector.
                If ``0``, deduce it from the parameters.

        Returns:
            The size of the random vector.

        Raises:
            ValueError: The lengths of the distribution parameter collections
                are not consistent.
        """
        parameters_as_tuple = isinstance(interfaced_distribution_parameters, tuple)

        # Bring together the collections of values of the various parameters.
        values = (
            interfaced_distribution_parameters
            if parameters_as_tuple
            else interfaced_distribution_parameters.values()
        )

        # Compute the unique collection sizes; expectation: {1}, {d} or {1,d}.
        sizes = {len(v) for vv in [values, parameter_values] for v in vv}
        n_sizes = len(sizes)

        # Set the size if undefined.
        if not size:
            size = max(sizes) if sizes else 1

        # Check the consistency of the size with the parameters.
        if (
            n_sizes > 2
            or (n_sizes == 2 and sizes != {1, size})
            or (n_sizes == 1 and not sizes.issubset({1, size}))
        ):
            msg = (
                "The lengths of the distribution parameter collections "
                "are not consistent."
            )
            raise ValueError(msg)

        return size

    def add_random_variable(
        self,
        name: str,
        distribution: str,
        size: int = 1,
        interfaced_distribution: str = "",
        interfaced_distribution_parameters: tuple[Any] | StrKeyMapping = (),
        **parameters: Any,
    ) -> None:
        """Add a random variable from a probability distribution.

        Args:
            name: The name of the random variable.
            distribution: The name of a class
                implementing a probability distribution,
                e.g. ``"OTUniformDistribution"`` or ``"SPUniformDistribution"``,
                or an interface to a library of probability distributions,
                e.g. ``"OTDistribution"`` or ``"SPDistribution"``.
            size: The dimension of the random variable.
                The parameters of the distribution are shared
                by all the components of the random variable.
            interfaced_distribution: The name of the distribution
                in the library of probability distributions
                when ``distribution`` is the name of a class
                implementing an interface to this library.
            interfaced_distribution_parameters: The parameters of the distribution
                in the library of probability distributions
                when ``distribution`` is the name of a class
                implementing an interface to this library;
                if empty, use the default ones.
            **parameters: The parameters of the distribution;
                otherwise, use the default ones.

        Warnings:
            The probability distributions must have
            the same identifier. For instance,
            one cannot mix a random variable
            distributed as an :class:`.OTUniformDistribution` with identifier ``"OT"``
            and a random variable
            distributed as a :class:`.SPNormalDistribution` with identifier ``"SP"``.
        """
        kwargs = {k: [v] for k, v in parameters.items()}
        if interfaced_distribution:
            kwargs["interfaced_distribution"] = interfaced_distribution
            if interfaced_distribution_parameters:
                if isinstance(interfaced_distribution_parameters, tuple):
                    formatted_parameters = tuple(
                        [v] for v in interfaced_distribution_parameters
                    )
                else:
                    formatted_parameters = {
                        k: [v] for k, v in interfaced_distribution_parameters.items()
                    }
                kwargs.update({
                    "interfaced_distribution_parameters": formatted_parameters
                })

        self.add_random_vector(
            name,
            distribution,
            size,
            **kwargs,
        )

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
                self.build_joint_distribution()
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
                If ``True``, return a dictionary.
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
                If ``True``, compute the cumulative density function.
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
                self.distributions[self.uncertain_variables[0]]
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
                If ``None``, create a new array.

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
        use the approach defined in :meth:`.DesignSpace.normalize_vect`
        with `minus_lb`.

        For the components of the deterministic variables,
        use the approach defined in :meth:`.DesignSpace.normalize_vect`
        with `minus_lb`.

        Args:
            x_vect: The values of the design variables.
            minus_lb: If ``True``, remove the lower bounds at normalization.
            use_dist: If ``True``, normalize the components of the random variables
                with their cumulative probability distributions.
            out: The array to store the normalized vector.
                If ``None``, create a new array.

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
        """Define a new :class:`.DesignSpace` from the uncertain variables only.

        Args:
            as_design_space: If ``False``,
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
        uncertain_space = self.filter(self.uncertain_variables, copy=True)
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
        copula: Any = None,
    ) -> ParameterSpace:
        """Initialize the parameter space from a dataset.

        Args:
            dataset: The dataset used for the initialization.
            groups: The groups of the dataset to be considered.
                If empty, consider all the groups.
            uncertain: Whether the variables should be uncertain or not.
            copula: A name of copula defining the dependency between random variables.
        """
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
                            "OTUniformDistribution",
                            minimum=float(l_b[idx]),
                            maximum=float(u_b[idx]),
                        )
                else:
                    parameter_space.add_variable(name, size, "float", l_b, u_b, value)

        parameter_space.build_joint_distribution(copula)
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
            dict_ = self.__uncertain_variables_to_definitions
            dict_[new_name] = dict_.pop(current_name)
            dict_ = self.distributions
            dict_[new_name] = dict_.pop(current_name)

    def add_variables_from(self, space: DesignSpace, *names: str) -> None:  # noqa: D102
        if not isinstance(space, ParameterSpace):
            super().add_variables_from(space, *names)
            return

        for name in names:
            if name in space.uncertain_variables:
                definition = space.__uncertain_variables_to_definitions[name]
                self.add_random_vector(name, definition[0], **definition[1])
                self.set_current_variable(name, space._current_value[name])
            else:
                self._add_variable_from(space, name)
