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
"""Random space."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final
from typing import cast

from numpy import array
from prettytable import PrettyTable

from gemseo.space._random.variables import RandomVariables
from gemseo.space._random.variables_view import RandomVariablesView
from gemseo.space.base import BaseVariableSpace
from gemseo.space.variable.random import RandomVariable
from gemseo.space.variable.random import _check_distribution_libraries
from gemseo.util.data_conversion import split_array_to_dict_of_arrays
from gemseo.util.string import _format_value_in_pretty_table_16

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from numpy import ndarray
    from openturns.model_copula import DistributionImplementation

    from gemseo.uncertainty.distribution.core.base_joint import BaseJointDistribution
    from gemseo.uncertainty.distribution.core.base_settings import (
        BaseDistributionSettings,
    )


_VARIABLES_VIEW: Final[str] = f"_{BaseVariableSpace.__name__}__variables_view"
"""The name of the attribute holding the view over the registry of variables.

An attribute set by any space of variables of this release, hence absent from the
state of a space pickled by a release predating the hierarchy of spaces.
"""


class RandomSpace(BaseVariableSpace[RandomVariables, RandomVariablesView]):
    """A space of random variables.

    A random space stores random variables,
    together with their names, sizes, types
    and bounds, which are the limits of the supports
    of their probability distributions,
    and provides the operations to manipulate them.
    These bounds are read-only and descriptive:
    unlike the bounds of a design variable,
    they are neither checked when evaluating a function
    nor used to bound the perturbations
    of a finite-difference approximation.

    The random variables are defined
    from the settings of the marginal probability distributions of their components
    and from copulas characterizing their dependency structure
    (see [add_copula][gemseo.space.random.RandomSpace.add_copula]).
    All the probabilistic data is read through the public registry
    [variables][gemseo.space.base.BaseVariableSpace.variables]:
    `variables.distribution` for the joint probability distribution of the space,
    `variables[name].distribution` for the distribution of a random variable,
    and `variables[name].distribution.range` and `.support`
    for its range and support.

    Unlike a [DesignSpace][gemseo.space.design.DesignSpace],
    a random space has no bounds setters, no current value
    and no normalization;
    its only mapping to and from the unit hypercube is the iso-probabilistic
    [transform_vect][gemseo.space.random.RandomSpace.transform_vect]
    and [untransform_vect][gemseo.space.random.RandomSpace.untransform_vect].

    Its [reference_value][gemseo.space.random.RandomSpace.reference_value],
    namely the single representative value per variable
    that a consumer of the space uses when it needs one,
    is the mean of the probability distributions of its random variables.
    """

    _variables_class: ClassVar[type[RandomVariables]] = RandomVariables

    _variables_view_class: ClassVar[type[RandomVariablesView]] = RandomVariablesView

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the random space from a pickled state.

        Args:
            state: The state of the random space.

        Raises:
            TypeError: When the state is the one of a `ParameterSpace`.
        """
        # The name ParameterSpace, removed in favor of RandomSpace, is redirected
        # onto this class by gemseo._deprecation, so a space pickled by a previous
        # release is restored here, with a state this class cannot read.
        if _VARIABLES_VIEW not in state:
            msg = (
                "This state is the one of a ParameterSpace, "
                "a class removed in favor of RandomSpace, "
                "which stores its probability distributions differently. "
                "Rebuild the random space from the settings of the marginal "
                "probability distributions of its random variables."
            )
            raise TypeError(msg)

        self.__dict__.update(state)

    def add_variable(self, name: str, *settings: BaseDistributionSettings) -> None:
        """Add a random variable.

        There is a distribution settings model per component of the random variable,
        so a random vector with identical marginal distributions is added
        by repeating the settings,
        e.g. `space.add_variable("x", *[settings] * 3)` for a 3-length random vector.

        Args:
            name: The name of the random variable.
            *settings: The settings of the marginal probability distributions
                of the components.

        Raises:
            ValueError: When no distribution settings are passed,
                when mixing probability distributions from different families,
                e.g. an
                [OTDistribution][gemseo.uncertainty.distribution.openturns.distribution.OTDistribution]
                and a
                [SPDistribution][gemseo.uncertainty.distribution.scipy.distribution.SPDistribution]
                or
                when the variable name already exists.

        Note:
            A random variable cannot be replaced in place;
            remove it with
            [remove_variable][gemseo.space.base.BaseVariableSpace.remove_variable]
            and add it again,
            which moves it to the end of the random space.
        """
        # Checked here, before the model is built, so that an empty call
        # and mixing libraries within this call raise a plain ValueError,
        # instead of the pydantic validation error
        # that would result from checking them in the RandomVariable model.
        if not settings:
            msg = (
                f"The random variable {name} is defined by the settings "
                "of the marginal probability distribution of each of its components, "
                "hence at least one."
            )
            raise ValueError(msg)

        _check_distribution_libraries({setting._library_name for setting in settings})
        self._add_variable(name, RandomVariable(distribution_settings=settings))

    def filter_dimensions(self, name: str, dimensions: Sequence[int]) -> RandomSpace:
        """
        Warning:
            The random variable is rebuilt from the settings
            of the marginal probability distributions of the components to be kept,
            and the copula covering it, if any, is removed,
            so that the random variables it covered become independent;
            keeping every dimension in order changes nothing.
        """  # noqa: D205, D212, D415
        return super().filter_dimensions(name, dimensions)

    @property
    def reference_value(self) -> dict[str, ndarray]:
        """The reference value of the random space.

        It is the mean of the probability distributions
        of the random variables of the space,
        read from these distributions on each call,
        so it follows a change of their settings
        without any action from the user.
        It is empty when the space has no random variable.
        """
        distribution = self._variables.distribution
        if distribution is None:
            return {}

        return self.convert_array_to_dict(distribution.mean)

    def add_copula(
        self, names: str | Iterable[str], copula: DistributionImplementation
    ) -> None:
        """Add a copula defining the dependency structure between random variables.

        This function can be called several times
        in order to add several copulas associated to different random variables.
        All the variables which are not linked through any copula will be independent.

        The arguments are ordered as
        [copulas][gemseo.space._random.variables_view.RandomVariablesView.copulas]
        yields them,
        so copying the dependency structure of a random space reads as

        ```python
        for names, copula in space.variables.copulas:
            other_space.add_copula(names, copula)
        ```

        Args:
            names: The name of the random variable
                or the names of the random variables covered by the copula.
            copula: The copula.

        Raises:
            ValueError: When no random variable name is passed,
                when the joint probability distribution does not support
                dependent random variables,
                i.e. when the joint distribution settings do not have a `copula` field,
                when there is no variable with that name,
                or when there is already a copula for one of the random variables.
        """
        self._variables.add_copula(names, copula)

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
            either stored in an array shaped as `(n_samples, dimension)`
            or in a dictionary
            whose keys are the names of the random variables
            and whose values are the realizations,
            shaped as `(n_samples, size)`.

        Raises:
            ValueError: When the random space is empty.
        """
        sample = self.__get_distribution().compute_samples(n_samples)
        if as_dict:
            names_to_sizes = {
                name: variable.size for name, variable in self._variables.items()
            }
            return split_array_to_dict_of_arrays(
                sample, names_to_sizes, list(self._variables)
            )

        return sample

    def transform_vect(self, x_vect: ndarray) -> ndarray:  # noqa: D102
        return self.__transform(x_vect, inverse=False)

    def untransform_vect(
        self,
        x_vect: ndarray,
        no_check: bool = False,
    ) -> ndarray:
        """Map a point of the unit hypercube to the space.

        The mapping is iso-probabilistic:
        it applies the inverse cumulative distribution function
        of the joint probability distribution of the random variables.

        Args:
            x_vect: A vector with components in $[0,1]$.
            no_check: Whether to check if the components are in $[0,1]$.

        Returns:
            A point of the space.

        Raises:
            ValueError: When the random space is empty,
                e.g. `ValueError: The random space is empty.`,
                or when the last dimension of the vector
                is not the dimension of the space,
                e.g. `ValueError: Expected an array of shape (..., 1); got (2,).`,
                or when a component of the vector is outside $[0,1]$
                and `no_check` is `False`.
        """
        # Report an empty random space before anything else,
        # as the dimension of the unit hypercube is then zero.
        self.__get_distribution()
        if not no_check:
            self.__check_unit_vect(x_vect)

        return self.__transform(x_vect, inverse=True)

    def __get_distribution(self) -> BaseJointDistribution:
        """Return the joint probability distribution of the random space.

        Returns:
            The joint probability distribution of the random space.

        Raises:
            ValueError: When the random space is empty.
        """
        # An empty space is the only state without a joint probability distribution,
        # so let check() report it, with the message shared by all the spaces.
        self.check()
        return cast("BaseJointDistribution", self._variables.distribution)

    def __transform(self, x_vect: ndarray, inverse: bool) -> ndarray:
        """Evaluate the iso-probabilistic transformation of the random variables.

        This transformation maps the random variables
        to (or from with `inverse=True`) independent random variables
        uniformly distributed over $[0,1]$,
        taking the dependence between the random variables into account.
        When the random variables are independent,
        it reduces to the cumulative density functions (or their inverses)
        of the marginal distributions.

        Args:
            x_vect: A point of the random space,
                or a point of the unit hypercube with `inverse=True`.
                The components are read along the last dimension,
                so an array of shape `(..., dimension)` maps several points at once.
            inverse: Whether the inverse cumulative density function
                is used as the evaluation function,
                or the cumulative density function.

        Returns:
            The transformed point.

        Raises:
            ValueError: When the random space is empty.
        """
        distribution = self.__get_distribution()
        transform = (
            distribution.map_from_uniform if inverse else distribution.map_to_uniform
        )
        if x_vect.ndim == 1:
            return transform(x_vect)

        # The transformation maps a single point,
        # so the dimensions preceding the components are flattened
        # and the shape of the input is restored afterwards.
        points = x_vect.reshape(-1, x_vect.shape[-1])
        return array(list(map(transform, points))).reshape(x_vect.shape)

    def __check_unit_vect(self, x_vect: ndarray) -> None:
        """Check that a point belongs to the unit hypercube of the random space.

        Args:
            x_vect: The point to be checked.

        Raises:
            ValueError: When the point does not have the dimension of the space
                or when its components are not in $[0,1]$.
        """
        dimension = self.dimension
        # Compare the trailing dimension as a slice,
        # so that a 0D array is reported instead of raising an IndexError.
        if x_vect.shape[-1:] != (dimension,):
            msg = f"Expected an array of shape (..., {dimension}); got {x_vect.shape}."
            raise ValueError(msg)

        if (x_vect > 1.0).any() or (x_vect < 0.0).any():
            msg = "The components of x_vect must be in [0, 1]."
            raise ValueError(msg)

    def get_pretty_table(
        self,
        fields: Sequence[str] = (),
        with_index: bool = False,
        capitalize: bool = False,
    ) -> PrettyTable:
        """Build a tabular view of the random space.

        The table has a name column and a distribution column,
        the latter being split into an initial distribution column
        and a transformation column
        when at least one probability distribution has a transformation.

        Args:
            fields: The names of the fields to be exported.
                This argument is ignored,
                as the fields of the table are fixed.
            with_index: Whether to show index of names for arrays.
                This is ignored for scalars.
            capitalize: Whether to capitalize the field names
                and replace `"_"` by `" "`.

        Returns:
            A tabular view of the random space.
        """
        table = PrettyTable(["Name" if capitalize else "name"])
        table.custom_format = _format_value_in_pretty_table_16
        distributions = []
        transformations = []
        for name, variable in self._variables.items():
            name_template = f"{name}"
            if with_index and variable.size > 1:
                name_template += "[{index}]"

            for index in range(variable.size):
                table.add_row([name_template.format(name=name, index=index)])

            for marginal in variable.distribution.marginals:
                distributions.append(repr(marginal))
                transformations.append(marginal.transformation)

        for name in ("Name",) if capitalize else ("name",):
            table.align[name] = "l"

        if not self._variables:
            return table

        first_variable = next(iter(self._variables.values()))
        default_variable_name = first_variable.distribution.marginals[
            0
        ].default_variable_name
        if any(
            transformation != default_variable_name
            for transformation in transformations
        ):
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

    def _render_footer(self) -> str:
        """
        Note:
            The footer of a random space lists its copulas,
            as a copula relates several random variables
            and so cannot be shown in the tabular view.
        """  # noqa: D205, D212, D415
        copulas = self._variables.copulas
        if not copulas:
            return ""

        blocks = ", ".join(
            f"({', '.join(names)}) -> {type(copula).__name__}"
            for names, copula in copulas
        )
        return f"Copulas: {blocks}"

    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False

        return self._variables.copulas == other._variables.copulas
