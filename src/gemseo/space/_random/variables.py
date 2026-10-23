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
"""Versioned random variables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.space._core.variables import Variables
from gemseo.space._core.variables import check_components
from gemseo.space.variable.random import RandomVariable
from gemseo.space.variable.random import _check_distribution_libraries
from gemseo.uncertainty.distribution.factory import distribution_factory
from gemseo.util.string import convert_strings_to_iterable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from typing import Any

    from openturns.model_copula import DistributionImplementation

    from gemseo.uncertainty.distribution.core.base_joint import BaseJointDistribution


class RandomVariables(Variables[RandomVariable]):
    """A registry of random variables.

    In addition to the generic registry behavior,
    this registry is the single source of truth for
    the dependency structure of the random variables,
    namely their copulas,
    and for the joint probability distribution of all the random variables.

    This joint probability distribution is rebuilt
    from the settings of the marginal probability distributions of the variables
    and from the copulas
    every time the registry is mutated.

    Note:
        A random space exposes this registry to its users
        as a
        [RandomVariablesView][gemseo.space._random.variables_view.RandomVariablesView],
        which is read-only.
    """

    __distribution: BaseJointDistribution | None
    """The joint probability distribution of the random variables, if any."""

    __copulas: list[tuple[tuple[str, ...], Any]]
    """The independent copulas defined by blocks of random variables."""

    __supports_dependency: bool
    """Whether the wrapped UQ library supports dependent variables."""

    __distribution_library_name: str
    """The name of the library implementing the probability distributions."""

    def __init__(self) -> None:  # noqa: D107
        self.__distribution = None
        self.__copulas = []
        self.__supports_dependency = True
        self.__distribution_library_name = ""
        super().__init__()

    @property
    def distribution(self) -> BaseJointDistribution | None:
        """The joint probability distribution of the random variables, if any."""
        return self.__distribution

    @property
    def copulas(self) -> tuple[tuple[tuple[str, ...], Any], ...]:
        """The independent copulas defined by blocks of random variables."""
        return tuple(self.__copulas)

    def __setitem__(self, name: str, variable: RandomVariable) -> None:
        settings = variable.distribution_settings
        library_names = {setting._library_name for setting in settings}
        if self.__distribution_library_name:
            library_names.add(self.__distribution_library_name)

        _check_distribution_libraries(library_names)

        if not len(self):
            self.__supports_dependency = (
                "copula" in self.__get_joint_class(settings).settings_class.model_fields
            )

        # Build the candidate joint probability distribution before mutating
        # anything, so that a failure to rebuild it leaves the registry unchanged.
        candidate_name_to_variable = {**self, name: variable}
        distribution = self.__build_distribution(
            candidate_name_to_variable, self.__copulas
        )

        self.__distribution_library_name = next(iter(library_names))
        super().__setitem__(name, variable)
        self.__distribution = distribution

    def __delitem__(self, name: str) -> None:
        # Validate the name,
        # then build the candidate copulas and joint probability distribution
        # before mutating anything,
        # so that a failure to rebuild it leaves the registry unchanged.
        self[name]
        candidate_copulas = [
            copula for copula in self.__copulas if name not in copula[0]
        ]
        candidate_name_to_variable = {
            other_name: variable
            for other_name, variable in self.items()
            if other_name != name
        }
        distribution = self.__build_distribution(
            candidate_name_to_variable, candidate_copulas
        )

        super().__delitem__(name)
        self.__copulas = candidate_copulas
        self.__distribution = distribution
        if not candidate_name_to_variable:
            self.__distribution_library_name = ""

    def rename(self, current_name: str, new_name: str) -> None:  # noqa: D102
        # Build the candidate copulas and, if any of them is affected by the rename,
        # the candidate joint probability distribution, before mutating anything,
        # so that a failure to rebuild it leaves the registry unchanged.
        # Skip that work when the rename cannot succeed anyway
        # (unknown name or name collision),
        # so that the base implementation raises with its usual error message
        # instead of a confusing one from an inconsistent candidate
        # copula/variable state.
        renamed = False
        if current_name in self and (new_name == current_name or new_name not in self):
            candidate_copulas = []
            for names, copula in self.__copulas:
                if current_name in names:
                    names = tuple(
                        new_name if name == current_name else name for name in names
                    )
                    renamed = True
                candidate_copulas.append((names, copula))

            if renamed:
                candidate_name_to_variable = {
                    (new_name if name == current_name else name): variable
                    for name, variable in self.items()
                }
                distribution = self.__build_distribution(
                    candidate_name_to_variable, candidate_copulas
                )

        super().rename(current_name, new_name)

        if renamed:
            self.__copulas = candidate_copulas
            self.__distribution = distribution

    def filter_components(self, name: str, components: Sequence[int]) -> None:
        """Keep only certain components of a random variable.

        The random variable is rebuilt from the settings
        of the marginal probability distributions of the components to be kept,
        so that its size, bounds and probability distribution follow.

        The copula covering this random variable, if any, is removed,
        as its dimension no longer matches that of the random variable;
        the random variables it covered become independent.

        Keeping every component in order changes nothing,
        neither the random variable nor the copulas covering it.

        Args:
            name: The name of the random variable.
            components: The components to be kept.

        Raises:
            ValueError: When no component is to be kept.

        Note:
            This method increments the version number,
            unless every component is kept in order.
        """
        check_components(name, components)

        # Build the replacement random variable before mutating anything,
        # so that a failure leaves the registry and its copulas unchanged.
        variable = self[name].filter_components(components)

        if variable is self[name]:
            # Every component is kept, so the dimension of the random variable
            # still matches that of the copulas covering it.
            return

        # Remove the stale copulas before replacing the random variable,
        # so that the joint probability distribution is rebuilt once,
        # from the new sizes.
        for copula in list(self.__copulas):
            if name in copula[0]:
                self.__copulas.remove(copula)

        self[name] = variable

    def add_copula(
        self, names: str | Iterable[str], copula: DistributionImplementation
    ) -> None:
        """Add a copula defining the dependency structure between random variables.

        This function can be called several times in order to add several copulas
        associated to different random variables.
        All the variables which are not linked through any copula will be
        independent.

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
        # Read once, so that an iterator of names is not exhausted
        # by the checks below before being stored with the copula.
        names = tuple(convert_strings_to_iterable(names))
        if not names:
            msg = "A copula must cover at least one random variable."
            raise ValueError(msg)

        for name in names:
            if name not in self:
                msg = f"There is no variable name {name!r}."
                raise ValueError(msg)

            for existing_names, _ in self.__copulas:
                if name in existing_names:
                    msg = f"The random variable {name!r} has already a copula."
                    raise ValueError(msg)

        if not self.__supports_dependency:
            msg = (
                f"{self.__distribution.__class__.__name__} does not support "
                "dependent variables."
            )
            raise ValueError(msg)

        # Build the candidate copulas and joint probability distribution before
        # mutating anything, so that a copula whose dimension does not match the
        # covered variables leaves the registry unchanged.
        candidate_copulas = [*self.__copulas, (names, copula)]
        distribution = self.__build_distribution(dict(self), candidate_copulas)
        self.__copulas = candidate_copulas
        self.__distribution = distribution

    @staticmethod
    def __get_joint_class(
        marginal_settings: Sequence[Any],
    ) -> type[BaseJointDistribution]:
        """Return the class of the joint probability distribution of the marginals.

        Args:
            marginal_settings: The settings of the marginal probability distributions.

        Returns:
            The class of the joint probability distribution.
        """
        marginal_class_name = marginal_settings[0].target_class_name
        marginal_class = distribution_factory.get_class(marginal_class_name)
        return marginal_class.joint_distribution_class

    @classmethod
    def __build_distribution(
        cls,
        name_to_variable: dict[str, RandomVariable],
        copulas: list[tuple[tuple[str, ...], DistributionImplementation]],
    ) -> BaseJointDistribution | None:
        """Build the joint probability distribution of a candidate state.

        This does not mutate the registry,
        so that a candidate state can be validated
        before being committed.

        Args:
            name_to_variable: The candidate ordered mapping
                from a random variable name to a random variable.
            copulas: The candidate independent copulas
                defined by blocks of random variables.

        Returns:
            The joint probability distribution,
            or `None` when there is no variable.
        """
        if not name_to_variable:
            return None

        marginal_settings = []
        for variable in name_to_variable.values():
            marginal_settings.extend(variable.distribution_settings)

        joint_class = cls.__get_joint_class(marginal_settings)
        settings_class = joint_class.settings_class
        if copulas:
            new_copulas = []
            names = list(name_to_variable)
            variable_sizes = {name: name_to_variable[name].size for name in names}
            for variable_names, copula in copulas:
                indices = []
                for variable_name in variable_names:
                    pos = sum(
                        variable_sizes[names[i]]
                        for i in range(names.index(variable_name))
                    )
                    indices.extend(range(pos, pos + variable_sizes[variable_name]))

                new_copulas.append((indices, copula))

            settings = settings_class(
                marginal_settings=marginal_settings,
                copula=new_copulas,
            )
        else:
            settings = settings_class(marginal_settings=marginal_settings)

        return joint_class(settings)
