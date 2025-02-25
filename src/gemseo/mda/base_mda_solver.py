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
# Copyright 2024 Capgemini
"""The base class for MDA solvers."""

from __future__ import annotations

import logging
from abc import abstractmethod
from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import abs as np_abs
from numpy import array
from numpy import concatenate
from numpy import inf
from numpy import ndarray
from numpy import ones
from numpy.linalg import norm

from gemseo.algos.sequence_transformer.composite.relaxation_acceleration import (
    RelaxationAcceleration,
)
from gemseo.mda.base_mda import BaseMDA

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
    from gemseo.core.discipline import Discipline
    from gemseo.mda.base_mda_solver_settings import BaseMDASolverSettings
    from gemseo.typing import MutableStrKeyMapping
    from gemseo.typing import RealArray

LOGGER = logging.getLogger(__name__)


class BaseMDASolver(BaseMDA):
    """The base class for MDA solvers."""

    Settings: ClassVar[type[BaseMDASolverSettings]]
    """The Pydantic model for the settings."""

    settings: BaseMDASolverSettings
    """The settings of the MDA."""

    _current_residuals: dict[str, ndarray]
    """The mapping from residual names to current values."""

    _sequence_transformer: RelaxationAcceleration
    """The sequence transformer aimed at improving the convergence rate.

    The transformation applies a relaxation followed by an acceleration.
    """

    __lower_bound_vector: RealArray | None
    """The vector of lower bounds."""

    __upper_bound_vector: RealArray | None
    """The vector of upper bounds."""

    __resolved_variable_names_to_bounds: dict[
        str, tuple[RealArray | None, RealArray | None]
    ]
    """The mapping from variable names to lower/upper bounds."""

    __resolved_variable_names_to_slices: dict[str, slice]
    """The mapping from names to slices for converting array to data structures."""

    __resolved_variable_names: tuple[str, ...]
    """The names of the resolved variables.

    Resolved variables are coupling and state variables (for disciplines that does not
    solve their own residuals). These variables are modified by the MDA so as to make
    the corresponding residuals converge towards 0.
    """

    __resolved_residual_names: tuple[str, ...]
    """The names of the resolved residuals the MDA is solving.

    Resolved residuals are either related to coupling variables or explicitly attached
    to state variables (for disciplines that does not solve their own residuals). The
    MDA is meant to bring these residuals towards zero.
    """

    __n_consecutive_unsuccessful_iterations: int
    """The number of consecutive unsuccessful iterations."""

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        settings_model: BaseMDASolverSettings | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(disciplines, settings_model=settings_model, **settings)

        self._sequence_transformer = RelaxationAcceleration(
            self.settings.over_relaxation_factor,
            self.settings.acceleration_method,
        )

        self.__resolved_variable_names_to_slices = {}
        self.__resolved_variable_names = ()

        self.__resolved_residual_names = ()

        self._current_residuals = {}
        self.__n_consecutive_unsuccessful_iterations = 0

        self.__lower_bound_vector = None
        self.__upper_bound_vector = None
        self.__resolved_variable_names_to_bounds = {}

    @property
    def acceleration_method(self) -> AccelerationMethod:
        """The acceleration method."""
        return self._sequence_transformer.acceleration_method

    @acceleration_method.setter
    def acceleration_method(self, acceleration_method: AccelerationMethod) -> None:
        self._sequence_transformer.acceleration_method = acceleration_method

    @property
    def over_relaxation_factor(self) -> float:
        """The over-relaxation factor."""
        return self._sequence_transformer.over_relaxation_factor

    @over_relaxation_factor.setter
    def over_relaxation_factor(self, over_relaxation_factor: float) -> None:
        self._sequence_transformer.over_relaxation_factor = over_relaxation_factor

    @property
    def lower_bound_vector(self) -> RealArray | None:
        """The vector of resolved variables lower bound."""
        return self.__lower_bound_vector

    @property
    def upper_bound_vector(self) -> RealArray | None:
        """The vector of resolved variables upper bound."""
        return self.__upper_bound_vector

    @property
    def _resolved_variable_names(self) -> tuple[str, ...]:
        """The names of the variables (couplings and state) the MDA is solving."""
        return self.__resolved_variable_names

    @property
    def _resolved_residual_names(self) -> tuple[str, ...]:
        """The names of the residuals (couplings and residuals) the MDA is solving."""
        return self.__resolved_residual_names

    def get_current_resolved_variables_vector(self) -> ndarray:
        """Return the vector of resolved variables (couplings and state variables)."""
        return concatenate([
            self.io.input_grammar.data_converter.convert_data_to_array(
                self.__resolved_variable_names,
                self.io.data,
            )
        ])

    def get_current_resolved_residual_vector(self) -> ndarray:
        """Return the vector of residuals."""
        return concatenate([
            self.io.output_grammar.data_converter.convert_data_to_array(
                self.__resolved_residual_names,
                self._current_residuals,
            )
        ])

    def set_bounds(
        self,
        variable_names_to_bounds: Mapping[
            str, tuple[RealArray | None, RealArray | None]
        ],
    ) -> None:
        """Set the bounds for the resolved variables.

        Args:
            variable_names_to_bounds: The mapping from variable names to bounds.
        """
        self.__resolved_variable_names_to_bounds |= {
            name: bounds
            for name, bounds in variable_names_to_bounds.items()
            if name in self._resolved_variable_names
        }

        self.__update_bounds_vectors()

    def _check_stopping_criteria(self, update_iteration_metrics: bool = True) -> bool:
        """Check whether a stopping criterion has been reached.

        Args:
            update_iteration_metrics: Whether to update the iteration metrics before
                checking the stopping criteria.

        Returns:
            Whether a stopping criterion has been reached.
        """
        if update_iteration_metrics:
            self.__update_iteration_metrics()

        if self.normed_residual <= self.settings.tolerance:
            return True

        if self._current_iter >= self.settings.max_mda_iter:
            LOGGER.warning(
                "%s has reached its maximum number of iterations, "
                "but the normalized residual norm %s is still above the tolerance %s.",
                self.name,
                self.normed_residual,
                self.settings.tolerance,
            )
            return True

        if (
            self.__n_consecutive_unsuccessful_iterations
            >= self.settings.max_consecutive_unsuccessful_iterations
        ):
            LOGGER.warning(
                "%s has reached its maximum number of unsuccessful iterations, "
                "but the normalized residual norm %s is still above the tolerance %s.",
                self.name,
                self.normed_residual,
                self.settings.tolerance,
            )
            return True

        return False

    def _set_resolved_variables(self, resolved_couplings: Iterable[str]) -> None:
        """Set the resolved variables and associated residuals.

        The state variables are added to the provided coupling variable to form the list
        of resolved variables. The associated residuals are either identical to coupling
        variable or, for state variables, retrieved from the corresponding discipline.

        The order of resolved variable names and residuals names is consistent.

        Args:
            resolved_couplings: The name of coupling variables resolved by the MDA.
        """
        # Aggregate the residual variables from disciplines
        residual_variables = {}
        for discipline in self._disciplines:
            if discipline.io.state_equations_are_solved:
                continue

            residual_variables.update(discipline.io.residual_to_state_variable)

        state_variables = residual_variables.values()
        resolved_variables = set(resolved_couplings).union(state_variables)
        resolved_variables.difference_update(self._non_numeric_array_variables)
        self.__resolved_variable_names = tuple(sorted(resolved_variables))

        # State variable names are replaced with associated residual names
        residuals = list(self._resolved_variable_names)
        for key, value in residual_variables.items():
            # The order is maintained to guarantee consistency
            if value in residuals:
                residuals[residuals.index(value)] = key

        self.__resolved_residual_names = tuple(residuals)

    def _compute_names_to_slices(self) -> None:
        """Compute the mapping of variable names to slices for converting data to array.

        Two mappings are computed, one for the resolved variables (couplings and state
        variables), one for the associated residuals.

        The mappings are cached and computed only once. When possible, the unique
        grammar (input or output) of a converter that contains all the coupling data is
        chosen. Otherwise, converters from the 2 grammars are used.
        """
        if self.__resolved_variable_names_to_slices:
            return

        self.__resolved_variable_names_to_slices = (
            self.io.input_grammar.data_converter.compute_names_to_slices(
                self._resolved_variable_names,
                self.io.data,
            )[0]
        )

        # Initialize the vectors of bounds once the variable sizes are known.
        total_size = sum(
            slice_.stop - slice_.start
            for slice_ in self.__resolved_variable_names_to_slices.values()
        )

        self.__lower_bound_vector = -inf * ones(total_size)
        self.__upper_bound_vector = +inf * ones(total_size)

        self.__update_bounds_vectors()

    def _update_local_data_from_array(self, array_: ndarray) -> None:
        """Update the local data from an array.

        Args:
            array_: An array.
        """
        self.io.data |= self.io.output_grammar.data_converter.convert_array_to_data(
            array_,
            self.__resolved_variable_names_to_slices,
        )

    def _compute_normalized_residual_norm(self) -> float:
        """Compute the normalized residual norm at the current point.

        Returns:
            The normalized residual norm.
        """
        residual = self.get_current_resolved_residual_vector()

        scaling = self.scaling
        scaling_data = self._scaling_data
        ResidualScaling = self.ResidualScaling  # noqa: N806

        if scaling == ResidualScaling.NO_SCALING:
            normed_residual = float(norm(residual))

        elif scaling == ResidualScaling.INITIAL_RESIDUAL_NORM:
            normed_residual = float(norm(residual))

            if scaling_data is None:
                scaling_data = normed_residual if normed_residual != 0 else 1.0

            normed_residual /= scaling_data

        elif scaling == ResidualScaling.N_COUPLING_VARIABLES:
            if scaling_data is None:
                scaling_data = residual.size**0.5
            normed_residual = norm(residual) / scaling_data

        elif scaling == ResidualScaling.INITIAL_SUBRESIDUAL_NORM:
            if scaling_data is None:
                scaling_data = []
                for slice_ in self.__resolved_variable_names_to_slices.values():
                    initial_norm = float(norm(residual[slice_]))
                    initial_norm = initial_norm if initial_norm != 0.0 else 1.0
                    scaling_data.append((slice_, initial_norm))

            normalized_norms = []
            for current_slice, initial_norm in scaling_data:
                normalized_norms.append(norm(residual[current_slice]) / initial_norm)

            normed_residual = max(normalized_norms)

        elif scaling == ResidualScaling.INITIAL_RESIDUAL_COMPONENT:
            if scaling_data is None:
                scaling_data = residual + (residual == 0)

            normed_residual = np_abs(residual / scaling_data).max()

        elif scaling == ResidualScaling.SCALED_INITIAL_RESIDUAL_COMPONENT:
            if scaling_data is None:
                scaling_data = residual + (residual == 0)

            normed_residual = float(norm(residual / scaling_data))
            normed_residual /= residual.size**0.5

        else:
            # Use the StrEnum casting to raise an explicit error.
            ResidualScaling(scaling)

        self._scaling_data = scaling_data

        return normed_residual

    @abstractmethod
    def _execute(self) -> None:  # noqa:D103
        super()._execute()
        self._sequence_transformer.clear()
        self.__n_consecutive_unsuccessful_iterations = 0

    def _compute_residuals(self, input_data: MutableStrKeyMapping) -> None:
        """Compute the residual vector.

        Residuals are computed either as the difference between input and output values
        of coupling variables, or, for state variables, directly from the associated
        residual variable.

        This method should be used only after the disciplines' local data have been
        modified, otherwise there will be no difference between input and output values.

        Args:
            input_data: The input data to compute residual of coupling variables.
        """
        convert_data_to_array = (
            self.io.output_grammar.data_converter.convert_data_to_array
        )

        for residual_name in self._resolved_residual_names:
            residual = convert_data_to_array([residual_name], self.io.data)
            if residual_name in self._resolved_variable_names:
                # No -= assignment to avoid possible casting problems.
                residual = residual - convert_data_to_array([residual_name], input_data)

            self._current_residuals[residual_name] = residual

    def __update_bounds_vectors(self) -> None:
        """Set the bounds of the sequence transformer."""
        if self.__lower_bound_vector is None or self.__upper_bound_vector is None:
            return

        for name, (
            lower_bound,
            upper_bound,
        ) in self.__resolved_variable_names_to_bounds.items():
            slice_ = self.__resolved_variable_names_to_slices[name]
            self.__lower_bound_vector[slice_] = (
                lower_bound if lower_bound is not None else -inf
            )
            self.__upper_bound_vector[slice_] = (
                upper_bound if upper_bound is not None else +inf
            )

        self._sequence_transformer.lower_bound = self.__lower_bound_vector
        self._sequence_transformer.upper_bound = self.__upper_bound_vector

    def __update_iteration_metrics(self) -> None:
        """Update the iteration metrics."""
        if self._current_iter == 0:
            self._starting_indices.append(len(self.residual_history))

            if self.reset_history_each_run:
                self._starting_indices.clear()
                self.residual_history.clear()

        old_normalized_residual_norm = copy(self.normed_residual)
        self.normed_residual = self._compute_normalized_residual_norm()

        if self.normed_residual >= old_normalized_residual_norm:
            self.__n_consecutive_unsuccessful_iterations += 1
        else:
            self.__n_consecutive_unsuccessful_iterations = 0

        self.residual_history.append(self.normed_residual)
        self._current_iter += 1

        self.io.update_output_data({
            self.NORMALIZED_RESIDUAL_NORM: array([self.normed_residual])
        })

        if self.settings.log_convergence:
            msg = "{} running... Normalized residual norm = {} (iter. {})"
            LOGGER.info(
                msg.format(self.name, f"{self.normed_residual:.2e}", self._current_iter)
            )
