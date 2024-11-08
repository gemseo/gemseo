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
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import abs as np_abs
from numpy import array
from numpy import concatenate
from numpy import ndarray
from numpy.linalg import norm

from gemseo.algos.sequence_transformer.composite.relaxation_acceleration import (
    RelaxationAcceleration,
)
from gemseo.mda.base_mda import BaseMDA

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
    from gemseo.core.data_converters.base import BaseDataConverter
    from gemseo.core.discipline import Discipline
    from gemseo.mda.base_mda_solver_settings import BaseMDASolverSettings
    from gemseo.typing import MutableStrKeyMapping

LOGGER = logging.getLogger(__name__)


class BaseMDASolver(BaseMDA):
    """The base class for MDA solvers."""

    Settings: ClassVar[type[BaseMDASolverSettings]]
    """The Pydantic model for the settings."""

    settings: BaseMDASolverSettings
    """The settings of the MDA."""

    _sequence_transformer: RelaxationAcceleration
    """The sequence transformer aimed at improving the convergence rate.

    The transformation applies a relaxation followed by an acceleration.
    """

    __resolved_variable_names_to_slices: dict[BaseDataConverter, dict[str, slice]]
    """The mapping from names to slices for converting array to data structures.

    Since the coupling data may have inputs and / or outputs, there is one mapping per
    converter because a converter is bound to a grammar.
    """

    __resolved_variable_names: tuple[str, ...]
    """The names of the resolved variables.

    Resolved variables are coupling and state variables (for disciplines that does not
    solve their own residuals). These variables are modified by the MDA so as to make
    the corresponding residuals converge towards 0.
    """

    __resolved_residual_names_to_slices: dict[BaseDataConverter, dict[str, slice]]
    """The mapping from residual names to slices for converting array to data structure.

    Since the resolved residual data may have inputs and/or outputs, there is one
    mapping per converter because a converter is bound to a grammar.
    """

    __resolved_residual_names: tuple[str, ...]
    """The names of the resolved residuals the MDA is solving.

    Resolved residuals are either related to coupling variables or explicitly attached
    to state variables (for disciplines that does not solve their own residuals). The
    MDA is meant to bring these residuals towards zero.
    """

    _current_residuals: dict[str, ndarray]
    """The mapping from residual names to current value."""

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

        self.__resolved_residual_names_to_slices = {}
        self.__resolved_residual_names = ()

        self._current_residuals = {}

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
    def _resolved_variable_names(self) -> tuple[str, ...]:
        """The names of the variables (couplings and state) the MDA is solving."""
        return self.__resolved_variable_names

    @property
    def _resolved_residual_names(self) -> tuple[str, ...]:
        """The names of the residuals (couplings and residuals) the MDA is solving."""
        return self.__resolved_residual_names

    def get_current_resolved_variables_vector(self) -> ndarray:
        """Return the vector of resolved variables (couplings and state variables)."""
        if not self.__resolved_variable_names:
            return array([])

        self.__compute_names_to_slices()

        arrays = []

        for (
            converter,
            couplings_names_to_slices,
        ) in self.__resolved_variable_names_to_slices.items():
            if couplings_names_to_slices:
                arrays += [
                    converter.convert_data_to_array(
                        couplings_names_to_slices,
                        self.io.data,
                    )
                ]

        return concatenate(arrays)

    def get_current_resolved_residual_vector(self) -> ndarray:
        """Return the vector of residuals."""
        if not self.__resolved_residual_names:
            return array([])

        self.__compute_names_to_slices()

        arrays = []

        for (
            converter,
            couplings_names_to_slices,
        ) in self.__resolved_residual_names_to_slices.items():
            if couplings_names_to_slices:
                arrays += [
                    converter.convert_data_to_array(
                        couplings_names_to_slices,
                        self._current_residuals,
                    )
                ]

        return concatenate(arrays)

    def _warn_convergence_criteria(self) -> tuple[bool, bool]:
        """Log a warning if max_iter is reached and if max residuals is above tolerance.

        Returns:
            * Whether the normed residual is lower than the tolerance.
            * Whether the maximum number of iterations is reached.
        """
        residual_is_small = self.normed_residual <= self.settings.tolerance
        max_iter_is_reached = self.settings.max_mda_iter <= self._current_iter
        if max_iter_is_reached and not residual_is_small:
            msg = (
                "%s has reached its maximum number of iterations "
                "but the normed residual %s is still above the tolerance %s."
            )
            LOGGER.warning(
                msg, self.name, self.normed_residual, self.settings.tolerance
            )
        return residual_is_small, max_iter_is_reached

    @property
    def _stop_criterion_is_reached(self) -> bool:
        """Whether a stop criterion is reached."""
        self._compute_normalized_residual_norm()
        residual_is_small, max_iter_is_reached = self._warn_convergence_criteria()
        return residual_is_small or max_iter_is_reached

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
        for discipline in self.disciplines:
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

    def __compute_names_to_slices(self) -> None:
        """Compute the mapping of variable names to slices for converting data to array.

        Two mappings are computed, one for the resolved variables (couplings and state
        variables), one for the associated residuals.

        The mappings are cached and computed only once. When possible, the unique
        grammar (input or output) of a converter that contains all the coupling data is
        chosen. Otherwise, converters from the 2 grammars are used.
        """
        if self.__resolved_variable_names_to_slices:
            return

        resolved_coupling_names = set(self._resolved_variable_names)

        if resolved_coupling_names.issubset(self.input_grammar.keys()):
            input_coupling_names = self._resolved_variable_names
            output_coupling_names = ()
        elif resolved_coupling_names.issubset(self.output_grammar.keys()):
            input_coupling_names = ()
            output_coupling_names = self._resolved_variable_names
        else:
            input_coupling_names = sorted(
                resolved_coupling_names.intersection(self.input_grammar.keys())
            )
            output_coupling_names = sorted(
                resolved_coupling_names.difference(input_coupling_names)
            )

        if input_coupling_names:
            converter = self.input_grammar.data_converter
            self.__resolved_variable_names_to_slices[converter] = (
                converter.compute_names_to_slices(input_coupling_names, self.io.data)[0]
            )

        if output_coupling_names:
            converter = self.output_grammar.data_converter
            self.__resolved_variable_names_to_slices[converter] = (
                converter.compute_names_to_slices(
                    output_coupling_names,
                    self.io.data,
                )[0]
            )

        self.__resolved_residual_names_to_slices = {}
        for (
            converter,
            names_to_slices,
        ) in self.__resolved_variable_names_to_slices.items():
            self.__resolved_residual_names_to_slices[converter] = {}

            for name, slice_ in names_to_slices.items():
                if name in self.__resolved_residual_names:
                    self.__resolved_residual_names_to_slices[converter][name] = slice_
                else:
                    residual = self.__resolved_residual_names[
                        self.__resolved_variable_names.index(name)
                    ]
                    self.__resolved_residual_names_to_slices[converter][residual] = (
                        slice_
                    )

    def _update_local_data_from_array(self, array_: ndarray) -> None:
        """Update the local data from an array.

        Args:
            array_: An array.
        """
        for (
            converter,
            couplings_names_to_slices,
        ) in self.__resolved_variable_names_to_slices.items():
            self.io.data.update(
                converter.convert_array_to_data(array_, couplings_names_to_slices)
            )

    def _compute_normalized_residual_norm(self, store_it: bool = True) -> float:
        """Compute the normalized residual norm at the current point.

        Args:
            store_it: Whether to store the normed residual.

        Returns:
            The normed residual.
        """
        if self._current_iter == 0 and self.reset_history_each_run:
            self.residual_history = []
            self._starting_indices = []

        residual = self.get_current_resolved_residual_vector()

        scaling = self.scaling
        _scaling_data = self._scaling_data
        ResidualScaling = self.ResidualScaling  # noqa: N806

        if scaling == ResidualScaling.NO_SCALING:
            normed_residual = float(norm(residual))

        elif scaling == ResidualScaling.INITIAL_RESIDUAL_NORM:
            normed_residual = float(norm(residual))

            if _scaling_data is None:
                _scaling_data = normed_residual if normed_residual != 0 else 1.0

            normed_residual /= _scaling_data

        elif scaling == ResidualScaling.N_COUPLING_VARIABLES:
            if _scaling_data is None:
                _scaling_data = residual.size**0.5
            normed_residual = norm(residual) / _scaling_data

        elif scaling == ResidualScaling.INITIAL_SUBRESIDUAL_NORM:
            if _scaling_data is None:
                _scaling_data = []

                for (
                    coupling_names_to_slices
                ) in self.__resolved_variable_names_to_slices.values():
                    for slice_ in coupling_names_to_slices.values():
                        initial_norm = float(norm(residual[slice_]))
                        initial_norm = initial_norm if initial_norm != 0.0 else 1.0
                        _scaling_data.append((slice_, initial_norm))

            normalized_norms = []
            for current_slice, initial_norm in _scaling_data:
                normalized_norms.append(norm(residual[current_slice]) / initial_norm)

            normed_residual = max(normalized_norms)

        elif scaling == ResidualScaling.INITIAL_RESIDUAL_COMPONENT:
            if _scaling_data is None:
                _scaling_data = residual + (residual == 0)

            normed_residual = np_abs(residual / _scaling_data).max()

        elif scaling == ResidualScaling.SCALED_INITIAL_RESIDUAL_COMPONENT:
            if _scaling_data is None:
                _scaling_data = residual + (residual == 0)

            normed_residual = float(norm(residual / _scaling_data))
            normed_residual /= residual.size**0.5

        else:
            # Use the StrEnum casting to raise an explicit error.
            ResidualScaling(scaling)

        self.normed_residual = normed_residual
        self._scaling_data = _scaling_data

        if self.settings.log_convergence:
            LOGGER.info(
                "%s running... Normed residual = %s (iter. %s)",
                self.name,
                f"{self.normed_residual:.2e}",
                self._current_iter,
            )

        if store_it:
            if self._current_iter == 0:
                self._starting_indices.append(len(self.residual_history))
            self.residual_history.append(self.normed_residual)
            self._current_iter += 1

        self.io.update_output_data({
            self.NORMALIZED_RESIDUAL_NORM: array([self.normed_residual])
        })

        return self.normed_residual

    @abstractmethod
    def _execute(self) -> None:  # noqa:D103
        super()._execute()
        self._sequence_transformer.clear()

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
        self.__compute_names_to_slices()

        for (
            converter,
            couplings_names_to_slices,
        ) in self.__resolved_residual_names_to_slices.items():
            for name in couplings_names_to_slices:
                local_data_array = converter.convert_data_to_array([name], self.io.data)
                if name in self._resolved_variable_names:
                    input_data_array = converter.convert_data_to_array(
                        [name], input_data
                    )
                    residual = local_data_array - input_data_array
                else:
                    residual = local_data_array
                self._current_residuals[name] = residual
