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
"""The base class for MDA solvers."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Iterable

from numpy import abs
from numpy import array
from numpy import concatenate
from numpy import ndarray
from numpy.linalg import norm

from gemseo.mda.mda import MDA

if TYPE_CHECKING:
    from gemseo.core.data_converters.base import BaseDataConverter

LOGGER = logging.getLogger(__name__)


class BaseMDASolver(MDA):
    """The base class for MDA solvers."""

    __coupling_names_to_slices: dict[BaseDataConverter, dict[str, slice]]
    """The mapping from names to slices for converting array to data structures.

    Since the coupling data may have inputs and / or outputs, there is one mapping per
    converter because a converter is bound to a grammar.
    """

    __resolved_coupling_names: tuple[str, ...] | None
    """The names of the coupling variables the MDA is solving."""

    def __init__(self, *args, **kwargs):  # noqa:D107
        super().__init__(*args, **kwargs)
        self.__resolved_coupling_names = None
        self.__coupling_names_to_slices = {}

    @property
    def _resolved_coupling_names(self) -> tuple[str] | None:
        """The names of the coupling variables the MDA is solving.

        Raises:
            RuntimeError: Whenever one attempt to set again the resolved coupling names.
        """
        return self.__resolved_coupling_names

    @_resolved_coupling_names.setter
    def _resolved_coupling_names(self, couplings: Iterable[str]) -> None:
        if self.__resolved_coupling_names is not None:
            raise RuntimeError(
                "The resolved coupling names have already been set, any changes could "
                "make the MDA solution inconsistent."
            )

        self.__resolved_coupling_names = tuple(sorted(couplings))

    def _warn_convergence_criteria(self) -> tuple[bool, bool]:
        """Log a warning if max_iter is reached and if max residuals is above tolerance.

        Returns:
            * Whether the normed residual is lower than the tolerance.
            * Whether the maximum number of iterations is reached.
        """
        residual_is_small = self.normed_residual <= self.tolerance
        max_iter_is_reached = self.max_mda_iter <= self._current_iter
        if max_iter_is_reached and not residual_is_small:
            msg = (
                "%s has reached its maximum number of iterations "
                "but the normed residual %s is still above the tolerance %s."
            )
            LOGGER.warning(msg, self.name, self.normed_residual, self.tolerance)
        return residual_is_small, max_iter_is_reached

    @property
    def _stop_criterion_is_reached(self) -> bool:
        """Whether a stop criterion is reached."""
        residual_is_small, max_iter_is_reached = self._warn_convergence_criteria()
        return residual_is_small or max_iter_is_reached

    def _current_working_couplings(self) -> ndarray:
        """Return the current values of the working coupling variables."""
        if not self.__resolved_coupling_names:
            return array([])

        self.__compute_names_to_slices()

        arrays = []

        for (
            converter,
            couplings_names_to_slices,
        ) in self.__coupling_names_to_slices.items():
            if couplings_names_to_slices:
                arrays += [
                    converter.convert_data_to_array(
                        couplings_names_to_slices,
                        self._local_data,
                    )
                ]

        return concatenate(arrays)

    def __compute_names_to_slices(self) -> None:
        """Compute the mapping of coupling names to slices for converting data to array.

        The mapping is cached and only computed once. When possible, the unique grammar
        (input or output) of a converter that contains all the coupling data is chosen.
        Otherwise, converters from the 2 grammars are used.
        """
        if self.__coupling_names_to_slices:
            return

        resolved_coupling_names = set(self._resolved_coupling_names)

        if resolved_coupling_names.issubset(self.input_grammar.keys()):
            input_coupling_names = self._resolved_coupling_names
            output_coupling_names = ()
        elif resolved_coupling_names.issubset(self.output_grammar.keys()):
            input_coupling_names = ()
            output_coupling_names = self._resolved_coupling_names
        else:
            input_coupling_names = sorted(
                resolved_coupling_names.intersection(self.input_grammar.keys())
            )
            output_coupling_names = sorted(
                resolved_coupling_names.difference(input_coupling_names)
            )

        if input_coupling_names:
            converter = self.input_grammar.data_converter
            self.__coupling_names_to_slices[
                converter
            ] = converter.compute_names_to_slices(
                input_coupling_names, self._local_data
            )[0]

        if output_coupling_names:
            converter = self.output_grammar.data_converter
            self.__coupling_names_to_slices[
                converter
            ] = converter.compute_names_to_slices(
                output_coupling_names,
                self._local_data,
            )[0]

    def _update_local_data(self, array_: ndarray) -> None:
        """Update the local data from an array.

        Args:
            array_: An array.
        """
        for (
            converter,
            couplings_names_to_slices,
        ) in self.__coupling_names_to_slices.items():
            self._local_data.update(
                converter.convert_array_to_data(array_, couplings_names_to_slices)
            )

    def _end_iteration(
        self,
        current_couplings: ndarray,
        new_couplings: ndarray,
    ) -> tuple[ndarray, bool]:
        """Execute the final steps of an MDA iteration.

        Args:
            current_couplings: The current couplings.
            new_couplings: The new couplings.

        Returns:
            The up-to-date couplings and whether the iterating process shall stop.
        """
        self._update_local_data(new_couplings)

        self._compute_residual(
            current_couplings,
            new_couplings,
            log_normed_residual=self._log_convergence,
        )

        if self._stop_criterion_is_reached:
            return new_couplings, True

        return new_couplings, False

    def _compute_residual(
        self,
        current_couplings: ndarray,
        new_couplings: ndarray,
        store_it: bool = True,
        log_normed_residual: bool = False,
    ) -> float:
        """Compute the residual on the inputs of the MDA.

        Args:
            current_couplings: The values of the couplings before the execution.
            new_couplings: The values of the couplings after the execution.
            store_it: Whether to store the normed residual.
            log_normed_residual: Whether to log the normed residual.

        Returns:
            The normed residual.
        """
        if self._current_iter == 0 and self.reset_history_each_run:
            self.residual_history = []
            self._starting_indices = []

        residual = (current_couplings - new_couplings).real

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
                _scaling_data = new_couplings.size**0.5
            normed_residual = norm(residual) / _scaling_data

        elif scaling == ResidualScaling.INITIAL_SUBRESIDUAL_NORM:
            if _scaling_data is None:
                _scaling_data = []

                for (
                    coupling_names_to_slices
                ) in self.__coupling_names_to_slices.values():
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

            normed_residual = abs(residual / _scaling_data).max()

        elif scaling == ResidualScaling.SCALED_INITIAL_RESIDUAL_COMPONENT:
            if _scaling_data is None:
                _scaling_data = residual + (residual == 0)

            normed_residual = float(norm(residual / _scaling_data))
            normed_residual /= new_couplings.size**0.5

        else:
            # Use the StrEnum casting to raise an explicit error.
            ResidualScaling(scaling)

        self.normed_residual = normed_residual
        self._scaling_data = _scaling_data

        if log_normed_residual:
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

        self._local_data[self.RESIDUALS_NORM] = array([self.normed_residual])

        return self.normed_residual

    @abstractmethod
    def _run(self) -> None:  # noqa:D103
        super()._run()
        self._sequence_transformer.clear()
