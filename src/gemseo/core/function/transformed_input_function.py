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

"""A function taking its input in the working coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.function._blocks import transform_blocks
from gemseo.core.function._nan import check_for_nan
from gemseo.core.function.base_problem_function import BaseProblemFunction

if TYPE_CHECKING:
    from gemseo.core.problem.database import Database
    from gemseo.space.transformation.base import BaseSpaceTransformation
    from gemseo.util.typing import NumberArray


class TransformedInputFunction(BaseProblemFunction):
    """A function adapting its input from the working coordinates to the original ones.

    This is the *adaptation* half of the evaluation of a problem.
    It holds the transformation,
    and it knows nothing of the problem, of its database or of its counters:
    recording is the business of the other half,
    which it wraps.
    Stopping a run on a `NaN` is split between the two halves:
    this one checks the point in the working coordinates,
    before mapping it back,
    since that map may cast it to integers and hide a `NaN`;
    the half it wraps checks what the function returns,
    under the setting this one passes on.

    The function itself is not transformed.
    Only its argument is,
    and by the chain rule its Jacobian:
    `f(x)` returns the same value whichever coordinates it is asked in.
    """

    _wrapped_function: BaseProblemFunction
    """The half this one wraps, which it reads the counters and the settings from."""

    __working_database: Database | None
    """The store of the evaluations in the working coordinates, if any."""

    __transformation: BaseSpaceTransformation
    """The map from the original coordinates to the working ones."""

    def __init__(
        self,
        function: BaseProblemFunction,
        transformation: BaseSpaceTransformation,
        stop_if_nan: bool = True,
        working_database: Database | None = None,
    ) -> None:
        """
        Args:
            function: The half to wrap,
                which evaluates the function in the original coordinates.
                This is an evaluation half,
                or another adaptation half when the transformations are stacked.
            transformation: The map from the original coordinates to the working ones.
            stop_if_nan: Whether to stop the evaluation
                when a function returns `NaN`.
            working_database: The store of the evaluations
                in the working coordinates.
                If `None`, record nothing of them.
        """  # noqa: D205, D212
        self.__transformation = transformation
        self.__working_database = working_database
        super().__init__(
            function,
            # A point of the working space is what this function takes,
            # and the problem it belongs to works in that space:
            # there is nothing left to normalize on the way in.
            with_normalized_inputs=False,
        )
        self.stop_if_nan = stop_if_nan
        # The `original` of the half this one wraps,
        # which already stops at the right function.
        self.original = function.original

    @property
    def stop_if_nan(self) -> bool:
        """Whether the evaluation stops when the function returns a `NaN`.

        The setting belongs to the evaluation half,
        so this reads and writes through to it,
        as a problem whose functions are adapted holds this half alone:
        were the setting not passed on,
        a driver turning the stop off would still be stopped by the other half.
        """
        return self._wrapped_function.stop_if_nan

    @stop_if_nan.setter
    def stop_if_nan(self, value: bool) -> None:
        self._wrapped_function.stop_if_nan = value

    @property
    def enable_statistics(self) -> bool:
        """Whether the calls of the function are counted.

        Counting belongs to the evaluation half,
        so this reads that half.
        """
        return self._wrapped_function.enable_statistics

    @property
    def n_calls(self) -> int:
        """The number of times the function has been evaluated.

        This layer adapts the argument and records nothing,
        so the count is the one of the half it wraps.
        """
        return self._wrapped_function.n_calls

    @n_calls.setter
    def n_calls(self, value: int) -> None:
        self._wrapped_function.n_calls = value

    @property
    def _vectorize(self) -> bool:
        """Whether the function is evaluated on a matrix of samples.

        Read from the half this one wraps
        rather than passed in again,
        so that both cut the same samples out of a matrix by construction.
        """
        return self._wrapped_function._vectorize

    @property
    def _gradient_name(self) -> str:
        """The name under which a Jacobian is recorded in a database.

        Read from the half this one wraps.
        """
        return self._wrapped_function._gradient_name

    def _compute_output(self, input_value: NumberArray) -> NumberArray:
        """Compute an output value from a value of the working space.

        Args:
            input_value: The input value, in the working coordinates.

        Returns:
            The output value.
        """
        # Checked here,
        # before the map,
        # which may cast the point to integers and turn a NaN into an
        # ordinary, huge value that the check of the half below would miss.
        check_for_nan(input_value)
        original_value = self.__transformation.inverse_transform_value(input_value)
        # `evaluate` rather than `func`,
        # so that the half that records counts this as one evaluation,
        # exactly as it did before the two were split.
        output_value = self._wrapped_function.evaluate(original_value)
        # The output value is not checked here:
        # the half just called checks what the function returns,
        # under the very setting this one passed it.
        self.__record(input_value, self.name, output_value)
        return output_value

    def _compute_jacobian(self, input_value: NumberArray) -> NumberArray:
        """Compute a Jacobian from a value of the working space.

        The point is handed to the tangent map,
        which a non-affine transformation needs.
        This is why the Jacobian path is written out
        rather than flattened into a sequence of one-argument callables.

        Args:
            input_value: The input value, in the working coordinates.

        Returns:
            The Jacobian with respect to the working coordinates.
        """
        # Checked here,
        # before the map,
        # for the same reason as the input point in `_compute_output`.
        check_for_nan(input_value)
        original_value = self.__transformation.inverse_transform_value(input_value)
        # The Jacobian is not checked here:
        # the half just called checks it,
        # for the same reason as the output value in `_compute_output`.
        jacobian = self._wrapped_function.jac(original_value)
        if self._vectorize:
            # The Jacobian of a matrix of samples is the block diagonal matrix
            # of their own,
            # so the map of a single Jacobian is applied to every block at once.
            working_jacobian = transform_blocks(
                jacobian,
                len(original_value),
                original_value.shape[1],
                lambda blocks: self.__transformation.transform_jacobian(
                    blocks, original_value
                ),
            )
        else:
            working_jacobian = self.__transformation.transform_jacobian(
                jacobian, original_value
            )

        self.__record(input_value, self._gradient_name, working_jacobian)
        return working_jacobian

    def __record(self, input_value: NumberArray, name: str, value: NumberArray) -> None:
        """Record a value in the working coordinates, when asked to.

        A vectorized evaluation is recorded sample by sample,
        as the evaluation half records it in the coordinates the user declared.

        Args:
            input_value: The input value, in the working coordinates.
            name: The name under which to record the value.
            value: The value to record.
        """
        working_database = self.__working_database
        if working_database is None:
            return

        if not self._vectorize:
            working_database.store(
                working_database.get_hashable_ndarray(input_value), {name: value}
            )
            return

        working_database.store_batch(
            input_value, value, name, is_jacobian=name == self._gradient_name
        )
