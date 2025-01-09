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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Isabelle Santos
#        :author: Giulio Gargantini
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

r"""The discipline describing the dynamics of a single mass connected by springs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import asarray
from numpy import isscalar
from numpy import linspace
from scipy.interpolate import interp1d

from gemseo.core.discipline import Discipline
from gemseo.core.discipline.base_discipline import CacheType

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping

POSITION: Final[str] = "position"
"""The name of the state variable representing the position."""

VELOCITY: Final[str] = "velocity"
"""The name of the state variable representing the velocity."""

TIME: Final[str] = "time"
"""The name of the time variable."""

STATE_NAMES: Final[tuple[str, str]] = (POSITION, VELOCITY)
"""The names of the state variables."""

STATE_DOT_NAMES: Final[tuple[str, str]] = (f"{POSITION}_dot", f"{VELOCITY}_dot")
"""The names of the derivatives of the state variables."""

_times: Final[RealArray] = linspace(0.0, 10, 30)
INITIAL_TIME: Final[RealArray] = asarray([0.0])
INITIAL_POSITION: Final[RealArray] = asarray([0.0])
INITIAL_VELOCITY: Final[RealArray] = asarray([0.0])


class SpringsDynamicsDiscipline(Discipline):
    """A discipline defining the dynamics of a mass connected to two springs."""

    _mass: float
    """The value of the oscillating mass."""

    _left_stiff: float
    """The stiffness of the spring to the left."""

    _right_stiff: float
    """The stiffness of the spring to the right."""

    _left_position: RealArray | float
    """The position of the other end of the spring on the left.

    If the position of the other extremity is constant in time, it is a float.
    Otherwise, the trajectory of said position in time is represented by an array,
    tracking the position of the other extremity of the spring in the instants defined
    in the vector '_time'.
    """

    _right_position: RealArray | float
    """The position of the other end of the spring on the right.

    If the position of the other extremity is constant in time, it is a float.
    Otherwise, the trajectory of said position in time is represented by an array,
    tracking the position of the other extremity of the spring in the instants defined
    in the vector '_time'.
    """

    _times: RealArray
    """The times where the position of the oscillating mass is evaluated."""

    _state_names: Sequence[str]
    """The names of the state variables.

    By default, the names of the state variables are 'position' and 'velocity'."""

    _state_dot_var_names: Sequence[str]
    """The names of the time derivatives of the state variables.

    By default, the names of the time derivatives of the state variables are
    'position_dot' and 'velocity_dot'."""

    _left_position_name: str
    """Name of the discipline input defining the position of the other extremity of the
    spring to the left.

    If the position of said extremity is not an input of the discipline,
    '_left_position_name' is None."""

    _right_position_name: str
    """Name of the discipline input defining the position of the other extremity of the
    spring to the right.

    If the position of said extremity is not an input of the discipline,
    '_right_position_name' is None."""

    def __init__(
        self,
        mass: float,
        left_stiffness: float,
        right_stiffness: float,
        times: RealArray | None = None,
        left_position: RealArray | float = 0.0,
        right_position: RealArray | float = 0.0,
        is_left_position_fixed=False,
        is_right_position_fixed=False,
        state_names: Sequence[str] = STATE_NAMES,
        state_dot_var_names: Sequence[str] = STATE_DOT_NAMES,
        left_position_name: str = "",
        right_position_name: str = "",
        **kwargs,
    ) -> None:
        """Args:
            mass: The value of the mass.
            left_stiffness: The stiffness of the spring on the left-hand side.
            right_stiffness: The stiffness of the spring on the right-hand side.
            left_position: The position of the other extremity
                of the spring to the left.
            right_position: The position of the other extremity
                of the spring to the right.
            is_left_position_fixed: True if the other end of the spring
                to the left is fixed.
            is_right_position_fixed: True if the other end of the spring
                to the right is fixed.
            times: the time vector for the evaluation of the left and right positions.
            state_names: The names of the state variables
                (by default 'position' and 'velocity').
            state_dot_var_names: The names of the time derivatives of
                the state variables (by default 'position_dot' and 'velocity_dot').
            left_position_name: Name of the input describing the position of the mass
                on the left, if is_left_position_fixed is False.
            right_position_name: Name of the input describing the position of the mass
                on the right, if is_right_position_fixed is False.

        Returns:
            The Discipline describing a single point mass.
        """  # noqa: D205, D212, D415
        self._mass = mass
        self._left_stiff = left_stiffness
        self._right_stiff = right_stiffness
        self._left_position = left_position
        self._right_position = right_position
        self._times = times
        self._state_names = state_names
        self._state_dot_var_names = state_dot_var_names
        self._left_position_name = "" if is_left_position_fixed else left_position_name
        self._right_position_name = (
            "" if is_right_position_fixed else right_position_name
        )

        input_names = [TIME, *self._state_names]
        output_names = state_dot_var_names

        super().__init__(**kwargs)

        if not is_left_position_fixed:
            input_names.append(left_position_name)
        if not is_right_position_fixed:
            input_names.append(right_position_name)
        self.io.input_grammar.update_from_names(input_names)
        self.io.output_grammar.update_from_names(output_names)

        self.default_input_data = {
            TIME: INITIAL_TIME,
            self._state_names[0]: INITIAL_POSITION,
            self._state_names[1]: INITIAL_VELOCITY,
        }
        if not is_left_position_fixed and self._times is not None:
            self.default_input_data[left_position_name] = self._times * 0.0
        if not is_right_position_fixed and self._times is not None:
            self.default_input_data[right_position_name] = self._times * 0.0

        self.add_differentiated_inputs(["time", *state_names])

        self.set_cache(cache_type=CacheType.NONE)

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping:
        input_data = {
            input_name: self.io.data[input_name]
            for input_name in self.io.input_grammar.names
        }

        position_dot, velocity_dot = self._compute_generic_mass_rhs_function(
            time=input_data[TIME],
            position=input_data[self._state_names[0]],
            velocity=input_data[self._state_names[1]],
            mass=self._mass,
            left_stiff=self._left_stiff,
            right_stiff=self._right_stiff,
            left_position=input_data.get(self._left_position_name, self._left_position),
            right_position=input_data.get(
                self._right_position_name, self._right_position
            ),
            times=self._times,
        )

        return {
            self._state_dot_var_names[0]: position_dot,
            self._state_dot_var_names[1]: velocity_dot,
        }

    @staticmethod
    def _compute_generic_mass_rhs_function(
        time: RealArray,
        position: RealArray,
        velocity: RealArray,
        mass: float = 1.0,
        left_stiff: float = 1.0,
        right_stiff: float = 1.0,
        left_position: RealArray | float = 0.0,
        right_position: RealArray | float = 0.0,
        times: RealArray | None = None,
    ) -> tuple[RealArray, RealArray]:
        """Right-hand side function of the differential equation for a single mass.

        Args:
            time: The time at which the right-hand side function is evaluated.
            position: The position of the mass at `time`.
            velocity: The velocity of the mass at `time`.
            mass: The value of the mass.
            left_stiff: The stiffness of the spring on the left-hand side of the mass.
            right_stiff: The stiffness of the spring on the right-hand side of the mass.
            left_position: The position of the mass on the left-hand side of the
                current mass at `time`.
            right_position: The position of the mass on the right-hand side
                of the current mass at `time`.
            times: The times to compute `left_position` and `right_position` at.
                This parameter is only used if `left_position` and `right_position` are
                vectors. This is not the same `times` as in ODEDiscipline!

        Returns:
            Vector containing the derivative of `state` (i.e. of the position
            and velocity) at `time`.
        """
        if isscalar(right_position) or times is None:
            right_position = right_position
        else:
            msg_error = "Incoherent lengths of times and right_position"
            assert times.size == right_position.size, msg_error
            interpolated_function = interp1d(times, right_position, assume_sorted=True)
            right_position = interpolated_function(time)

        if isscalar(left_position) or times is None:
            left_position = left_position
        else:
            msg_error = "Incoherent lengths of times and left_position"
            assert times.size == left_position.size, msg_error
            interpolated_function = interp1d(times, left_position, assume_sorted=True)
            left_position = interpolated_function(time)

        position_dot = velocity
        velocity_dot = (
            -(left_stiff + right_stiff) * position
            + left_stiff * left_position
            + right_stiff * right_position
        ) / mass
        return position_dot, velocity_dot
