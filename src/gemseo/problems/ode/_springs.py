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
r"""The discipline for describing the motion of masses connected by springs.

Consider a system of :math:`n` point masses with masses :math:`m_1$, :math:`m_2$,...
:math:`m_n` connected in series by springs. The displacement of the point masses
relative to the position at rest are denoted by :math:`x_1`, :math:`x_2`,...
:math:`x_n`. Each spring has stiffness :math:`k_1`, :math:`k_2`,... :math:`k_{n+1}`.

Motion is assumed to only take place in one dimension, along the axes of the springs.

The extremities of the first and last spring are fixed. This means that by convention,
:math:`x_0 = x_{n+1} = 0`.


For :math:`n=2`, the system is as follows:

.. asciiart::

    |                                                                                 |
    |        k1           ________          k2           ________          k3         |
    |  /\      /\        |        |   /\      /\        |        |   /\      /\       |
    |_/  \    /  \     __|   m1   |__/  \    /  \     __|   m2   |__/  \    /  \     _|
    |     \  /    \  /   |        |      \  /    \  /   |        |      \  /    \  /  |
    |      \/      \/    |________|       \/      \/    |________|       \/      \/   |
    |                         |                              |                        |
                           ---|--->                       ---|--->
                              |   x1                         |   x2


The force of a spring with stiffness :math:`k` is

.. math::

    \vec{F} = -kx

where :math:`x` is the displacement of the extremity of the spring.

Newton's second law applied to any point mass :math:`m_i` can be written as

.. math::

    m_i \ddot{x_i} = \sum \vec{F} = k_i (x_{i-1} - x_i) + k_{i+1} (x_{i+1} - x_i)
                 = k_i x_{i-1} + k_{i+1} x_{i+1} - (k_i + k_{i+1}) x_i


This can be re-written as a system of first-order ordinary differential equations:

.. math::

    \left\{ \begin{cases}
        \dot{x_i} &= y_i \\
        \dot{y_i} &=
            - \frac{k_i + k_{i+1}}{m_i}x_i
            + \frac{k_i}{m_i}x_{i-1} + \frac{k_{i+1}}{m_i}x_{i+1}
    \end{cases} \right.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Final
from typing import NamedTuple

from numpy import array
from numpy import cos
from numpy import linspace
from numpy import ndarray
from numpy import repeat
from numpy import sin
from numpy import sqrt
from scipy.interpolate import interp1d

from gemseo.core.discipline import MDODiscipline
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.disciplines.ode.ode_discipline import ODEDiscipline
from gemseo.disciplines.remapping import RemappingDiscipline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import RealArray

POSITION: Final[str] = "position"
"""The name of the state variable representing the position."""

VELOCITY: Final[str] = "velocity"
"""The name of the state variable representing the velocity."""

STATE_NAMES: Final[tuple[str, str]] = (POSITION, VELOCITY)
"""The names of the state variables."""

STATE_DOT_NAMES: Final[tuple[str, str]] = (f"{POSITION}_dot", f"{VELOCITY}_dot")
"""The names of the derivatives of the state variables."""

_times = linspace(0.0, 10, 30)
_initial_time = array([0.0])
_initial_position = array([0.0])
_initial_velocity = array([0.0])


def generic_mass_rhs_function(
    time: ndarray = _initial_time,
    position: ndarray = _initial_position,
    velocity: ndarray = _initial_velocity,
    mass: float = 1.0,
    left_stiff: float = 1.0,
    right_stiff: float = 1.0,
    left_position: ndarray | float = 0.0,
    right_position: ndarray | float = 0.0,
    times: ndarray | None = None,
) -> tuple[ndarray, ndarray]:
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
        right_position: The position of the mass on the right-hand side of the current
            mass at `time`.
        times: The times to compute `left_position` and `right_position` at.
            This parameter is only used if `left_position` and `right_position` are
            vectors. This is not the same times as in ODEDiscipline!

    Returns:
        Vector containing the derivative of `state` (i.e. of the position and velocity)
        at `time`.
    """
    if isinstance(right_position, ndarray) and times is not None:
        assert times.size == right_position.size
        interpolated_function = interp1d(times, right_position, assume_sorted=True)
        right_pos = interpolated_function(time)
    else:
        right_pos = right_position

    if isinstance(left_position, ndarray) and times is not None:
        assert times.size == left_position.size
        interpolated_function = interp1d(times, left_position, assume_sorted=True)
        left_pos = interpolated_function(time)
    else:
        left_pos = left_position

    position_dot = velocity
    velocity_dot = (
        -(left_stiff + right_stiff) * position
        + left_stiff * left_pos
        + right_stiff * right_pos
    ) / mass
    return position_dot, velocity_dot


def create_mass_mdo_discipline(
    mass: float,
    left_stiff: float,
    right_stiff: float,
    left_pos: ndarray | float = 0.0,
    right_pos: ndarray | float = 0.0,
    is_left_pos_fixed=False,
    is_right_pos_fixed=False,
    times: ndarray | None = None,
) -> MDODiscipline:
    """Create an MDODiscipline representing the dynamic of a single mass in the chain.

    Args:
        mass: The value of the mass.
        left_stiff: The stiffness of the spring on the left-hand side.
        right_stiff: The stiffness of the spring on the right-hand side.
        left_pos: The position of the other extremity of the spring to the left.
        right_pos: The position of the other extremity of the spring to the right.
        is_left_pos_fixed: True if the other end of the spring to the left is fixed.
        is_right_pos_fixed: True if the other end of the spring to the right is fixed.
        times: the time vector for the evaluation of the left and right positions.

    Returns:
        The MDODiscipline describing a single point mass.
    """

    def _mass_rhs_generic(
        time: ndarray = _initial_time,
        position: ndarray = _initial_position,
        velocity: ndarray = _initial_velocity,
        left_position=left_pos,
        right_position=right_pos,
        time_vec=times,
    ) -> tuple[ndarray, ndarray]:
        position_dot, velocity_dot = generic_mass_rhs_function(
            time=time,
            position=position,
            velocity=velocity,
            mass=mass,
            left_stiff=left_stiff,
            right_stiff=right_stiff,
            left_position=left_position,
            right_position=right_position,
            times=time_vec,
        )
        return position_dot, velocity_dot

    if not is_left_pos_fixed and times is not None:
        if isinstance(left_pos, float):
            left_pos = array([left_pos] * len(times))
        elif len(left_pos) == 1:
            left_pos = repeat(left_pos, len(times))
        elif len(times) != len(left_pos):
            msg = "Incoherent lengths of times and left_pos"
            raise ValueError(msg)

    if not is_right_pos_fixed and times is not None:
        if isinstance(right_pos, float):
            right_pos = array([right_pos] * len(times))
        elif len(right_pos) == 1:
            right_pos = repeat(right_pos, len(times))
        elif len(times) != len(right_pos):
            msg = "Incoherent lengths of times and right_pos"
            raise ValueError(msg)

    if is_left_pos_fixed and is_right_pos_fixed:

        def _mass_rhs(
            time: ndarray = _initial_time,
            position: ndarray = _initial_position,
            velocity: ndarray = _initial_velocity,
        ) -> tuple[ndarray, ndarray]:
            position_dot, velocity_dot = _mass_rhs_generic(
                time=time,
                position=position,
                velocity=velocity,
                left_position=left_pos,
                right_position=right_pos,
                time_vec=times,
            )
            return position_dot, velocity_dot

    elif is_left_pos_fixed:

        def _mass_rhs(
            time: ndarray = _initial_time,
            position: ndarray = _initial_position,
            velocity: ndarray = _initial_velocity,
            right_position=right_pos,
        ) -> tuple[ndarray, ndarray]:
            position_dot, velocity_dot = _mass_rhs_generic(
                time=time,
                position=position,
                velocity=velocity,
                left_position=left_pos,
                right_position=right_position,
                time_vec=times,
            )
            return position_dot, velocity_dot

    elif is_right_pos_fixed:

        def _mass_rhs(
            time: ndarray = _initial_time,
            position: ndarray = _initial_position,
            velocity: ndarray = _initial_velocity,
            left_position=left_pos,
        ) -> tuple[ndarray, ndarray]:
            position_dot, velocity_dot = _mass_rhs_generic(
                time=time,
                position=position,
                velocity=velocity,
                left_position=left_position,
                right_position=right_pos,
                time_vec=times,
            )
            return position_dot, velocity_dot

    else:

        def _mass_rhs(
            time: ndarray = _initial_time,
            position: ndarray = _initial_position,
            velocity: ndarray = _initial_velocity,
            left_position=left_pos,
            right_position=right_pos,
        ) -> tuple[ndarray, ndarray]:
            position_dot, velocity_dot = _mass_rhs_generic(
                time=time,
                position=position,
                velocity=velocity,
                left_position=left_position,
                right_position=right_position,
                time_vec=times,
            )
            return position_dot, velocity_dot

    return AutoPyDiscipline(_mass_rhs, grammar_type=MDODiscipline.GrammarType.SIMPLE)


def create_mass_ode_discipline(
    mass: float,
    left_stiff: float,
    right_stiff: float,
    times: RealArray,
    left_position: float = 0.0,
    right_position: float = 0.0,
    state_names: Iterable[str] = STATE_NAMES,
    state_dot_var_names: Iterable[str] = STATE_DOT_NAMES,
    left_position_name: str = "left_position",
    right_position_name: str = "right_position",
    is_left_pos_fixed: bool = False,
    is_right_pos_fixed: bool = False,
    **ode_solver_options: Any,
) -> ODEDiscipline:
    """Create a discipline describing the motion of a single mass in the chain.

    Args:
        mass: The value of the mass.
        left_stiff: The stiffness of the spring on the left-hand side.
        right_stiff: The stiffness of the spring on the right-hand side.
        left_position: The position of the other extremity of the spring to the left.
        right_position: The position of the other extremity of the spring to the right.
        times: The times at which the solution must be evaluated.
        state_names: The names of the state variables.
        state_dot_var_names: The names of the derivatives of the state variables
            relative to time.
        left_position_name: The names of the variable describing the motion of the mass
            to the left.
        right_position_name: The names of the variable describing the motion of the mass
            to the left.
        is_left_pos_fixed: True if the other end of the spring to the left is fixed.
        is_right_pos_fixed: True if the other end of the spring to the right is fixed.
        **ode_solver_options: The options of the ODE solver.

    Returns:
        The MDODiscipline describing a single point mass.
    """
    mass_discipline = create_mass_mdo_discipline(
        mass,
        left_stiff,
        right_stiff,
        left_position,
        right_position,
        is_left_pos_fixed,
        is_right_pos_fixed,
        times,
    )

    input_mapping = dict(zip(state_names, STATE_NAMES))
    input_mapping["time"] = "time"
    if not is_left_pos_fixed:
        input_mapping[left_position_name] = "left_position"
    if not is_right_pos_fixed:
        input_mapping[right_position_name] = "right_position"
    output_mapping = dict(zip(state_dot_var_names, STATE_DOT_NAMES))
    renamed_mass_discipline = RemappingDiscipline(
        mass_discipline,
        input_mapping,
        output_mapping,
    )
    renamed_mass_discipline.input_grammar.restrict_to(input_mapping.keys())

    return ODEDiscipline(
        discipline=renamed_mass_discipline,
        state_names=state_names,
        times=times,
        return_trajectories=True,
        **ode_solver_options,
    )


class Mass(NamedTuple):
    """A mass."""

    mass: float
    """The values of the mass."""

    position: float
    """The position of the mass."""

    left_stiffness: float
    """The stiffness of the left spring."""


def create_chained_masses(
    last_stiffness: float,
    *masses: Mass,
    times: RealArray = _times,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> list[ODEDiscipline]:
    """Create coupled ODE disciplines describing masses connected by springs.

    Args:
        last_stiffness: The stiffness of the right most spring.
        *masses: The masses defined by their values and positions.
        times: The times for which the problem should be solved.
        rtol: relative tolerance for the solution of the ODE.
        atol: absolute tolerance for the solution of the ODE.

    Returns:
        The ODE disciplines.
    """
    _masses = [mass.mass for mass in masses]
    positions = [mass.position for mass in masses]
    stiffnesses = [mass.left_stiffness for mass in masses]
    stiffnesses.append(last_stiffness)

    disciplines = []
    for i, mass in enumerate(_masses):
        kwargs = {}
        if i > 0:
            kwargs["left_position_name"] = f"{POSITION}{i - 1}_trajectory"
            kwargs["left_position"] = array([positions[i - 1]])
        else:
            kwargs["is_left_pos_fixed"] = True

        if i < (len(_masses) - 1):
            kwargs["right_position_name"] = f"{POSITION}{i + 1}_trajectory"
            kwargs["right_position"] = array([positions[i + 1]])
        else:
            kwargs["is_right_pos_fixed"] = True

        disciplines.append(
            create_mass_ode_discipline(
                mass=mass,
                left_stiff=stiffnesses[i],
                right_stiff=stiffnesses[i + 1],
                times=times,
                state_names=[f"{n}{i}" for n in STATE_NAMES],
                state_dot_var_names=[f"{n}{i}" for n in STATE_DOT_NAMES],
                rtol=rtol,
                atol=atol,
                **kwargs,
            )
        )

    return disciplines


def compute_analytic_mass_position(
    initial_position: float,
    initial_velocity: float,
    left_stiff: float,
    right_stiff: float,
    mass: float,
    times: float | RealArray,
) -> float | RealArray:
    r"""Compute the analytic position of the mass :math:`m` connected with two springs.

    The equation describing the motion of the mass is

    .. math::

    \\left\\{ \begin{cases}
        \\dot{x} &= y \\
        \\dot{y} &= \frac{k_1 + k_2}{m} x
    \\end{cases} \right.

    where :math:`k_1` and :math:`k_2` are the stiffnesses of the springs.

    If :math:`x(t=0) = x_0` and :math:`y(t=0) = y_0`, then the general expression for
    the position :math:`x(t)` at time :math:`t` is

    .. math::

        x(t) = \frac{1}{2}\big( x + \frac{y_0}{\\omega} \big) \\exp^{i\\omega t)
            + \frac{1}{2}\big( x - \frac{y_0}{\\omega} \big) \\exp^{-i\\omega t)

    with :math:`\\omega = \frac{-k_1+k_2}{m}`.

    Args:
        initial_position: The initial position.
        initial_velocity: The initial velocity.
        left_stiff: The stiffness of the spring to the left.
        right_stiff: The stiffness of the spring to the right.
        mass: The value of the mass.
        times: The time(s) at which the position of the mass should be evaluated.

    Returns:
        The position(s) of the mass.
    """
    omega = sqrt((right_stiff + left_stiff) / mass)
    omega_time = omega * times
    return (initial_position * cos(omega_time)) + (
        initial_velocity / omega * sin(omega_time)
    )
