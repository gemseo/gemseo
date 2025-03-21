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

r"""An object returning a Discipline for a chain of masses connected by springs.

The coupling between the dynamics of the different masses can be done at the level of
the RHS of the ODE (by the method :func:`.create_discipline_with_coupled_dynamics`),
or by coupling directly as many instances of :class:`.ODEDiscipline` as there are masses
(by the method :func:`.create_coupled_ode_disciplines`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import asarray
from numpy import linspace

from gemseo.core.chains.chain import MDOChain
from gemseo.disciplines.ode.ode_discipline import ODEDiscipline
from gemseo.problems.ode.springs.spring_ode_discipline import SpringODEDiscipline
from gemseo.problems.ode.springs.springs_dynamics_discipline import (
    SpringsDynamicsDiscipline,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.typing import RealArray


_times = linspace(0.0, 10, 30)
_initial_time = asarray([0.0])
_initial_position = asarray([0.0])
_initial_velocity = asarray([0.0])


class CoupledSpringsGenerator:
    """A class to create disciplines representing chain of masses linked by springs."""

    _initial_positions: RealArray | Sequence[float]
    """The initial positions of the masses in the chain"""

    _leftmost_position: RealArray | float
    """The position of the left extremity of the chain."""

    _masses: RealArray | Sequence[float]
    """The values of the masses in the chain."""

    _rightmost_position: RealArray | float
    """The position of the right extremity of the chain."""

    _rightmost_stiffness: float
    """The stiffness of the rightmost spring."""

    _times: RealArray
    """The times for which the problem should be solved."""

    def __init__(
        self,
        masses: RealArray | Sequence[float],
        stiffnesses: RealArray | Sequence[float],
        times: RealArray = _times,
        leftmost_position: RealArray | float = 0.0,
        rightmost_position: RealArray | float = 0.0,
    ) -> None:
        """
        Args:
            masses: The values of the masses.
            stiffnesses: The stiffnesses of the springs.
            times: The times for which the problem should be solved.
            leftmost_position: The position of the other extremity
                of the spring to the left.
            rightmost_position: The position of the other extremity
                of the spring to the right.
        """  # noqa: D205, D212, D415
        if len(stiffnesses) != (len(masses) + 1):
            msg = "Incompatible lengths of 'masses' and 'stiffnesses'."
            raise ValueError(msg)

        self._times = times
        self._masses = masses
        self._stiffnesses = stiffnesses

        self._state_names = tuple(
            (f"position_{ii}", f"velocity_{ii}") for ii in range(len(masses))
        )
        self._state_dot_names = tuple(
            (f"position_{ii}_dot", f"velocity_{ii}_dot") for ii in range(len(masses))
        )

        self._leftmost_position = leftmost_position
        self._rightmost_position = rightmost_position

    @property
    def state_names(self):
        """The names of the state variables.

        Returns:
            The names of the state variables (position and velocity).
        """
        return tuple(name for names in self._state_names for name in names)

    @property
    def state_dot_names(self):
        """The names of the time derivatives of the state variables.

        Returns:
            The names of the time derivatives of the state variables as computed by
            the RHS discipline.
        """
        return tuple(name for names in self._state_dot_names for name in names)

    @property
    def position_names(self):
        """The names of the inputs defining the position of a mass.

        Returns:
            The names of the inputs defining the position.
        """
        return tuple(names[0] for names in self._state_names)

    @property
    def initial_position_names(self):
        """The names of the inputs defining the initial position.

        Returns:
            The names of the inputs defining the initial position.
        """
        return tuple(f"initial_{names[0]}" for names in self._state_names)

    @property
    def times(self):
        """The times at which the solution os evaluated.

        Returns:
            The times of evaluation of the solution.
        """
        return self._times

    def create_coupled_ode_disciplines(
        self, **ode_solver_options
    ) -> Sequence[ODEDiscipline]:
        """Create coupled ODE disciplines describing masses connected by springs.

        Args:
            **ode_solver_options: The options of the ODE solver.

        Returns:
            The ODE disciplines.
        """
        disciplines = []
        for i, mass in enumerate(self._masses):
            kwargs = {}
            if i > 0:
                kwargs["left_position_name"] = f"{self.position_names[i - 1]}"
                kwargs["left_position"] = _initial_position
            else:
                kwargs["is_left_position_fixed"] = True
                kwargs["left_position"] = self._leftmost_position

            if i < (len(self._masses) - 1):
                kwargs["right_position_name"] = f"{self.position_names[i + 1]}"
                kwargs["right_position"] = _initial_position
            else:
                kwargs["is_right_position_fixed"] = True
                kwargs["right_position"] = self._rightmost_position

            disciplines.append(
                SpringODEDiscipline(
                    mass=mass,
                    left_stiffness=self._stiffnesses[i],
                    right_stiffness=self._stiffnesses[i + 1],
                    times=self._times,
                    state_names=self._state_names[i],
                    state_dot_names=self._state_dot_names[i],
                    **kwargs,
                    **ode_solver_options,
                )
            )

        return disciplines

    def create_discipline_with_coupled_dynamics(
        self, **ode_solver_options
    ) -> ODEDiscipline:
        """Create an ODE discipline describing masses connected by springs.

        The dynamic of the discipline (that is the right hand side of the corresponding
        ODE) is represented by coupled disciplines.

        Args:
            **ode_solver_options: The options of the ODE solver.

        Returns:
            The ODE discipline.
        """
        mass_spring_disciplines = [
            SpringsDynamicsDiscipline(
                mass=mass,
                left_stiffness=self._stiffnesses[ii],
                right_stiffness=self._stiffnesses[ii + 1],
                left_position=_initial_position if ii > 0 else self._leftmost_position,
                right_position=_initial_position
                if ii < len(self._masses) - 1
                else self._rightmost_position,
                is_left_position_fixed=(ii == 0),
                is_right_position_fixed=(ii == len(self._masses) - 1),
                times=None,
                state_names=self._state_names[ii],
                state_dot_var_names=self._state_dot_names[ii],
                left_position_name=self.position_names[ii - 1] if ii > 0 else None,
                right_position_name=self.position_names[ii + 1]
                if ii < len(self._masses) - 1
                else None,
            )
            for ii, mass in enumerate(self._masses)
        ]

        mda_dynamics = MDOChain(mass_spring_disciplines)

        return ODEDiscipline(
            rhs_discipline=mda_dynamics,
            state_names=dict(zip(self.state_names, self.state_dot_names)),
            return_trajectories=True,
            times=self._times,
            **ode_solver_options,
        )
