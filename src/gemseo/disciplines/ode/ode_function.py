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
#        :author: Francois Gallard
#        :author: Isabelle Santos
#        :author: Giulio Gargantini
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""ODE Function."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from numpy import array
from numpy import concatenate

from gemseo.core.mdo_functions.discipline_adapter_generator import (
    DisciplineAdapterGenerator,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.core.discipline import Discipline
    from gemseo.core.mdo_functions.discipline_adapter import DisciplineAdapter
    from gemseo.typing import RealArray


class ODEFunction:
    """A function wrapping a discipline for ODEs.

    This function has time and state as arguments,
    and an attribute 'terminal'.
    """

    _adapter: DisciplineAdapter
    """The :class:`MDOFunction` wrapping the discipline."""

    __parameter_names: tuple[str, ...]
    """The names of the parameters."""

    terminal: bool
    """An attribute used by ODE libraries."""

    def __init__(
        self,
        discipline: Discipline,
        state_names: Iterable[str] | Mapping[str, str],
        time_name: str,
        terminal: bool = False,
    ) -> None:
        """
        Args:
            discipline: The wrapped discipline.
            state_names: Either the names of the state variables,
                passed as ``(state_name, ...)``,
                or the names of the state variables
                bound to the associated discipline outputs,
                passed as ``{state_name: output_name, ...}``.
            time_name: The name of the time variables.
            terminal: Whether this is a termination function.
        """  # noqa: D205, D212, D415
        self.terminal = False
        excluded_names = [time_name, *state_names]
        self.__parameter_names = tuple(
            name
            for name in discipline.io.input_grammar.defaults
            if name not in excluded_names
        )
        generator = DisciplineAdapterGenerator(discipline=discipline)
        if isinstance(state_names, Mapping):
            output_names = state_names.values()
            state_names = tuple(state_names.keys())
        else:
            output_names = discipline.io.output_grammar

        self._adapter = generator.get_function(
            input_names=[time_name, *state_names],
            output_names=output_names,
            differentiated_input_names_substitute=[*state_names],
        )

        self.terminal = terminal

    def __call__(self, time: RealArray, state: RealArray) -> RealArray:
        """
        Args:
            time: The time at the evaluation of the function.
            state: The state of the ODE at the evaluation of the function.

        Returns:
            The value of the function at the given time and state.

        """  # noqa: D205, D212, D415
        return self._adapter.evaluate(concatenate((array([time]), state)))

    def evaluate_jacobian(self, time: RealArray, state: RealArray) -> RealArray:
        """
        Args:
            time: The time at the evaluation of the function.
            state: The state of the ODE at the evaluation of the function.

        Returns:
            The Jacobian wrt state of the function at the given time and state.

        """  # noqa: D205, D212, D415
        return self._adapter.jac(concatenate((array([time]), state))).reshape((
            state.size,
            -1,
        ))
