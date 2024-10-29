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
"""A discipline for solving ordinary differential equations (ODEs)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from numpy import array
from numpy import concatenate

from gemseo.core.mdo_functions.discipline_adapter_generator import (
    DisciplineAdapterGenerator,
)
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.core.discipline import Discipline
    from gemseo.core.mdo_functions.discipline_adapter import DisciplineAdapter
    from gemseo.disciplines.ode.ode_discipline import ODEDiscipline
    from gemseo.typing import RealArray


class BaseFunctor:
    """A function wrapping a RHS discipline.

    This function has time and state as arguments,
    and an attribute 'terminal'.

    Its subclasses are used to evaluate the outputs of the RHS discipline
    or its Jacobian.
    """

    _adapter: DisciplineAdapter
    """The :class:`MDOFunction` wrapping the discipline defining the dynamic."""

    __parameter_names: tuple[str, ...]
    """The names of the parameters."""

    __ode_discipline: ODEDiscipline
    """The ODE discipline providing the local data."""

    terminal: bool
    """An attribute used by ODE libraries."""

    def __init__(
        self,
        ode_discipline: ODEDiscipline,
        discipline: Discipline,
        state_names: Iterable[str] | Mapping[str, str],
        time_name: str,
    ) -> None:
        """
        Args:
            ode_discipline: The ODE discipline providing the local data.
            discipline: The wrapped discipline defining the dynamic.
            state_names: Either the names of the state variables,
                passed as ``(state_name, ...)``,
                or the names of the state variables
                bound to the associated discipline outputs,
                passed as ``{state_name: output_name, ...}``.
            time_name: The name of the time variables.
        """  # noqa: D205, D212, D415
        self.terminal = False
        self.__ode_discipline = ode_discipline
        excluded_names = [time_name, *state_names]
        self.__parameter_names = tuple(
            name for name in discipline.default_input_data if name not in excluded_names
        )
        generator = DisciplineAdapterGenerator(discipline=discipline)
        if isinstance(state_names, Mapping):
            output_names = state_names.values()
            state_names = tuple(state_names.keys())
        else:
            output_names = discipline.io.output_grammar.names

        self._adapter = generator.get_function(
            input_names=[
                time_name,
                *state_names,
                *self.__parameter_names,
            ],
            output_names=output_names,
        )

    def _compute_input_vector(self, time: RealArray, state: RealArray) -> RealArray:
        """Compute an input vector concatenating time, state and parameters.

        Args:
            time: The current time.
            state: The current state.

        Returns:
            The concatenation of time, state and parameters.
        """
        return concatenate((
            array([time]),
            state,
            concatenate_dict_of_arrays_to_array(
                self.__ode_discipline.local_data,
                names=self.__parameter_names,
            ),
        ))
