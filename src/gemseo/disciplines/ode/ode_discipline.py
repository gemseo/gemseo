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

from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from gemseo.algos.ode.factory import ODESolverLibraryFactory
from gemseo.algos.ode.ode_problem import ODEProblem
from gemseo.core.discipline.discipline import Discipline
from gemseo.disciplines.ode.functor import Functor
from gemseo.disciplines.ode.jactor import Jactor
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from numpy.typing import ArrayLike

    from gemseo.algos.ode.base_ode_solver_library import BaseODESolverLibrary
    from gemseo.typing import RealArray


class ODEDiscipline(Discipline):
    """A discipline for solving ordinary differential equations (ODE)."""

    output_trajectory: bool
    """Whether to output both the state trajectories and the states at final time.

    Otherwise, output only their values at final time.
    """

    _ode_problem: ODEProblem
    """The ODE problem to be solved."""

    __final_state_names: tuple[str, ...]
    """The names of the variables at final time."""

    __ode_solver: BaseODESolverLibrary
    """The ODE solver."""

    __ode_solver_options: Mapping[str, Any]
    """The options of the ODE solver."""

    __state_names: Iterable[str] | Mapping[str, str]
    """The names of the state variables, eventually bound to the
     names of their time derivatives."""

    __time_name: str
    """The name of the time variable."""

    __trajectory_state_names: tuple[str, ...]
    """The names of the trajectories of the state variables."""

    __TERMINATION_TIME: Final[str] = "termination_time"
    """The string constant for termination time."""

    __TIMES: Final[str] = "times"
    """The string constant for times."""

    def __init__(
        self,
        discipline: Discipline,
        times: ArrayLike,
        time_name: str = "time",
        state_names: Iterable[str] | Mapping[str, str] = (),
        final_state_names: Mapping[str, str] = READ_ONLY_EMPTY_DICT,
        state_trajectory_names: Mapping[str, str] = READ_ONLY_EMPTY_DICT,
        return_trajectories: bool = False,
        name: str = "",
        termination_event_disciplines: Iterable[Discipline] = (),
        solve_at_algorithm_times: bool | None = None,
        ode_solver_name: str = "RK45",
        **ode_solver_options: Any,
    ):
        """
        Args:
            discipline: The discipline defining the right-hand side function of the ODE.
            times: Either the initial and final times
                or the times of interest where the state must be stored,
                including the initial and final times.
                When only initial and final times are provided,
                the times of interest are the instants chosen by the ODE solver
                to compute the state trajectories.
            time_name: The name of the time variable.
            state_names: Either the names of the state variables,
                passed as ``(state_name, ...)``,
                or the names of the state variables
                bound to the associated discipline outputs,
                passed as ``{state_name: output_name, ...}``.
                If empty, use all the discipline inputs.
            final_state_names: The names of the state variables
                bound to their names at final time.
                If empty,
                use ``state_name_final`` for a state variable named ``state_name``.
            state_trajectory_names: The names of the state variables
                bound to the names of their trajectories.
                If empty,
                use ``state_name_trajectory`` for a state variable named ``state_name``.
            return_trajectories: Whether to output
                both the trajectories of the state variables
                and their values at final time.
                Otherwise, output only their values at final time.
            termination_event_disciplines: The disciplines encoding termination events.
                Each discipline must have the same inputs as ``discipline``
                and only one output defined as an arrays of size 1
                indicating the value of an event function.
                The resolution of the ODE problem stops
                when one of the event functions crosses the threshold 0.
                If empty, the integration covers the entire time interval.
            solve_at_algorithm_times: Whether to solve the ODE chosen by the algorithm.
                Otherwise, use times defined in the vector `times`.
                If ``None``,
                it is initialized as ``False``
                if no terminal event is considered,
                and ``True`` otherwise.
            ode_solver_name: The name of the ODE solver.
            **ode_solver_options: The options of the ODE solver.
        """  # noqa: D205, D212, D415
        # Define the names of the state variables
        if state_names:
            self.__state_names = state_names
        else:
            state_names = self.__state_names = [
                name for name in discipline.io.input_grammar.names if name != time_name
            ]

        missing_names = set(state_names) - set(discipline.default_input_data)
        if missing_names:
            msg = f"Missing default inputs in discipline for {missing_names}."
            raise ValueError(msg)

        self.output_trajectory = return_trajectories
        self.__time_name = time_name

        # Store information concerning ODE solver
        self.__ode_solver = ODESolverLibraryFactory().create(ode_solver_name)
        self.__ode_solver_options = ode_solver_options

        # Create ODE problem
        initial_state = self.__get_state_vector(discipline.default_input_data)
        event_functions = tuple(
            Functor(self, discipline, state_names, time_name)
            for discipline in termination_event_disciplines
        )
        self._ode_problem = ODEProblem(
            func=Functor(self, discipline, state_names, time_name),
            initial_state=initial_state,
            times=times,
            jac_wrt_time_state=Jactor(self, discipline, state_names, time_name),
            event_functions=event_functions,
            solve_at_algorithm_times=solve_at_algorithm_times,
        )

        # Define the names for the trajectories and final states.
        self.__final_state_names = tuple(
            final_state_names.get(state_name, f"{state_name}_final")
            for state_name in state_names
        )
        if self.output_trajectory:
            self.__trajectory_state_names = tuple(
                state_trajectory_names.get(state_name, f"{state_name}_trajectory")
                for state_name in state_names
            )
        else:
            self.__trajectory_state_names = ()

        super().__init__(name=name)
        self.io.input_grammar = discipline.io.input_grammar
        output_names = [
            *self.__final_state_names,
            self.__TIMES,
            self.__TERMINATION_TIME,
        ]
        output_names.extend(self.__trajectory_state_names)
        self.io.output_grammar.update_from_names(output_names)
        self.default_input_data = copy(discipline.default_input_data)

    def __get_state_vector(self, input_data: Mapping[str, RealArray]) -> RealArray:
        """Return the state vector from a dictionary.

        Args:
            input_data: The state values.

        Returns:
            The state vector.
        """
        return concatenate_dict_of_arrays_to_array(input_data, names=self.__state_names)

    def _run(self) -> None:
        self._ode_problem.initial_state = self.__get_state_vector(self.io.data)
        self.__ode_solver.execute(self._ode_problem, **self.__ode_solver_options)
        result = self._ode_problem.result
        if not result.algorithm_has_converged:
            msg = (
                f"ODE solver {result.algorithm_name} failed to converge. "
                f"Message = {result.algorithm_termination_message}"
            )
            raise RuntimeError(msg)

        self.io.data.update(
            dict(zip(self.__final_state_names, result.terminal_event_state))
        )
        if self.output_trajectory:
            self.io.data.update(
                dict(zip(self.__trajectory_state_names, result.state_trajectories))
            )

        self.io.data.update({self.__TERMINATION_TIME: result.terminal_event_time})
        self.io.data[self.__TIMES] = result.times
