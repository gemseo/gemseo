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
from typing import Any
from typing import Final

from numpy import atleast_1d

from gemseo.algos.ode.factory import ODESolverLibraryFactory
from gemseo.algos.ode.ode_problem import ODEProblem
from gemseo.core.discipline.discipline import Discipline
from gemseo.disciplines.ode.ode_function import ODEFunction
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.string_tools import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.algos.ode.base_ode_solver_library import BaseODESolverLibrary
    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping


class ODEDiscipline(Discipline):
    """A discipline for solving Ordinary Differential Equations (ODE)."""

    _rhs_discipline: Discipline
    """The discipline defining the RHS of the ODE."""

    termination_event_disciplines: Iterable[Discipline]
    """The disciplines defining the stopping conditions."""

    _ode_problem: ODEProblem
    """The ODE problem to be solved."""

    _output_trajectory: bool
    """Whether to output both the state trajectories."""

    __design_variables_names: Iterable[str]
    """The names of the design variables of the ODE."""

    __final_state_names: tuple[str, ...]
    """The names of the variables at final time."""

    __final_time_name: str
    """The name of the variable for the final time."""

    __initial_state_names: tuple[str, ...]
    """The names of the variables for the initial conditions."""

    __initial_time_name: str
    """The name of the variable for the initial time."""

    __ode_solver: BaseODESolverLibrary
    """The ODE solver."""

    __ode_solver_options: Mapping[str, Any]
    """The options of the ODE solver."""

    __state_names: Iterable[str] | Mapping[str, str]
    """The names of the state variables, eventually bound to the
     names of their time derivatives."""

    __time_name: str
    """The name of the time variable."""

    __trajectory_state_names: Iterable[str]
    """The names of the trajectories of the state variables."""

    __TERMINATION_TIME: Final[str] = "termination_time"
    """The string constant for termination time."""

    __TIMES: Final[str] = "times"
    """The string constant for times."""

    def __init__(
        self,
        rhs_discipline: Discipline,
        times: RealArray,
        time_name: str = "time",
        state_names: Iterable[str] | Mapping[str, str] = (),
        initial_state_names: Mapping[str, str] = READ_ONLY_EMPTY_DICT,
        initial_time_name: str = "",
        final_state_names: Mapping[str, str] = READ_ONLY_EMPTY_DICT,
        final_time_name: str = "",
        state_trajectory_names: Mapping[str, str] = READ_ONLY_EMPTY_DICT,
        return_trajectories: bool = False,
        name: str = "",
        termination_event_disciplines: Iterable[Discipline] = (),
        solve_at_algorithm_times: bool = False,
        ode_solver_name: str = "RK45",
        **ode_solver_settings: Any,
    ):
        """
        Args:
            rhs_discipline: The discipline defining the right-hand side function
                of the ODE.
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
                bound to the associated ``rhs_discipline`` outputs,
                passed as ``{state_name: output_name, ...}``.
                If empty, use all the ``rhs_discipline`` inputs.
            initial_state_names: The names of the state variables
                bound to the names of the variables denoting the initial conditions.
                If empty,
                use ``"state_initial"`` for a state variable named ``"state"``.
            initial_time_name: The name of the variable for the initial time.
                If empty, use ``f"initial_{time_name}"``.
            final_state_names: The names of the state variables
                bound to their names at final time.
                If empty,
                use ``"state_final"`` for a state variable named ``"state"``.
            final_time_name: The name of the variable for the final time.
                If empty, use ``f"final_{time_name}"``.
            state_trajectory_names: The names of the state variables
                bound to the names of their trajectories.
                If empty,
                use ``"state"`` for a state variable named ``"state"``.
            return_trajectories: Whether to output
                both the trajectories of the state variables
                and their values at final time.
                Otherwise, output only their values at final time.
            termination_event_disciplines: The disciplines encoding termination events.
                Each discipline must have the same inputs as ``rhs_discipline``
                and only one output defined as an arrays of size 1
                indicating the value of an event function.
                The resolution of the ODE problem stops
                when one of the event functions crosses the threshold 0.
                If empty, the integration covers the entire time interval.
            solve_at_algorithm_times: Whether to solve the ODE chosen by the algorithm.
            ode_solver_name: The name of the ODE solver.
            **ode_solver_settings: The settings of the ODE solver.

        Raises:
            ValueError: If an expected state variable does not appear in
                ``rhs_discipline``.
        """  # noqa: D205, D212, D415
        self._rhs_discipline = rhs_discipline
        self._output_trajectory = (
            return_trajectories or state_trajectory_names or len(times) > 2
        )
        self.termination_event_disciplines = termination_event_disciplines

        # Define the names of the time variables and initial time variable.
        self.__time_name = time_name
        self.__initial_time_name = initial_time_name or f"initial_{self.__time_name}"
        self.__final_time_name = final_time_name or f"final_{self.__time_name}"

        # Define the names of the state variables and their time derivatives
        if state_names:
            wrong_state_names = set(state_names) - set(rhs_discipline.io.input_grammar)
            if wrong_state_names:
                msg = (
                    f"{pretty_repr(wrong_state_names, use_and=True)} are not input"
                    f" variables of the RHS discipline."
                )
                raise ValueError(msg)

            if isinstance(state_names, Mapping):
                state_names = {
                    input_name: state_names[input_name]
                    for input_name in rhs_discipline.io.input_grammar.names
                    if input_name in state_names
                }
                self.__state_names = state_names.keys()
                state_dot_names = state_names.values()
                wrong_output_names = set(state_dot_names) - set(
                    rhs_discipline.io.output_grammar
                )
                if wrong_output_names:
                    msg = (
                        f"{pretty_repr(wrong_output_names, use_and=True)} are not "
                        f"output variables of the RHS discipline."
                    )
                    raise ValueError(msg)
            else:
                self.__state_names = tuple(
                    input_name
                    for input_name in rhs_discipline.io.input_grammar.names
                    if input_name in state_names
                )
                state_dot_names = rhs_discipline.io.output_grammar.names
        else:
            self.__state_names = tuple(
                name
                for name in rhs_discipline.io.input_grammar.names
                if name != time_name
            )
            state_dot_names = rhs_discipline.io.output_grammar.names

        excluded_names = [self.__time_name, *self.__state_names]
        self.__design_variables_names = tuple(
            name
            for name in rhs_discipline.default_input_data
            if name not in excluded_names
        )

        self.__initial_state_names = tuple(
            initial_state_names.get(state_name, f"initial_{state_name}")
            for state_name in self.__state_names
        )

        self.__ode_solver = ODESolverLibraryFactory().create(ode_solver_name)
        self.__ode_solver_options = ode_solver_settings

        super().__init__(name=name)

        mapping_initial_state = {
            initial_name: self.local_data.get(
                initial_name, rhs_discipline.default_input_data[state_name]
            )
            for (initial_name, state_name) in zip(
                self.__initial_state_names, self.__state_names
            )
        }

        mapping_parameters = {
            parameter: self.local_data.get(
                parameter, rhs_discipline.default_input_data[parameter]
            )
            for parameter in self.__design_variables_names
        }

        mapping_inputs = {
            self.__initial_time_name: self.local_data.get(
                self.__initial_time_name, times[0]
            ),
            self.__final_time_name: self.local_data.get(
                self.__final_time_name, times[-1]
            ),
            **mapping_initial_state,
            **mapping_parameters,
        }

        if self._output_trajectory:
            mapping_inputs[self.__TIMES] = times

        self._rhs_discipline.default_input_data.update(mapping_parameters)
        for termination_discipline in self.termination_event_disciplines:
            termination_discipline.default_input_data.update(mapping_parameters)

        # Define ODEProblem
        initial_state = concatenate_dict_of_arrays_to_array(
            {k: atleast_1d(v) for k, v in mapping_initial_state.items()},
            names=self.__initial_state_names,
        )
        event_functions = tuple(
            ODEFunction(
                termination_discipline, time_name, self.__state_names, terminal=True
            )
            for termination_discipline in termination_event_disciplines
        )
        ode_func = ODEFunction(
            rhs_discipline, time_name, self.__state_names, output_names=state_dot_names
        )
        self._ode_problem = ODEProblem(
            func=ode_func,
            initial_state=initial_state,
            times=times,
            jac_function_wrt_state=ode_func.evaluate_jacobian,
            event_functions=event_functions,
            solve_at_algorithm_times=solve_at_algorithm_times,
        )

        # Define the names for the trajectories and final states.
        self.__final_state_names = tuple(
            final_state_names.get(state_name, f"final_{state_name}")
            for state_name in self.__state_names
        )
        if self._output_trajectory:
            self.__trajectory_state_names = tuple(
                state_trajectory_names.get(state_name, state_name)
                for state_name in self.__state_names
            )
        else:
            self.__trajectory_state_names = READ_ONLY_EMPTY_DICT

        # Initialize inputs and outputs
        self.io.input_grammar.update_from_data(mapping_inputs)
        self.default_input_data = mapping_inputs

        output_names = [
            *self.__final_state_names,
            self.__TERMINATION_TIME,
        ]
        if return_trajectories:
            output_names.append(self.__TIMES)
            output_names.extend(self.__trajectory_state_names)
        self.io.output_grammar.update_from_names(output_names)

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        mapping_parameters = {
            k: self.local_data[k] for k in self.__design_variables_names
        }
        self._ode_problem.update_times(
            initial_time=self.local_data.get(self.__initial_time_name, None),
            final_time=self.local_data.get(self.__final_time_name, None),
            times=self.local_data.get(self.__TIMES, None),
        )

        self._rhs_discipline.default_input_data.update(mapping_parameters)
        for termination_discipline in self.termination_event_disciplines:
            termination_discipline.default_input_data.update(mapping_parameters)

        self._ode_problem.initial_state = concatenate_dict_of_arrays_to_array(
            {k: atleast_1d(v) for k, v in input_data.items()},
            names=self.__initial_state_names,
        )
        self.__ode_solver.execute(self._ode_problem, **self.__ode_solver_options)
        result = self._ode_problem.result
        if not result.algorithm_has_converged:
            msg = (
                f"ODE solver {result.algorithm_name} failed to converge. "
                f"Message = {result.algorithm_termination_message}"
            )
            raise RuntimeError(msg)

        output_data = dict(zip(self.__final_state_names, result.final_state))

        if self._output_trajectory:
            output_data.update(
                dict(zip(self.__trajectory_state_names, result.state_trajectories))
            )

        output_data[self.__TERMINATION_TIME] = result.termination_time
        output_data[self.__TIMES] = result.times
        return output_data
