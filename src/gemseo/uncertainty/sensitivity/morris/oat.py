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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""Class to apply the OAT technique used by :class:`.MorrisIndices`.

OAT technique
-------------

The purpose of the One-At-a-Time (OAT) methodology is to quantify the elementary effect

.. math::

   df_i = f(X_1+dX_1,\ldots,X_{i-1}+dX_{i-1},X_i+dX_i,\ldots,X_d)
          -
          f(X_1+dX_1,\ldots,X_{i-1}+dX_{i-1},X_i,\ldots,X_d)

associated with a small variation :math:`dX_i` of :math:`X_i` with

.. math::

   df_1 = f(X_1+dX_1,\ldots,X_d)-f(X_1,\ldots,X_d)

The elementary effects :math:`df_1,\ldots,df_d` are computed sequentially
from an initial point

.. math::

   X=(X_1,\ldots,X_d)

From these elementary effects, we can compare their absolute values
:math:`|df_1|,\ldots,|df_d|` and sort :math:`X_1,\ldots,X_d` accordingly.
"""
from __future__ import annotations

from copy import deepcopy
from typing import ClassVar
from typing import Mapping

from numpy import array
from numpy import atleast_1d
from numpy import inf
from numpy import maximum
from numpy import minimum
from numpy import ndarray

from gemseo.algos.database import DatabaseValueType
from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.utils import get_all_outputs
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array


class _OATSensitivity(MDODiscipline):
    """An :class:`.MDODiscipline` to compute finite diff.

    of a :class:`.DOEScenario`.
    """

    _PREFIX: ClassVar[str] = "fd"
    __SEPARATOR: ClassVar[str] = "!"

    def __init__(
        self,
        scenario: DOEScenario,
        parameter_space: DesignSpace,
        step: float,
    ) -> None:
        """.. # noqa: D107 D205 D212 D415
        Args:
            scenario: The scenario for the analysis.
            parameter_space: A parameter space.
            step: The variation step of an input relative to its range,
                between 0 and 0.5 (i.e. between 0 and 50% input range variation).

        Raises:
            ValueError: If the relative variation step is lower than or equal to 0
                or greater than or equal to 0.5.
        """
        if not 0 < step < 0.5:
            raise ValueError(
                "Relative variation step must be "
                f"strictly comprised between 0 and 0.5; got {step}."
            )
        super().__init__()
        input_names = parameter_space.variables_names
        self.input_grammar.update(input_names)
        self.__output_names = get_all_outputs(scenario.disciplines)
        output_names = [
            self.get_fd_name(input_name, output_name)
            for output_name in self.__output_names
            for input_name in input_names
        ]
        self.output_grammar.update(output_names)

        # The scenario is evaluated many times, this setting
        # prevents conflicts between runs.
        scenario.clear_history_before_run = True
        self.scenario = scenario

        self.step = step
        self.parameter_space = parameter_space
        self.output_range = self.output_range = {
            name: [inf, -inf] for name in self.__output_names
        }

    def __update_output_range(
        self,
        data: DatabaseValueType,
    ) -> None:
        """Update the lower and upper bounds of the outputs from data.

        Args:
            data: The names and values of the outputs.
        """
        for output_name in self.__output_names:
            output_value = atleast_1d(data[output_name])
            output_range = self.output_range[output_name]
            output_range[0] = minimum(output_value, output_range[0])
            output_range[1] = maximum(output_value, output_range[1])

    def _run(self) -> None:
        inputs = self.get_input_data()
        samples = array([concatenate_dict_of_arrays_to_array(inputs, inputs.keys())])

        # The opt_problem must be reset, this allows us to evaluate it again with
        # different samples without getting max iter reached exceptions.
        opt_problem = self.scenario.formulation.opt_problem
        opt_problem.reset()
        self.scenario.execute(
            {
                "algo": "CustomDOE",
                "algo_options": {"samples": samples},
            }
        )

        previous_data = opt_problem.database.last_item

        self.__update_output_range(previous_data)

        for input_name in self.get_input_data_names():
            inputs = self.__update_inputs(inputs, input_name, self.step)
            samples = array(
                [concatenate_dict_of_arrays_to_array(inputs, inputs.keys())]
            )

            # Reset the opt_problem before each evaluation.
            opt_problem.reset()
            self.scenario.execute(
                {"algo": "CustomDOE", "algo_options": {"samples": samples}}
            )

            new_data = opt_problem.database.last_item

            self.__update_output_range(new_data)
            for output_name in self.__output_names:
                out_diff_name = self.get_fd_name(input_name, output_name)
                out_diff_value = new_data[output_name] - previous_data[output_name]
                self.local_data[out_diff_name] = atleast_1d(out_diff_value)

            previous_data = new_data

    @classmethod
    def get_io_names(cls, fd_name: str) -> list[str, str]:
        """Get the output and input names from finite difference name.

        Args:
            fd_name: A finite difference name.

        Returns:
            The output name, then the input name.
        """
        return fd_name.split(cls.__SEPARATOR)[1:3]

    def get_fd_name(self, input_name: str, output_name: str) -> str:
        """Return the output name associated to an input name.

        Args:
            input_name: An input name.
            output_name: An output name.

        Returns:
            The finite difference name.
        """
        return self.__SEPARATOR.join([self._PREFIX, output_name, input_name])

    def __update_inputs(
        self,
        inputs: Mapping[str, ndarray],
        input_name: str,
        step: float,
    ) -> Mapping[str, ndarray]:
        """Update the input data from a finite difference step and an input name.

        Args:
            inputs: The original input data.
            input_name: An input name.
            step: The variation step of an input relative to its range,
                between 0 and 0.5 (i.e. between 0 and 50% input range variation).

        Returns:
            The updated input data.
        """
        inputs = deepcopy(inputs)
        l_b = self.parameter_space.get_lower_bound(input_name)
        u_b = self.parameter_space.get_upper_bound(input_name)
        abs_step = step * (u_b - l_b)
        if inputs[input_name] + abs_step > u_b:
            inputs[input_name] -= abs_step
        else:
            inputs[input_name] += abs_step
        return inputs
