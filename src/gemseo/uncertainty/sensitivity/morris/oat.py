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

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import atleast_1d
from numpy import inf
from numpy import maximum
from numpy import minimum

from gemseo.core.discipline import MDODiscipline

if TYPE_CHECKING:
    from gemseo.algos.database import DatabaseValueType
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.doe_scenario import DOEScenario


class _OATSensitivity(MDODiscipline):
    """A discipline computing finite differences from a multidisciplinary system."""

    _PREFIX: ClassVar[str] = "fd"
    __SEPARATOR: ClassVar[str] = "!"

    __io_names_to_fd_names: dict[str, dict[str, str]]
    """The IO names of the original discipline bound to the output names."""

    __output_names: list[str]
    """The names of the disciplines to sample."""

    scenario: DOEScenario
    """The scenario defining the multidisciplinary system."""

    step: float
    """The variation step of an input relative to its range, between 0 and 0.5."""

    parameter_space: ParameterSpace
    """The design space on which to sample the discipline."""

    output_range: dict[str, list[float, float]]
    """The names of the outputs bound to their lower and upper bounds."""

    def __init__(
        self,
        scenario: DOEScenario,
        parameter_space: ParameterSpace,
        step: float,
    ) -> None:
        """
        Args:
            scenario: The multidisciplinary scenario for the analysis.
            parameter_space: A parameter space on which to compute finite differences.
            step: The variation step of an input relative to its range,
                between 0 and 0.5 (i.e. between 0 and 50% input range variation).

        Raises:
            ValueError: If the relative variation step is lower than or equal to 0
                or greater than or equal to 0.5.
        """  # noqa: D205, D212, D415
        if not 0 < step < 0.5:
            raise ValueError(
                "Relative variation step must be "
                f"strictly comprised between 0 and 0.5; got {step}."
            )
        super().__init__()
        input_names = parameter_space.variable_names
        self.input_grammar.update_from_names(input_names)
        problem = scenario.formulation.opt_problem
        self.__output_names = [problem.get_objective_name()] + [
            observable.name for observable in problem.observables
        ]
        output_names = [
            self.get_fd_name(input_name, output_name)
            for output_name in self.__output_names
            for input_name in input_names
        ]
        self.output_grammar.update_from_names(output_names)

        self.scenario = scenario
        self.step = step
        self.parameter_space = parameter_space
        self.output_range = self.output_range = {
            name: [inf, -inf] for name in self.__output_names
        }
        self.__io_names_to_fd_names = {
            input_name: {
                output_name: self.get_fd_name(input_name, output_name)
                for output_name in self.__output_names
            }
            for input_name in input_names
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
        problem = self.scenario.formulation.opt_problem
        problem.reset()
        input_sample = self.get_inputs_asarray()
        output_sample, _ = problem.evaluate_functions(input_sample, normalize=False)
        self.__update_output_range(output_sample)
        for input_name in self.get_input_data_names():
            next_input_sample = input_sample.copy()
            l_b = self.parameter_space.get_lower_bound(input_name)
            u_b = self.parameter_space.get_upper_bound(input_name)
            indices = self.parameter_space.get_variables_indexes([input_name])
            abs_step = self.step * (u_b - l_b)
            if next_input_sample[indices] + abs_step > u_b:
                next_input_sample[indices] -= abs_step
            else:
                next_input_sample[indices] += abs_step

            next_output_sample, _ = problem.evaluate_functions(
                next_input_sample, normalize=False
            )
            self.__update_output_range(output_sample)
            output_to_fd_names = self.__io_names_to_fd_names[input_name]
            for output_name in self.__output_names:
                self.local_data[output_to_fd_names[output_name]] = atleast_1d(
                    next_output_sample[output_name] - output_sample[output_name]
                )

            input_sample = next_input_sample
            output_sample = next_output_sample

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
