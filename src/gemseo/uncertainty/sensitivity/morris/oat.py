# -*- coding: utf-8 -*-
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

   df_i = f(X_1+dX_1,\ldots,X_i+dX_i,\ldots,X_d)-f(X_1,\ldots,X_i,\ldots,X_d)

associated with a small variation :math:`dX_i` of :math:`X_i` with

.. math::

   df_1 = f(X_1+dX_1,\ldots,X_i,\ldots,X_d)-f(X_1,\ldots,X_i,\ldots,X_d)

The elementary effects :math:`df_1,\ldots,df_d` are computed sequentially
from an initial point

.. math::

   X=(X_1,\ldots,X_d)

From these elementary effects, we can compare their absolute values
:math:`|df_1|,\ldots,|df_d|` and sort :math:`X_1,\ldots,X_d` accordingly.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from builtins import super
from copy import deepcopy
from typing import Dict, Mapping, Tuple

from numpy import ndarray

from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import MDODiscipline

LOGGER = logging.getLogger(__name__)


class OATSensitivity(MDODiscipline):
    """A :class:`.MDODiscipline` computing finite differences of another one.

    Args:
        discipline (MDODiscipline): A discipline.
        parameter_space (DesignSpace): A parameter space.
        step (float): A relative finite difference step between 0 and 0.5.
    """

    _PREFIX = "fd"

    def __init__(
        self,
        discipline,  # type: MDODiscipline
        parameter_space,  # type: DesignSpace
        step,  # type: float
    ):  # type: (...) -> None # noqa: D107
        super(OATSensitivity, self).__init__()
        inputs = parameter_space.variables_names
        self.input_grammar.initialize_from_data_names(inputs)
        outputs = [
            self.get_fd_name(input_, output)
            for output in discipline.get_output_data_names()
            for input_ in inputs
        ]
        self.output_grammar.initialize_from_data_names(outputs)
        self.discipline = discipline
        self.step = step
        self.parameter_space = parameter_space

    def _run(self):  # type: (...) -> None
        """Run method."""
        inputs = self.get_input_data()
        self.discipline.execute(inputs)
        out_prev = self.discipline.local_data
        for input_name in list(self.get_input_data_names()):
            inputs = self.__update_inputs(inputs, input_name, self.step)
            self.discipline.execute(inputs)
            out_curr = self.discipline.local_data
            out_diff = {
                name: out_curr[name] - out_prev[name]
                for name in self.discipline.get_output_data_names()
            }
            for name, value in out_diff.items():
                self.local_data[self.get_fd_name(input_name, name)] = value
            out_prev = out_curr

    @staticmethod
    def get_io_names(
        fd_name,  # type: str
    ):  # type: (...) -> Tuple[str,str]
        """Get the output and input names from finite difference name.

        Args:
            fd_name: A finite difference name.

        Returns:
            The output name.
            The input name
        """
        split_name = fd_name.split("!")
        output_name = split_name[1]
        input_name = split_name[2]
        return output_name, input_name

    def get_fd_name(
        self,
        input_name,  # type: str
        output_name,  # type: str
    ):  # type: (...) -> str
        """Return the output name associated to an input name.

        Args:
            input_name: An input name.
            output_name: An output name.

        Returns:
            The finite difference name.
        """
        return "{}!{}!{}".format(self._PREFIX, output_name, input_name)

    def __update_inputs(
        self,
        inputs,  # type: Mapping[str,ndarray]
        input_name,  # type:str
        step,  # type:float
    ):  # type: (...) -> Dict[str,ndarray]
        """Update the input data from a finite difference step and an input name.

        Args:
            inputs: The original input data.
            input_name: An input name.
            step: A relative finite difference step between 0 and 0.5.

        Returns:
            The updated input data.
        """
        if not 0 < step < 0.5:
            raise ValueError(
                "Relative finite difference step must be "
                "strictly comprised between 0 and 0.5; got {}".format(step)
            )
        inputs = deepcopy(inputs)
        l_b = self.parameter_space.get_lower_bound(input_name)
        u_b = self.parameter_space.get_upper_bound(input_name)
        abs_step = step * (u_b - l_b)
        if inputs[input_name] + abs_step > u_b:
            inputs[input_name] -= abs_step
        else:
            inputs[input_name] += abs_step
        return inputs
