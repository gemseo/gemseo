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
#                        documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
SSBJ Disciplines wrappers
*************************
"""

from __future__ import division, unicode_literals

import time
from numbers import Number

from gemseo.core.discipline import MDODiscipline
from gemseo.problems.sobieski.core import SobieskiProblem

DTYPE_COMPLEX = "complex128"
DTYPE_DOUBLE = "float64"


class SobieskiBaseWrapper(MDODiscipline):
    """Base wrapper for Sobieski problem discipline wrappers and JSON grammars."""

    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + ("dtype",)

    def __init__(self, dtype=DTYPE_DOUBLE):
        """Constructor.

        :param dtype: type of data, either "float64" or "complex128".
        :type dtype: str
        """
        super(SobieskiBaseWrapper, self).__init__(auto_detect_grammar_files=True)
        self.dtype = dtype
        self.sobieski_problem = SobieskiProblem(dtype=dtype)
        self.default_inputs = self.sobieski_problem.get_default_inputs(
            self.get_input_data_names()
        )
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def __setstate__(self, d):
        """Used by pickle to define what to deserialize.

        :param d : update self dict from d to deserialize
        """
        super(SobieskiBaseWrapper, self).__setstate__(d)
        self.sobieski_problem = SobieskiProblem(self.dtype)

    def _run(self):
        """Run the discipline."""
        raise NotImplementedError()


class SobieskiMission(SobieskiBaseWrapper):

    """Sobieski range wrapper using the Breguet formula."""

    _ATTR_TO_SERIALIZE = SobieskiBaseWrapper._ATTR_TO_SERIALIZE + ("enable_delay",)

    def __init__(self, dtype=DTYPE_DOUBLE, enable_delay=False):
        """Constructor of wrapper for range computation.

        :param dtype: type of data, either "float64" or "complex128".
        :type dtype: str
        """
        super(SobieskiMission, self).__init__(dtype=dtype)
        self.enable_delay = enable_delay

    def _run(self):
        """Compute range."""
        if self.enable_delay:
            if isinstance(self.enable_delay, Number):
                time.sleep(self.enable_delay)
            else:
                time.sleep(1.0)

        data_names = ["y_14", "y_24", "y_34", "x_shared"]
        y_14, y_24, y_34, x_shared = self.get_inputs_by_name(data_names)
        y_4 = self.sobieski_problem.blackbox_mission(x_shared, y_14, y_24, y_34)
        self.store_local_data(y_4=y_4)

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Compute the partial derivatives of all outputs wrt all inputs.

        :param inputs: Default value = None)
        :param outputs: Default value = None)
        """
        data_names = ["y_14", "y_24", "y_34", "x_shared"]
        y_14, y_24, y_34, x_shared = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.derive_blackbox_mission(
            x_shared, y_14, y_24, y_34
        )


class SobieskiStructure(SobieskiBaseWrapper):

    """Sobieski mass estimation wrapper."""

    def __init__(self, dtype=DTYPE_DOUBLE):
        """Constructor of wrapper for weight computation.

        :param dtype: type of data, either "float64" or "complex128".
        :type dtype: str
        """
        super(SobieskiStructure, self).__init__(dtype=dtype)

    def _run(self):
        """Compute weight."""

        data_names = ["x_1", "y_21", "y_31", "x_shared"]
        x_1, y_21, y_31, x_shared = self.get_inputs_by_name(data_names)
        y_1, y_11, y_12, y_14, g_1 = self.sobieski_problem.blackbox_structure(
            x_shared, y_21, y_31, x_1
        )
        self.store_local_data(y_1=y_1, y_11=y_11, y_12=y_12, y_14=y_14, g_1=g_1)

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Linearization of weight analysis.

        :param inputs: Default value = None)
        :param outputs: Default value = None)
        """
        data_names = ["x_1", "y_21", "y_31", "x_shared"]
        x_1, y_21, y_31, x_shared = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.derive_blackbox_structure(
            x_shared, y_21, y_31, x_1
        )


class SobieskiAerodynamics(SobieskiBaseWrapper):

    """Sobieski aerodynamic discipline wrapper."""

    def __init__(self, dtype=DTYPE_DOUBLE):
        """Constructor of wrapper for aerodynamic computation.

        :param dtype: type of data, "float64" or "complex128".
        :type dtype: str
        """
        super(SobieskiAerodynamics, self).__init__(dtype=dtype)

    def _run(self):
        """Compute aerodynamics."""
        data_names = ["x_2", "y_12", "y_32", "x_shared"]
        x_2, y_12, y_32, x_shared = self.get_inputs_by_name(data_names)
        y_2, y_21, y_23, y_24, g_2 = self.sobieski_problem.blackbox_aerodynamics(
            x_shared, y_12, y_32, x_2
        )
        self.store_local_data(y_2=y_2, y_21=y_21, y_23=y_23, y_24=y_24, g_2=g_2)

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Compute the partial derivatives of all outputs wrt all inputs.

        :param inputs: Default value = None)
        :param outputs: Default value = None)
        """
        data_names = ["x_2", "y_12", "y_32", "x_shared"]
        x_2, y_12, y_32, x_shared = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.derive_blackbox_aerodynamics(
            x_shared, y_12, y_32, x_2
        )


class SobieskiPropulsion(SobieskiBaseWrapper):
    """Sobieski propulsion propulsion wrapper."""

    def __init__(self, dtype=DTYPE_DOUBLE):
        """Constructor of wrapper for propulsion computation.

        :param dtype: type of data, either "float64" or "complex128"
        :type dtype: str
        """
        super(SobieskiPropulsion, self).__init__(dtype=dtype)

    def _run(self):
        """Compute propulsion."""
        data_names = ["x_3", "y_23", "x_shared"]

        x_3, y_23, x_shared = self.get_inputs_by_name(data_names)
        y_3, y_34, y_31, y_32, g_3 = self.sobieski_problem.blackbox_propulsion(
            x_shared, y_23, x_3
        )
        self.store_local_data(y_3=y_3, y_34=y_34, y_31=y_31, y_32=y_32, g_3=g_3)

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Compute the partial derivatives of all outputs wrt all inputs.

        :param inputs: Default value = None)
        :param outputs: Default value = None)
        """
        data_names = ["x_3", "y_23", "x_shared"]
        x_3, y_23, x_shared = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.derive_blackbox_propulsion(x_shared, y_23, x_3)
