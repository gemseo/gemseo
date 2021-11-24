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

from numpy import ndarray

from gemseo.core.discipline import MDODiscipline
from gemseo.problems.sobieski.core import SobieskiProblem

DTYPE_COMPLEX = "complex128"
DTYPE_DOUBLE = "float64"


class SobieskiBaseWrapperSimpleGram(MDODiscipline):
    """Base wrapper for Sobieski problem discipline wrappers and SimpleGrammar."""

    def __init__(self, dtype=DTYPE_DOUBLE):
        """Constructor.

        :param dtype: type of data, either "float64" or "complex128"
        :type dtype: str
        """
        super(SobieskiBaseWrapperSimpleGram, self).__init__(
            auto_detect_grammar_files=False,
            grammar_type=MDODiscipline.SIMPLE_GRAMMAR_TYPE,
        )
        self.sobieski_problem = SobieskiProblem(dtype=dtype)
        self.init_values = {}
        self.dtype = dtype

    def _set_default_inputs(self):
        """Sets the default inputs from grammrs and SobieskiProblem."""
        self.default_inputs = self.sobieski_problem.get_default_inputs(
            self.get_input_data_names()
        )

    def _run(self):
        """Run the discipline."""
        raise NotImplementedError()


class SobieskiMissionSG(SobieskiBaseWrapperSimpleGram):

    """Sobieski range wrapper using Breguet formula Uses simple grammars and not JSON,
    mainly for concept proof Please use JSON version with enhanced checks and
    features."""

    def __init__(self, dtype=DTYPE_DOUBLE):
        """Constructor of wrapper for weight computation.

        :param dtype: type of data, either "float64" or "complex128"
        :type dtype: str
        """
        super(SobieskiMissionSG, self).__init__(dtype=dtype)
        self.input_grammar.update_elements(
            **dict.fromkeys(("y_14", "y_24", "y_34", "x_shared"), ndarray)
        )
        self.output_grammar.update_elements(y_4=ndarray)
        self._set_default_inputs()

    def _run(self):
        """Compute range."""
        data_names = ["y_14", "y_24", "y_34", "x_shared"]
        y_14, y_24, y_34, x_shared = self.get_inputs_by_name(data_names)
        y_4 = self.sobieski_problem.blackbox_mission(x_shared, y_14, y_24, y_34)
        self.store_local_data(y_4=y_4)

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Linearization of weight analysis.

        :param inputs: Default value = None)
        :param outputs: Default value = None)
        """
        data_names = ["y_14", "y_24", "y_34", "x_shared"]
        y_14, y_24, y_34, x_shared = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.derive_blackbox_mission(
            x_shared, y_14, y_24, y_34
        )


class SobieskiStructureSG(SobieskiBaseWrapperSimpleGram):
    """Sobieski mass estimation wrapper Uses simple grammars and not JSON, mainly for
    concept proof Please use JSON version with enhanced checks and features."""

    def __init__(self, dtype=DTYPE_DOUBLE):
        """Constructor of wrapper for weight computation."""
        super(SobieskiStructureSG, self).__init__(dtype=dtype)
        self.input_grammar.update_elements(
            **dict.fromkeys(["x_1", "y_21", "y_31", "x_shared"], ndarray)
        )
        self.output_grammar.update_elements(
            **dict.fromkeys(["y_1", "y_11", "y_12", "y_14", "g_1"], ndarray)
        )
        self._set_default_inputs()

    def _run(self):
        """Compute weight."""
        data_names = ["x_shared", "y_21", "y_31", "x_1"]
        x_shared, y_21, y_31, x_1 = self.get_inputs_by_name(data_names)
        y_1, y_11, y_12, y_14, g_1 = self.sobieski_problem.blackbox_structure(
            x_shared, y_21, y_31, x_1
        )
        self.store_local_data(y_1=y_1, y_11=y_11, y_12=y_12, y_14=y_14, g_1=g_1)

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Linearization of weight analysis.

        :param inputs: Default value = None)
        :param outputs: Default value = None)
        """
        data_names = ["x_shared", "y_21", "y_31", "x_1"]
        x_shared, y_21, y_31, x_1 = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.derive_blackbox_structure(
            x_shared, y_21, y_31, x_1
        )


class SobieskiAerodynamicsSG(SobieskiBaseWrapperSimpleGram):

    """Sobieski aerodynamic discipline wrapper Uses simple grammars and not JSON, mainly
    for concept proof Please use JSON version with enhanced checks and features."""

    def __init__(self, dtype=DTYPE_DOUBLE):
        """Constructor of wrapper for aerodynamic computation.

        :param dtype: type of data, either "float64" or "complex128"
        :type dtype: str
        """
        super(SobieskiAerodynamicsSG, self).__init__(dtype=dtype)
        self.input_grammar.update_elements(
            **dict.fromkeys(["x_2", "y_12", "y_32", "x_shared"], ndarray)
        )
        self.output_grammar.update_elements(
            **dict.fromkeys(["y_2", "y_21", "y_23", "y_24", "g_2"], ndarray)
        )
        self._set_default_inputs()

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


class SobieskiPropulsionSG(SobieskiBaseWrapperSimpleGram):
    """Sobieski propulsion discipline wrapper Uses simple grammars and not JSON, mainly
    for concept proof Please use JSON version with enhanced checks and features."""

    def __init__(self, dtype=DTYPE_DOUBLE):
        """Constructor of wrapper for propulsion computation.

        :param dtype: type of data, either "float64" or "complex128"
        :type dtype: str
        """
        super(SobieskiPropulsionSG, self).__init__(dtype=dtype)
        self.input_grammar.update_elements(
            **dict.fromkeys(["x_3", "y_23", "x_shared"], ndarray)
        )
        self.output_grammar.update_elements(
            **dict.fromkeys(["y_3", "y_34", "y_31", "y_32", "g_3"], ndarray)
        )
        self._set_default_inputs()

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
