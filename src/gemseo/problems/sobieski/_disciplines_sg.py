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
"""The disciplines of the Sobieski's SSBJ use case with simple grammars.

This disciplines use simple grammars rather than JSON ones mainly for proof of concept.
Please use the JSON versions with enhanced checks and features.
"""
from __future__ import annotations

from typing import Iterable

from numpy import ndarray

from gemseo.core.discipline import MDODiscipline
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.core.utils import SobieskiBase


class SobieskiDisciplineWithSimpleGrammar(MDODiscipline):
    """Base discipline for the Sobieski's SSBJ use case with simple grammars."""

    dtype: str
    """The data type for the NumPy arrays."""

    init_values: dict[str, ndarray]
    """The initial values of the design variables."""

    sobieski_problem: SobieskiProblem
    """The Sobieski's SSBJ use case defining the MDO problem, e.g. disciplines,
    constraints, design space and reference optimum."""

    def __init__(
        self,
        dtype: str = SobieskiBase.DTYPE_DOUBLE,
    ) -> None:
        """
        Args:
            dtype: The data type for the NumPy arrays, either "float64" or "complex128".
        """
        super().__init__(grammar_type=MDODiscipline.SIMPLE_GRAMMAR_TYPE)
        self.sobieski_problem = SobieskiProblem(dtype=dtype)
        self.init_values = {}
        self.dtype = dtype

    def _set_default_inputs(self) -> None:
        """Set the default inputs from the grammars and the :class:`SobieskiProblem`."""
        self.default_inputs = self.sobieski_problem.get_default_inputs(
            self.get_input_data_names()
        )

    def _run(self) -> None:
        raise NotImplementedError()


class SobieskiMissionSG(SobieskiDisciplineWithSimpleGrammar):
    """Mission discipline of the Sobieski's SSBJ use case with a simple grammar.

    Compute the range with the Breguet formula.
    """

    enable_delay: bool | float
    """If ``True``, wait one second before computation.

    If a positive number, wait the corresponding number of seconds. If ``False``, compute
    directly.
    """

    def __init__(
        self,
        dtype: str = SobieskiBase.DTYPE_DOUBLE,
    ) -> None:
        super().__init__(dtype=dtype)
        self.input_grammar.update(
            dict.fromkeys(("y_14", "y_24", "y_34", "x_shared"), ndarray)
        )
        self.output_grammar.update({"y_4": ndarray})
        self._set_default_inputs()

    def _run(self) -> None:
        data_names = ["y_14", "y_24", "y_34", "x_shared"]
        y_14, y_24, y_34, x_shared = self.get_inputs_by_name(data_names)
        y_4 = self.sobieski_problem.mission.execute(x_shared, y_14, y_24, y_34)
        self.store_local_data(y_4=y_4)

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        data_names = ["y_14", "y_24", "y_34", "x_shared"]
        y_14, y_24, y_34, x_shared = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.mission.linearize(x_shared, y_14, y_24, y_34)


class SobieskiStructureSG(SobieskiDisciplineWithSimpleGrammar):
    """Structure discipline of the Sobieski's SSBJ use case with a simple grammar."""

    def __init__(
        self,
        dtype: str = SobieskiBase.DTYPE_DOUBLE,
    ) -> None:
        super().__init__(dtype=dtype)
        self.input_grammar.update(
            dict.fromkeys(["x_1", "y_21", "y_31", "x_shared"], ndarray)
        )
        self.output_grammar.update(
            dict.fromkeys(["y_1", "y_11", "y_12", "y_14", "g_1"], ndarray)
        )
        self._set_default_inputs()

    def _run(self) -> None:
        data_names = ["x_shared", "y_21", "y_31", "x_1"]
        x_shared, y_21, y_31, x_1 = self.get_inputs_by_name(data_names)
        y_1, y_11, y_12, y_14, g_1 = self.sobieski_problem.structure.execute(
            x_shared, y_21, y_31, x_1
        )
        self.store_local_data(y_1=y_1, y_11=y_11, y_12=y_12, y_14=y_14, g_1=g_1)

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        data_names = ["x_shared", "y_21", "y_31", "x_1"]
        x_shared, y_21, y_31, x_1 = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.structure.linearize(x_shared, y_21, y_31, x_1)


class SobieskiAerodynamicsSG(SobieskiDisciplineWithSimpleGrammar):
    """Aerodynamics discipline for the Sobieski's SSBJ use case with simple grammar."""

    def __init__(
        self,
        dtype: str = SobieskiBase.DTYPE_DOUBLE,
    ) -> None:
        super().__init__(dtype=dtype)
        self.input_grammar.update(
            dict.fromkeys(["x_2", "y_12", "y_32", "x_shared"], ndarray)
        )
        self.output_grammar.update(
            dict.fromkeys(["y_2", "y_21", "y_23", "y_24", "g_2"], ndarray)
        )
        self._set_default_inputs()

    def _run(self) -> None:
        data_names = ["x_2", "y_12", "y_32", "x_shared"]
        x_2, y_12, y_32, x_shared = self.get_inputs_by_name(data_names)
        y_2, y_21, y_23, y_24, g_2 = self.sobieski_problem.aerodynamics.execute(
            x_shared, y_12, y_32, x_2
        )
        self.store_local_data(y_2=y_2, y_21=y_21, y_23=y_23, y_24=y_24, g_2=g_2)

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        data_names = ["x_2", "y_12", "y_32", "x_shared"]
        x_2, y_12, y_32, x_shared = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.aerodynamics.linearize(
            x_shared, y_12, y_32, x_2
        )


class SobieskiPropulsionSG(SobieskiDisciplineWithSimpleGrammar):
    """Propulsion discipline of the Sobieski's SSBJ use case."""

    def __init__(
        self,
        dtype: str = SobieskiBase.DTYPE_DOUBLE,
    ) -> None:
        super().__init__(dtype=dtype)
        self.input_grammar.update(dict.fromkeys(["x_3", "y_23", "x_shared"], ndarray))
        self.output_grammar.update(
            dict.fromkeys(["y_3", "y_34", "y_31", "y_32", "g_3"], ndarray)
        )
        self._set_default_inputs()

    def _run(self) -> None:
        data_names = ["x_3", "y_23", "x_shared"]
        x_3, y_23, x_shared = self.get_inputs_by_name(data_names)
        y_3, y_34, y_31, y_32, g_3 = self.sobieski_problem.propulsion.execute(
            x_shared, y_23, x_3
        )
        self.store_local_data(y_3=y_3, y_34=y_34, y_31=y_31, y_32=y_32, g_3=g_3)

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        data_names = ["x_3", "y_23", "x_shared"]
        x_3, y_23, x_shared = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.propulsion.linearize(x_shared, y_23, x_3)
