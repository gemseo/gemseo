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

from typing import TYPE_CHECKING

from gemseo.core.discipline import Discipline
from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
from gemseo.problems.mdo.sobieski.core.utils import SobieskiBase

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray

    from gemseo.typing import StrKeyMapping


class SobieskiDisciplineWithSimpleGrammar(Discipline):
    """Base discipline for the Sobieski's SSBJ use case with simple grammars."""

    dtype: SobieskiBase.DataType
    """The data type for the NumPy arrays."""

    init_values: dict[str, ndarray]
    """The initial values of the design variables."""

    sobieski_problem: SobieskiProblem
    """The Sobieski's SSBJ use case defining the MDO problem, e.g. disciplines,
    constraints, design space and reference optimum."""

    default_grammar_type = Discipline.GrammarType.SIMPLE

    def __init__(
        self,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
    ) -> None:
        """
        Args:
            dtype: The data type for the NumPy arrays, either "float64" or "complex128".
        """  # noqa: D205 D212
        super().__init__()
        self.sobieski_problem = SobieskiProblem(dtype=dtype)
        self.init_values = {}
        self.dtype = dtype

    def _set_default_inputs(self) -> None:
        """Set the default inputs from the grammars and the :class:`SobieskiProblem`."""
        self.io.input_grammar.defaults = self.sobieski_problem.get_default_inputs(
            self.io.input_grammar
        )


class SobieskiMissionSG(SobieskiDisciplineWithSimpleGrammar):
    """Mission discipline of the Sobieski's SSBJ use case with a simple grammar.

    Compute the range with the Breguet formula.
    """

    enable_delay: bool | float
    """If ``True``, wait one second before computation.

    If a positive number, wait the corresponding number of seconds. If ``False``,
    compute directly.
    """

    def __init__(
        self,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
    ) -> None:
        super().__init__(dtype=dtype)
        self.io.input_grammar.update_from_names(("y_14", "y_24", "y_34", "x_shared"))
        self.io.output_grammar.update_from_names(["y_4"])
        self._set_default_inputs()

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        y_4 = self.sobieski_problem.mission.execute(
            input_data["x_shared"],
            input_data["y_14"],
            input_data["y_24"],
            input_data["y_34"],
        )
        return {"y_4": y_4}

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        local_data = self.io.data
        self.jac = self.sobieski_problem.mission.linearize(
            local_data["x_shared"],
            local_data["y_14"],
            local_data["y_24"],
            local_data["y_34"],
        )


class SobieskiStructureSG(SobieskiDisciplineWithSimpleGrammar):
    """Structure discipline of the Sobieski's SSBJ use case with a simple grammar."""

    def __init__(
        self,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
    ) -> None:
        super().__init__(dtype=dtype)
        self.io.input_grammar.update_from_names(["x_1", "y_21", "y_31", "x_shared"])
        self.io.output_grammar.update_from_names(["y_1", "y_11", "y_12", "y_14", "g_1"])
        self._set_default_inputs()

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        y_1, y_11, y_12, y_14, g_1 = self.sobieski_problem.structure.execute(
            input_data["x_shared"],
            input_data["y_21"],
            input_data["y_31"],
            input_data["x_1"],
        )
        return {
            "y_1": y_1,
            "y_11": y_11,
            "y_12": y_12,
            "y_14": y_14,
            "g_1": g_1,
        }

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        local_data = self.io.data
        self.jac = self.sobieski_problem.structure.linearize(
            local_data["x_shared"],
            local_data["y_21"],
            local_data["y_31"],
            local_data["x_1"],
        )


class SobieskiAerodynamicsSG(SobieskiDisciplineWithSimpleGrammar):
    """Aerodynamics discipline for the Sobieski's SSBJ use case with simple grammar."""

    def __init__(
        self,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
    ) -> None:
        super().__init__(dtype=dtype)
        self.io.input_grammar.update_from_names(["x_2", "y_12", "y_32", "x_shared"])
        self.io.output_grammar.update_from_names(["y_2", "y_21", "y_23", "y_24", "g_2"])
        self._set_default_inputs()

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        y_2, y_21, y_23, y_24, g_2 = self.sobieski_problem.aerodynamics.execute(
            input_data["x_shared"],
            input_data["y_12"],
            input_data["y_32"],
            input_data["x_2"],
        )
        return {
            "y_2": y_2,
            "y_21": y_21,
            "y_23": y_23,
            "y_24": y_24,
            "g_2": g_2,
        }

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        local_data = self.io.data
        self.jac = self.sobieski_problem.aerodynamics.linearize(
            local_data["x_shared"],
            local_data["y_12"],
            local_data["y_32"],
            local_data["x_2"],
        )


class SobieskiPropulsionSG(SobieskiDisciplineWithSimpleGrammar):
    """Propulsion discipline of the Sobieski's SSBJ use case."""

    def __init__(
        self,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
    ) -> None:
        super().__init__(dtype=dtype)
        self.io.input_grammar.update_from_names(["x_3", "y_23", "x_shared"])
        self.io.output_grammar.update_from_names(["y_3", "y_34", "y_31", "y_32", "g_3"])
        self._set_default_inputs()

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        y_3, y_34, y_31, y_32, g_3 = self.sobieski_problem.propulsion.execute(
            input_data["x_shared"], input_data["y_23"], input_data["x_3"]
        )
        return {
            "y_3": y_3,
            "y_34": y_34,
            "y_31": y_31,
            "y_32": y_32,
            "g_3": g_3,
        }

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        local_data = self.io.data
        self.jac = self.sobieski_problem.propulsion.linearize(
            local_data["x_shared"], local_data["y_23"], local_data["x_3"]
        )
