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
"""The disciplines of the Sobieski's SSBJ use case."""

from __future__ import annotations

import time
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING

from numpy import array

from gemseo.core.discipline import Discipline
from gemseo.disciplines.remapping import RemappingDiscipline
from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
from gemseo.problems.mdo.sobieski.core.utils import SobieskiBase

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping


class SobieskiDiscipline(Discipline):
    """Abstract base discipline for the Sobieski's SSBJ use case."""

    dtype: SobieskiBase.DataType
    """The data type for the NumPy arrays."""

    sobieski_problem: SobieskiProblem
    """The Sobieski's SSBJ use case defining the MDO problem, e.g. disciplines,
    constraints, design space and reference optimum."""

    GRAMMAR_DIRECTORY = Path(__file__).parent / "grammars"
    auto_detect_grammar_files = True

    _ATTR_NOT_TO_SERIALIZE = Discipline._ATTR_NOT_TO_SERIALIZE.union(
        [
            "sobieski_problem",
        ],
    )

    def __init__(
        self,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
    ) -> None:
        """
        Args:
            dtype: The data type for the NumPy arrays, either "float64" or "complex128".
        """  # noqa: D205 D212
        super().__init__()
        self.dtype = dtype
        self.sobieski_problem = SobieskiProblem(dtype=dtype)
        self.io.input_grammar.defaults = self.sobieski_problem.get_default_inputs(
            self.io.input_grammar
        )

    def __setstate__(self, state: StrKeyMapping) -> None:
        super().__setstate__(state)
        self.sobieski_problem = SobieskiProblem(self.dtype)

    @classmethod
    def create_with_physical_naming(
        cls,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
    ) -> RemappingDiscipline:
        """
        Args:
            dtype: The data type for the NumPy arrays, either "float64" or "complex128".
        """  # noqa: D205 D212
        raise NotImplementedError


class SobieskiMission(SobieskiDiscipline):
    """Mission discipline of the Sobieski's SSBJ use case.

    Compute the range with the Breguet formula.
    """

    # TODO: API: move enable_delay to a derived class.
    enable_delay: bool | float
    """If ``True``, wait one second before computation.

    If a positive number, wait the corresponding number of seconds. If ``False``,
    compute directly.
    """

    def __init__(
        self,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
        enable_delay: bool | float = False,
    ) -> None:
        """
        Args:
            enable_delay: If ``True``, wait one second before computation.
                If a positive number, wait the corresponding number of seconds.
                If ``False``, compute directly.
        """  # noqa: D205 D212
        super().__init__(dtype=dtype)
        self.enable_delay = enable_delay

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        if self.enable_delay:
            if isinstance(self.enable_delay, Number):
                time.sleep(self.enable_delay)
            else:
                time.sleep(1.0)

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
        if self.enable_delay:
            if isinstance(self.enable_delay, Number):
                time.sleep(self.enable_delay)
            else:
                time.sleep(1.0)

        local_data = self.io.data
        self.jac = self.sobieski_problem.mission.linearize(
            local_data["x_shared"],
            local_data["y_14"],
            local_data["y_24"],
            local_data["y_34"],
        )

    @classmethod
    def create_with_physical_naming(
        cls,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
        enable_delay: bool | float = False,
    ) -> RemappingDiscipline:
        """
        Args:
            enable_delay: If ``True``, wait one second before computation.
                If a positive number, wait the corresponding number of seconds.
                If ``False``, compute directly.
        """  # noqa: D205 D212
        return RemappingDiscipline(
            cls(dtype=dtype, enable_delay=enable_delay),
            {
                "t_w_4": ("y_14", 0),
                "f_w": ("y_14", 1),
                "altitude": ("x_shared", 1),
                "mach": ("x_shared", 2),
                "cl_cd": "y_24",
                "sfc": "y_34",
            },
            {"range": "y_4"},
        )


class SobieskiStructure(SobieskiDiscipline):
    """Structure discipline of the Sobieski's SSBJ use case."""

    def __init__(  # noqa: D107
        self,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
    ) -> None:
        super().__init__(dtype=dtype)
        self.io.input_grammar.defaults["c_0"] = array([
            self.sobieski_problem.constants[0]
        ])
        self.io.input_grammar.defaults["c_1"] = array([
            self.sobieski_problem.constants[1]
        ])
        self.io.input_grammar.defaults["c_2"] = array([
            self.sobieski_problem.constants[2]
        ])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        y_1, y_11, y_12, y_14, g_1 = self.sobieski_problem.structure.execute(
            input_data["x_shared"],
            input_data["y_21"],
            input_data["y_31"],
            input_data["x_1"],
            c_0=input_data["c_0"][0],
            c_1=input_data["c_1"][0],
            c_2=input_data["c_2"][0],
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
            c_2=local_data["c_2"][0],
        )

    @classmethod
    def create_with_physical_naming(  # noqa: D102
        cls, dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT
    ) -> RemappingDiscipline:
        return RemappingDiscipline(
            cls(dtype=dtype),
            {
                "cl": "y_21",
                "e_w": "y_31",
                "t_c": ("x_shared", 0),
                "ar": ("x_shared", 3),
                "sweep": ("x_shared", 4),
                "area": ("x_shared", 5),
                "taper_ratio": ("x_1", 0),
                "wingbox_area": ("x_1", 1),
                "min_f_w": "c_0",
                "m_w": "c_1",
                "max_lf": "c_2",
            },
            {
                "y_1": "y_1",
                "y_11": "y_11",
                "t_w_4": ("y_14", 0),
                "t_w_2": ("y_12", 0),
                "f_w": ("y_14", 1),
                "twist": ("y_12", 1),
                "stress": ("g_1", range(5)),
                "twist_c": ("g_1", range(5, 7)),
            },
        )


class SobieskiAerodynamics(SobieskiDiscipline):
    """Aerodynamics discipline for the Sobieski's SSBJ use case."""

    def __init__(  # noqa: D107
        self,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
    ) -> None:
        super().__init__(dtype=dtype)
        self.io.input_grammar.defaults["c_4"] = array([
            self.sobieski_problem.constants[4]
        ])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        y_2, y_21, y_23, y_24, g_2 = self.sobieski_problem.aerodynamics.execute(
            input_data["x_shared"],
            input_data["y_12"],
            input_data["y_32"],
            input_data["x_2"],
            c_4=input_data["c_4"][0],
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
            c_4=local_data["c_4"][0],
        )

    @classmethod
    def create_with_physical_naming(  # noqa: D102
        cls,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
    ) -> RemappingDiscipline:
        return RemappingDiscipline(
            cls(dtype=dtype),
            {
                "esf": "y_32",
                "t_w_2": ("y_12", 0),
                "twist": ("y_12", 1),
                "cf": "x_2",
                "t_c": ("x_shared", 0),
                "altitude": ("x_shared", 1),
                "mach": ("x_shared", 2),
                "sweep": ("x_shared", 4),
                "area": ("x_shared", 5),
                "min_cd": "c_4",
            },
            {
                "y_2": "y_2",
                "cl": "y_21",
                "cd": "y_23",
                "cl_cd": "y_24",
                "dp_dx": "g_2",
            },
        )


class SobieskiPropulsion(SobieskiDiscipline):
    """Propulsion discipline of the Sobieski's SSBJ use case."""

    def __init__(  # noqa: D107
        self,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
    ) -> None:
        super().__init__(dtype=dtype)
        self.io.input_grammar.defaults["c_3"] = array([
            self.sobieski_problem.constants[3]
        ])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        y_3, y_34, y_31, y_32, g_3 = self.sobieski_problem.propulsion.execute(
            input_data["x_shared"],
            input_data["y_23"],
            input_data["x_3"],
            c_3=input_data["c_3"][0],
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
            local_data["x_shared"],
            local_data["y_23"],
            local_data["x_3"],
            c_3=local_data["c_3"][0],
        )

    @classmethod
    def create_with_physical_naming(  # noqa: D102
        cls,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
    ) -> RemappingDiscipline:
        return RemappingDiscipline(
            cls(dtype=dtype),
            {
                "throttle": "x_3",
                "cd": "y_23",
                "altitude": ("x_shared", 1),
                "mach": ("x_shared", 2),
                "ref_weight": "c_3",
            },
            {
                "y_3": "y_3",
                "sfc": ("y_3", 0),
                "e_w": ("y_3", 1),
                "esf_c": ("g_3", range(2)),
                "esf": "y_32",
                "throttle_c": ("g_3", 2),
                "temperature": ("g_3", 3),
            },
        )


def create_disciplines(
    dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
) -> list[SobieskiDiscipline]:
    """Instantiate the structure, aerodynamics, propulsion and mission disciplines.

    Args:
        dtype: The NumPy type for data arrays, either "float64" or "complex128".

    Returns:
        The structure, aerodynamics, propulsion and mission disciplines.
    """
    return [
        SobieskiStructure(dtype),
        SobieskiAerodynamics(dtype),
        SobieskiPropulsion(dtype),
        SobieskiMission(dtype),
    ]


def create_disciplines_with_physical_naming(
    dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
) -> list[RemappingDiscipline]:
    """Instantiate the structure, aerodynamics, propulsion and mission disciplines.

    Use a physical naming for the input and output variables.

    Args:
        dtype: The NumPy type for data arrays, either "float64" or "complex128".

    Returns:
        The structure, aerodynamics, propulsion and mission disciplines.
    """
    return [
        SobieskiStructure.create_with_physical_naming(dtype),
        SobieskiAerodynamics.create_with_physical_naming(dtype),
        SobieskiPropulsion.create_with_physical_naming(dtype),
        SobieskiMission.create_with_physical_naming(dtype),
    ]
