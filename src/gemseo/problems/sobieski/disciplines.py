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
"""The disciplines of the Sobieski's SSBJ use case."""

from __future__ import division, unicode_literals

import time
from numbers import Number
from typing import Any, Iterable, List, Mapping, Optional, Union

from numpy import array

from gemseo.core.discipline import MDODiscipline
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.core.utils import SobieskiBase


class SobieskiDiscipline(MDODiscipline):
    """Abstract base discipline for the Sobieski's SSBJ use case.

    Attributes:
        dtype (str): The data type for the NumPy arrays.
        sobieski_problem (SobieskiProblem): The Sobieski's SSBJ use case
            defining the MDO problem,
            e.g. disciplines, constraints, design space and reference optimum.
    """

    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + ("dtype",)

    def __init__(
        self,
        dtype=SobieskiBase.DTYPE_DOUBLE,  # type: str
    ):  # type: (...) -> None
        """
        Args:
            dtype: The data type for the NumPy arrays, either "float64" or "complex128".
        """
        super(SobieskiDiscipline, self).__init__(auto_detect_grammar_files=True)
        self.dtype = dtype
        self.sobieski_problem = SobieskiProblem(dtype=dtype)
        self.default_inputs = self.sobieski_problem.get_default_inputs(
            self.get_input_data_names()
        )
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def __setstate__(
        self,
        state_dict,  # type: Mapping[str, Any]
    ):  # type: (...) -> None
        super(SobieskiDiscipline, self).__setstate__(state_dict)
        self.sobieski_problem = SobieskiProblem(self.dtype)

    def _run(self):  # type: (...) -> None
        raise NotImplementedError()


class SobieskiMission(SobieskiDiscipline):
    """Mission discipline of the Sobieski's SSBJ use case.

    Compute the range with the Breguet formula.

    Attributes:
        enable_delay (Union[bool,float]): If ``True``,
            wait one second before computation.
            If a positive number, wait the corresponding number of seconds.
            If ``False``, compute directly.
    """

    _ATTR_TO_SERIALIZE = SobieskiDiscipline._ATTR_TO_SERIALIZE + ("enable_delay",)

    def __init__(
        self,
        dtype=SobieskiBase.DTYPE_DOUBLE,  # type: str
        enable_delay=False,  # type: Union[bool,float]
    ):  # type: (...) -> None
        """
        Args:
            enable_delay: If ``True``, wait one second before computation.
                If a positive number, wait the corresponding number of seconds.
                If ``False``, compute directly.
        """
        super(SobieskiMission, self).__init__(dtype=dtype)
        self.enable_delay = enable_delay

    def _run(self):  # type: (...) -> None
        if self.enable_delay:
            if isinstance(self.enable_delay, Number):
                time.sleep(self.enable_delay)
            else:
                time.sleep(1.0)

        data_names = ["y_14", "y_24", "y_34", "x_shared"]
        y_14, y_24, y_34, x_shared = self.get_inputs_by_name(data_names)
        y_4 = self.sobieski_problem.mission.execute(x_shared, y_14, y_24, y_34)
        self.store_local_data(y_4=y_4)

    def _compute_jacobian(
        self,
        inputs=None,  # type: Optional[Iterable[str]]
        outputs=None,  # type: Optional[Iterable[str]]
    ):  # type: (...)-> None
        data_names = ["y_14", "y_24", "y_34", "x_shared"]
        y_14, y_24, y_34, x_shared = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.mission.linearize(x_shared, y_14, y_24, y_34)


class SobieskiStructure(SobieskiDiscipline):
    """Structure discipline of the Sobieski's SSBJ use case."""

    def __init__(
        self,
        dtype=SobieskiBase.DTYPE_DOUBLE,  # type: str
    ):  # type: (...) -> None
        super(SobieskiStructure, self).__init__(dtype=dtype)
        self.default_inputs["c_0"] = array([self.sobieski_problem.constants[0]])
        self.default_inputs["c_1"] = array([self.sobieski_problem.constants[1]])
        self.default_inputs["c_2"] = array([self.sobieski_problem.constants[2]])

    def _run(self):  # type: (...) -> None
        data_names = ["x_1", "y_21", "y_31", "x_shared", "c_0", "c_1", "c_2"]
        x_1, y_21, y_31, x_shared, c_0, c_1, c_2 = self.get_inputs_by_name(data_names)
        y_1, y_11, y_12, y_14, g_1 = self.sobieski_problem.structure.execute(
            x_shared, y_21, y_31, x_1, c_0=c_0[0], c_1=c_1[0], c_2=c_2[0]
        )
        self.store_local_data(y_1=y_1, y_11=y_11, y_12=y_12, y_14=y_14, g_1=g_1)

    def _compute_jacobian(
        self,
        inputs=None,  # type: Optional[Iterable[str]]
        outputs=None,  # type: Optional[Iterable[str]]
    ):  # type: (...)-> None
        data_names = ["x_1", "y_21", "y_31", "x_shared", "c_2"]
        x_1, y_21, y_31, x_shared, c_2 = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.structure.linearize(
            x_shared, y_21, y_31, x_1, c_2=c_2[0]
        )


class SobieskiAerodynamics(SobieskiDiscipline):
    """Aerodynamics discipline for the Sobieski's SSBJ use case."""

    def __init__(
        self,
        dtype=SobieskiBase.DTYPE_DOUBLE,  # type: str
    ):  # type: (...) -> None
        super(SobieskiAerodynamics, self).__init__(dtype=dtype)
        self.default_inputs["c_4"] = array([self.sobieski_problem.constants[4]])

    def _run(self):  # type: (...)-> None
        data_names = ["x_2", "y_12", "y_32", "x_shared", "c_4"]
        x_2, y_12, y_32, x_shared, c_4 = self.get_inputs_by_name(data_names)
        y_2, y_21, y_23, y_24, g_2 = self.sobieski_problem.aerodynamics.execute(
            x_shared, y_12, y_32, x_2, c_4=c_4[0]
        )
        self.store_local_data(y_2=y_2, y_21=y_21, y_23=y_23, y_24=y_24, g_2=g_2)

    def _compute_jacobian(
        self,
        inputs=None,  # type: Optional[Iterable[str]]
        outputs=None,  # type: Optional[Iterable[str]]
    ):  # type: (...)-> None
        data_names = ["x_2", "y_12", "y_32", "x_shared", "c_4"]
        x_2, y_12, y_32, x_shared, c_4 = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.aerodynamics.linearize(
            x_shared, y_12, y_32, x_2, c_4=c_4[0]
        )


class SobieskiPropulsion(SobieskiDiscipline):
    """Propulsion discipline of the Sobieski's SSBJ use case."""

    def __init__(
        self,
        dtype=SobieskiBase.DTYPE_DOUBLE,  # type: str
    ):  # type: (...) -> None
        super(SobieskiPropulsion, self).__init__(dtype=dtype)
        self.default_inputs["c_3"] = array([self.sobieski_problem.constants[3]])

    def _run(self):  # type: (...)-> None
        data_names = ["x_3", "y_23", "x_shared", "c_3"]

        x_3, y_23, x_shared, c_3 = self.get_inputs_by_name(data_names)
        y_3, y_34, y_31, y_32, g_3 = self.sobieski_problem.propulsion.execute(
            x_shared, y_23, x_3, c_3=c_3[0]
        )
        self.store_local_data(y_3=y_3, y_34=y_34, y_31=y_31, y_32=y_32, g_3=g_3)

    def _compute_jacobian(
        self,
        inputs=None,  # type: Optional[Iterable[str]]
        outputs=None,  # type: Optional[Iterable[str]]
    ):  # type: (...)-> None
        data_names = ["x_3", "y_23", "x_shared", "c_3"]
        x_3, y_23, x_shared, c_3 = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.propulsion.linearize(
            x_shared, y_23, x_3, c_3=c_3[0]
        )


def create_disciplines(
    dtype,  # type: str
):  # type: (...) -> List[SobieskiDiscipline]
    """Instantiate the structure, aerodynamics, propulsion and mission disciplines.

    Args:
        dtype: The NumPy type for data arrays, either "float64" or "complex128".

    Returns:
        Instances of the structure, aerodynamics, propulsion and mission disciplines.
    """
    return [
        SobieskiStructure(dtype),
        SobieskiAerodynamics(dtype),
        SobieskiPropulsion(dtype),
        SobieskiMission(dtype),
    ]