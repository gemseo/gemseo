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
"""The design space of the Sobieski's SSBJ problem."""

from __future__ import annotations

from pathlib import Path

from numpy import complex128

from gemseo.algos.design_space import DesignSpace
from gemseo.problems.sobieski.core.utils import SobieskiBase


class SobieskiDesignSpace(DesignSpace):
    """The design space of the Sobieski's SSBJ problem.

    .. note:: This design space includes both the design and coupling variables.
    """

    __design_variable_names: tuple[str, ...]
    """The names of the design variables."""

    __coupling_variable_names: tuple[str, ...]
    """The names of the coupling variables."""

    def __init__(
        self,
        use_original_names: bool = True,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
        use_original_design_variables_order: bool = False,
    ) -> None:
        """
        Args:
            use_original_names: Whether to use physical naming
                instead of original notations.
            dtype: The data type for the NumPy arrays, either "float64" or "complex128".
            use_original_design_variables_order: Whether to sort
                the :attr:`.DesignSpace` as in :cite:`SobieskiBLISS98`.
                If so,
                the order of the design variables will be
                ``"x_1"``, ``"x_2"``, ``"x_3"`` and ``"x_shared"``.
                Otherwise, ``"x_shared"``, ``"x_1"``, ``"x_2"`` and ``"x_3"``.
        """  # noqa: D205 D212
        super().__init__()
        part_a = "original_" if use_original_design_variables_order else ""
        if use_original_names:
            self.__design_variable_names = ("x_shared", "x_1", "x_2", "x_3")
            self.__coupling_variable_names = (
                "y_14",
                "y_32",
                "y_31",
                "y_24",
                "y_34",
                "y_23",
                "y_21",
                "y_12",
            )
            part_b = ""
        else:
            self.__design_variable_names = (
                "t_c",
                "altitude",
                "mach",
                "ar",
                "sweep",
                "area",
                "taper_ratio",
                "wingbox_area",
                "cf",
                "throttle",
            )
            self.__coupling_variable_names = (
                "t_w_4",
                "f_w",
                "esf",
                "e_w",
                "cl_cd",
                "sfc",
                "cd",
                "cl",
                "t_w_2",
                "twist",
            )
            part_b = "_pn"

        design_space = DesignSpace.from_csv(
            Path(__file__).parent / f"sobieski_{part_a}design_space{part_b}.csv"
        )
        self.extend(design_space)
        if dtype == complex128:
            self.to_complex()

    def filter_design_variables(self, copy: bool = False) -> SobieskiDesignSpace:
        """Filter the design space to keep only the design variables.

        Args:
            copy: Whether to filter a copy of the design space
                or the design space itself.

        Returns:
            Either the filtered original design space or a copy.
        """
        return self.filter(self.__design_variable_names, copy=copy)

    def filter_coupling_variables(self, copy: bool = False) -> SobieskiDesignSpace:
        """Filter the design space to keep only the coupling variables.

        Args:
            copy: Whether to filter a copy of the design space
                or the design space itself.

        Returns:
            Either the filtered original design space or a copy.
        """
        return self.filter(self.__coupling_variable_names, copy=copy)
