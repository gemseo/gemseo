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
"""A function searching for the output and Jacobian values in a database."""
from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace

if TYPE_CHECKING:
    from gemseo.core.mdofunctions.mdo_function import MDOFunction
    from gemseo.core.mdofunctions.mdo_function import ArrayType


class SetPtFromDatabase:
    """Set a function and Jacobian from a database."""

    def __init__(
        self,
        database: Database,
        design_space: DesignSpace,
        mdo_function: MDOFunction,
        normalize: bool = False,
        jac: bool = True,
        x_tolerance: float = 1e-10,
    ) -> None:
        """
        Args:
            database: The database to read.
            design_space: The design space used for normalization.
            mdo_function: The function where the data from the database will be set.
            normalize: If True, the values of the inputs are unnormalized before call.
            jac: If True, a Jacobian pointer is also generated.
            x_tolerance: The tolerance on the distance between inputs.
        """  # noqa: D205, D212, D415
        self.__database = database
        self.__design_space = design_space
        self.__mdo_function = mdo_function
        self.__normalize = normalize
        self.__jac = jac
        self.__x_tolerance = x_tolerance

        self.__name = self.__mdo_function.name

        self.__mdo_function.func = self._f_from_db

        if jac:
            self.__mdo_function.jac = self._j_from_db

    def __read_in_db(self, x_n: ArrayType, fname: str) -> ArrayType:
        """Read the value of a function in the database for a given input value.

        Args:
            x_n: The value of the inputs to evaluate the function.
            fname: The name of the function.

        Returns:
            The value of the function if present in the database.

        Raises:
            ValueError: If the input value is not in the database.
        """
        if self.__normalize:
            x_db = self.__design_space.unnormalize_vect(x_n)
        else:
            x_db = x_n
        val = self.__database.get_f_of_x(fname, x_db, self.__x_tolerance)
        if val is None:
            msg = (
                "Function {} evaluation relies only on the database, "
                "and {}( x ) is not in the database for x={}."
            ).format(fname, fname, x_db)
            raise ValueError(msg)
        return val

    def _f_from_db(self, x_n: ArrayType) -> ArrayType:
        """Evaluate the function from the database.

        Args:
            x_n: The value of the inputs to evaluate the function.

        Returns:
            The value of the function read in the database.
        """
        return self.__read_in_db(x_n, self.__name)

    def _j_from_db(self, x_n: ArrayType) -> ArrayType:
        """Evaluate the Jacobian function from the database.

        Args:
            x_n: The value of the inputs to evaluate the Jacobian function.

        Returns:
            The value of the Jacobian function read in the database.
        """
        return self.__read_in_db(x_n, f"@{self.__name}")

    @property
    def expects_normalized_inputs(self) -> bool:
        """Whether to normalize."""
        return self.__normalize
