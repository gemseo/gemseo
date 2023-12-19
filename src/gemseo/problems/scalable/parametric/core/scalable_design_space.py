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
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The design space."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from numpy import full
from numpy import ndarray
from numpy import ones
from numpy import zeros

from gemseo.problems.scalable.parametric.core.default_settings import DEFAULT_D_0
from gemseo.problems.scalable.parametric.core.scalable_discipline_settings import (
    DEFAULT_SCALABLE_DISCIPLINE_SETTINGS,
)
from gemseo.problems.scalable.parametric.core.scalable_discipline_settings import (
    ScalableDisciplineSettings,
)
from gemseo.problems.scalable.parametric.core.variable import Variable
from gemseo.problems.scalable.parametric.core.variable_names import (
    SHARED_DESIGN_VARIABLE_NAME,
)
from gemseo.problems.scalable.parametric.core.variable_names import get_coupling_name
from gemseo.problems.scalable.parametric.core.variable_names import get_x_local_name

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from numpy.typing import NDArray


class ScalableDesignSpace:
    """The design space for the scalable problem.

    It is the space in which the design and coupling variables vary. For all the
    variables, the lower bound is 0, the upper bound is 1 and the default value is 0.5.
    """

    variables: list[Variable]
    """The design variables."""

    def __init__(
        self,
        scalable_discipline_settings: Iterable[
            ScalableDisciplineSettings
        ] = DEFAULT_SCALABLE_DISCIPLINE_SETTINGS,
        d_0: int = DEFAULT_D_0,
        names_to_default_values: Mapping[str, ndarray] = MappingProxyType({}),
    ) -> None:
        r"""
        Args:
            scalable_discipline_settings: The configurations of the scalable
                disciplines.
            d_0: The size of the shared design variable :math:`x_0`.
            names_to_default_values: The default values of the variables.
        """  # noqa: D205 D212
        self.variables = []
        name = SHARED_DESIGN_VARIABLE_NAME
        self.__add_variable(name, d_0, names_to_default_values.get(name))
        for index, settings in enumerate(scalable_discipline_settings):
            name = get_x_local_name(index + 1)
            self.__add_variable(
                name,
                settings.d_i,
                names_to_default_values.get(name),
            )
        for index, settings in enumerate(scalable_discipline_settings):
            name = get_coupling_name(index + 1)
            self.__add_variable(
                name,
                settings.p_i,
                names_to_default_values.get(name),
            )

    def __add_variable(
        self,
        name: str,
        size: int,
        default_value: NDArray[float] | None,
    ) -> None:
        """Add a variable to the design space.

        Args:
            name: The name of the variable.
            size: The size of the variable.
            default_value: The default value of the variable.
        """
        if default_value is None:
            default_value = full(size, 0.5)

        self.variables.append(
            Variable(name, size, zeros(size), ones(size), default_value)
        )
