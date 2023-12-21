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
"""The design space for the scalable problem."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.problems.scalable.parametric.core.default_settings import DEFAULT_D_0
from gemseo.problems.scalable.parametric.core.scalable_design_space import (
    ScalableDesignSpace as _ScalableDesignSpace,
)
from gemseo.problems.scalable.parametric.core.scalable_discipline_settings import (
    DEFAULT_SCALABLE_DISCIPLINE_SETTINGS,
)
from gemseo.problems.scalable.parametric.core.scalable_discipline_settings import (
    ScalableDisciplineSettings,
)
from gemseo.problems.scalable.parametric.core.variable_names import get_u_local_name

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from numpy.typing import NDArray


class ScalableDesignSpace(ParameterSpace):
    """The design space for the scalable problem.

    It is the space in which the design and coupling variables vary. For all the
    variables, the lower bound is 0, the upper bound is 1 and the default value is 0.5.
    """

    def __init__(
        self,
        discipline_settings: Iterable[
            ScalableDisciplineSettings
        ] = DEFAULT_SCALABLE_DISCIPLINE_SETTINGS,
        d_0: int = DEFAULT_D_0,
        names_to_default_values: Mapping[str, NDArray[float]] = MappingProxyType({}),
        add_uncertain_variables: bool = False,
    ) -> None:
        r"""Args:
            discipline_settings: The configurations of the different disciplines.
                If ``None``,
                use a single discipline
                with default :class:`.ScalableDisciplineSettings`.
            d_0: The size of the shared design variable :math:`x_0`.
            names_to_default_values: The default values of the variables.
            add_uncertain_variables: Whether to add the uncertain variables
                impacting the coupling variables
                as :math:`y_{i,j}:=y_{i,j}+\epsilon_{i,j}`
                where :math:`\epsilon_{i,j}` are independent and identically distributed
                standard Gaussian variables.

        Notes:
            The lengths of ``n_local`` and ``n_coupling`` must be equal
            and correspond to the number of scalable disciplines.
            ``n_local[i]`` (resp. ``n_coupling[i]``)
            is the number of local design variables (resp. coupling variables)
            of the *i*-th scalable discipline.
        """  # noqa: D205 D212
        super().__init__()
        design_space = _ScalableDesignSpace(
            scalable_discipline_settings=discipline_settings,
            d_0=d_0,
            names_to_default_values=names_to_default_values,
        )
        for variable in design_space.variables:
            self.add_variable(
                name=variable.name,
                size=variable.size,
                l_b=variable.lower_bound,
                u_b=variable.upper_bound,
                value=variable.default_value,
            )

        if add_uncertain_variables:
            for index, settings in enumerate(discipline_settings):
                self.add_random_variable(
                    get_u_local_name(index + 1),
                    "OTNormalDistribution",
                    settings.p_i,
                )
