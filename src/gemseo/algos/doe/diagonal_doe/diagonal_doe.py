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
#                           documentation
#        :author: Matthias De Lozzo
"""Build a diagonal DOE for scalable model construction."""

from __future__ import annotations

from collections.abc import Container
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Optional
from typing import Union

from numpy import hstack
from numpy import linspace
from numpy import newaxis

from gemseo.algos.doe.base_doe_library import BaseDOELibrary
from gemseo.algos.doe.base_doe_library import DOEAlgorithmDescription
from gemseo.algos.doe.diagonal_doe.settings.diagonal_doe_settings import (
    DiagonalDOE_Settings,
)

if TYPE_CHECKING:
    from gemseo.algos.design_space import DesignSpace
    from gemseo.typing import RealArray

OptionType = Optional[Union[str, int, float, bool, Container[str]]]


class DiagonalDOE(BaseDOELibrary):
    """Class used to create a diagonal DOE."""

    ALGORITHM_INFOS: ClassVar[dict[str, DOEAlgorithmDescription]] = {
        "DiagonalDOE": DOEAlgorithmDescription(
            algorithm_name="DiagonalDOE",
            description="Diagonal design of experiments",
            internal_algorithm_name="DiagonalDOE",
            library_name="GEMSEO",
            Settings=DiagonalDOE_Settings,
        )
    }

    def __init__(self, algo_name: str = "DiagonalDOE") -> None:  # noqa:D107
        super().__init__(algo_name)

    def _generate_unit_samples(
        self, design_space: DesignSpace, **settings: OptionType
    ) -> RealArray:
        n_samples = settings.get(self._N_SAMPLES)
        reverse = settings.get("reverse")

        name_by_index = {}
        start = 0
        for name in design_space:
            size = design_space.get_size(name)
            for index in range(start, start + size):
                name_by_index[index] = name

            start += size

        samples = []
        for index in range(design_space.dimension):
            if str(index) in reverse or name_by_index[index] in reverse:
                start = 1.0
                end = 0.0
            else:
                start = 0.0
                end = 1.0

            samples.append(linspace(start, end, n_samples)[:, newaxis])

        return hstack(samples)
