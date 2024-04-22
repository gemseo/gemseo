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
#        :author: Matthias
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Thin plate spline (TPS) regression."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar

from gemseo.mlearning.regression.algos.rbf import RBFRegressor

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.core.algos.ml_algo import TransformerType
    from gemseo.typing import NumberArray


class TPSRegressor(RBFRegressor):
    """Thin plate spline (TPS) regression."""

    SHORT_ALGO_NAME: ClassVar[str] = "TPS"

    def __init__(  # noqa: D107
        self,
        data: IODataset,
        transformer: TransformerType = RBFRegressor.IDENTITY,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
        smooth: float = 0.0,
        norm: str
        | Callable[[NumberArray, NumberArray], float] = RBFRegressor.EUCLIDEAN,
        **parameters: Any,
    ) -> None:
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            function=RBFRegressor.Function.THIN_PLATE,
            smooth=smooth,
            norm=norm,
            **parameters,
        )
