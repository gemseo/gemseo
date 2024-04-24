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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Charlie Vanaret
#                 Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The first strongly coupled discipline of the customizable Sellar MDO problem."""

from collections.abc import Iterable
from typing import ClassVar

from numpy import array
from numpy import maximum
from numpy import sign
from numpy import sqrt
from numpy import where
from scipy.sparse import csr_array
from scipy.sparse import diags

from gemseo.problems.mdo.sellar import WITH_2D_ARRAY
from gemseo.problems.mdo.sellar.base_sellar import BaseSellar
from gemseo.problems.mdo.sellar.variables import GAMMA
from gemseo.problems.mdo.sellar.variables import X_1
from gemseo.problems.mdo.sellar.variables import X_2
from gemseo.problems.mdo.sellar.variables import X_SHARED
from gemseo.problems.mdo.sellar.variables import Y_1
from gemseo.problems.mdo.sellar.variables import Y_2
from gemseo.utils.compatibility.scipy import SparseArrayType


class Sellar1(BaseSellar):
    """The discipline to compute the coupling variable :math:`y_1`."""

    _INPUT_NAMES: ClassVar[tuple[str]] = (X_1, X_SHARED, Y_2, GAMMA)

    _OUTPUT_NAMES: ClassVar[tuple[str]] = (Y_1,)

    __k: float
    """The shared coefficient controlling the coupling strength."""

    __zeros_n: SparseArrayType
    """The zero matrix of dimension n."""

    def __init__(self, n: int = 1, k: float = 1.0) -> None:
        """
        Args:
            k: The shared coefficient controlling the coupling strength.
        """  # noqa: D107 D205 D205 D212 D415
        super().__init__(n)
        self.__k = k
        self.__zeros_n = csr_array((n, n))

    def _run(self) -> None:
        x_1 = self._local_data[X_1]
        x_shared = self._local_data[X_SHARED]
        y_2 = self._local_data[Y_2]
        gamma = self._local_data[GAMMA]
        if WITH_2D_ARRAY:  # pragma: no cover
            x_shared = x_shared[0]

        y_1_sq = x_shared[0] ** 2 + x_shared[1] + x_1 - gamma * self.__k * y_2
        y_1 = maximum(sqrt(where(y_1_sq.real >= 0, y_1_sq, -y_1_sq)), 1e-16)
        self.store_local_data(y_1=y_1)

    def _compute_jacobian(
        self,
        inputs: Iterable[str] = (),
        outputs: Iterable[str] = (),
    ) -> None:
        x_1 = self._local_data[X_1]
        x_shared = self._local_data[X_SHARED]
        y_2 = self._local_data[Y_2]
        gamma = self._local_data[GAMMA]
        if WITH_2D_ARRAY:  # pragma: no cover
            x_shared = x_shared[0]

        y_1_sign = sign(x_shared[0] ** 2 + x_shared[1] + x_1 - gamma * self.__k * y_2)
        inv_denom = y_1_sign / self.local_data[Y_1]
        self.jac = {Y_1: {}}
        jac = self.jac[Y_1]
        jac[X_1] = diags(0.5 * inv_denom)
        jac[X_2] = self.__zeros_n
        jac[X_SHARED] = array([x_shared[0] * inv_denom, 0.5 * inv_denom]).T
        jac[Y_2] = diags(-0.5 * self.__k * gamma * inv_denom)
        jac[GAMMA] = (-0.5 * self.__k * y_2 * inv_denom).reshape((x_1.size, 1))
