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

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import array
from numpy import diag
from numpy import maximum
from numpy import sign
from numpy import sqrt
from numpy import where
from scipy.sparse import block_diag
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

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping
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

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x_shared = input_data[X_SHARED]
        x_1 = input_data[X_1]
        y_2 = input_data[Y_2]
        gamma = input_data[GAMMA]
        if WITH_2D_ARRAY:  # pragma: no cover
            x_shared = x_shared[0]
        else:
            defaults = self.io.input_grammar.defaults
            x_shared = x_shared.reshape((-1, defaults[X_SHARED].size))
            x_1 = x_1.reshape((-1, defaults[X_1].size))
            y_2 = y_2.reshape((-1, defaults[Y_2].size))
            gamma = gamma.reshape((-1, defaults[GAMMA].size))

        y_1_sq = (
            x_shared[..., [0]] ** 2 + x_shared[..., [1]] + x_1 - gamma * self.__k * y_2
        )
        y_1 = maximum(sqrt(where(y_1_sq.real >= 0, y_1_sq, -y_1_sq)), 1e-16)
        return {"y_1": y_1.ravel()}

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        input_data = self.io.data
        x_shared = input_data[X_SHARED]
        x_1 = input_data[X_1]
        y_1 = input_data[Y_1]
        y_2 = input_data[Y_2]
        gamma = input_data[GAMMA]
        n_samples = 1
        defaults = self.io.input_grammar.defaults
        if WITH_2D_ARRAY:  # pragma: no cover
            x_shared = x_shared[0]
        else:
            x_shared = x_shared.reshape((-1, defaults[X_SHARED].size))
            x_1 = x_1.reshape((-1, defaults[X_1].size))
            y_2 = y_2.reshape((-1, defaults[Y_2].size))
            gamma = gamma.reshape((-1, defaults[GAMMA].size))
            n_samples = self._get_n_samples(x_shared, x_1, y_2, gamma)
            y_1 = y_1.reshape((n_samples, -1))

        y_1_sign = sign(
            x_shared[..., [0]] ** 2 + x_shared[..., [1]] + x_1 - gamma * self.__k * y_2
        )
        inv_denom = y_1_sign / y_1
        self.jac = {Y_1: {}}
        jac = self.jac[Y_1]
        if n_samples > 1 and not WITH_2D_ARRAY:
            jac[X_1] = block_diag([diag(0.5 * x) for x in inv_denom], format="csr")
            jac[X_2] = csr_array((n_samples * self._n, n_samples * self._n))
            jac[X_SHARED] = block_diag(
                [
                    array([x_shared_i[0] * inv_denom_i, 0.5 * inv_denom_i]).T
                    for x_shared_i, inv_denom_i in zip(
                        x_shared, inv_denom, strict=False
                    )
                ],
                format="csr",
            )
            jac[Y_2] = block_diag(
                [
                    diags(-0.5 * self.__k * gamma[min(i, defaults[GAMMA].size - 1)] * x)
                    for i, x in enumerate(inv_denom)
                ],
                format="csr",
            )
            jac[GAMMA] = block_diag(
                [
                    (-0.5 * self.__k * y_2_i * inv_denom_i).reshape((self._n, 1))
                    for y_2_i, inv_denom_i in zip(y_2, inv_denom, strict=False)
                ],
                format="csr",
            )
        else:
            inv_denom = inv_denom.ravel()
            jac[X_1] = diags(0.5 * inv_denom)
            jac[X_2] = self.__zeros_n
            jac[X_SHARED] = array([x_shared[0, 0] * inv_denom, 0.5 * inv_denom]).T
            temp = -0.5 * self.__k * inv_denom
            jac[Y_2] = diags(temp * gamma.ravel())
            jac[GAMMA] = (temp * y_2.ravel()).reshape((self._n, 1))
