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
"""The second strongly coupled discipline of the customizable Sellar MDO problem."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import ones
from numpy import repeat
from numpy import where
from scipy.sparse import block_diag
from scipy.sparse import csr_array
from scipy.sparse import eye

from gemseo.problems.mdo.sellar import WITH_2D_ARRAY
from gemseo.problems.mdo.sellar.base_sellar import BaseSellar
from gemseo.problems.mdo.sellar.variables import X_1
from gemseo.problems.mdo.sellar.variables import X_2
from gemseo.problems.mdo.sellar.variables import X_SHARED
from gemseo.problems.mdo.sellar.variables import Y_1
from gemseo.problems.mdo.sellar.variables import Y_2

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping
    from gemseo.utils.compatibility.scipy import SparseArrayType


class Sellar2(BaseSellar):
    """The discipline to compute the coupling variable :math:`y_2`."""

    _INPUT_NAMES: ClassVar[tuple[str]] = (X_2, X_SHARED, Y_1)

    _OUTPUT_NAMES: ClassVar[tuple[str]] = (Y_2,)

    __k: float
    """The shared coefficient controlling the coupling strength."""

    __eye_n: RealArray
    """The identity matrix of dimension n."""

    __k_eye_n: RealArray
    """The identity matrix of dimension n multiplied by the coefficient k."""

    __ones_n: RealArray
    """The (1x2) matrix of dimension n."""

    __zeros_n: SparseArrayType
    """The zero matrix of dimension n."""

    def __init__(self, n: int = 1, k: float = 1.0) -> None:
        """
        Args:
            k: The shared coefficient controlling the coupling strength
        """  # noqa: D107 D205 D205 D212 D415
        super().__init__(n)
        self.__k = k
        self.__eye_n = eye(n)
        self.__ones_n = ones((n, 2))
        self.__k_eye_n = k * self.__eye_n
        self.__zeros_n = csr_array((n, n))

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x_shared = input_data[X_SHARED]
        x_2 = input_data[X_2]
        y_1 = input_data[Y_1]
        if WITH_2D_ARRAY:  # pragma: no cover
            x_shared = x_shared[0]
        else:
            defaults = self.io.input_grammar.defaults
            x_shared = x_shared.reshape((-1, defaults[X_SHARED].size))
            x_2 = x_2.reshape((-1, defaults[X_2].size))
            y_1 = y_1.reshape((-1, defaults[Y_1].size))

        out = x_shared[..., [0]] + x_shared[..., [1]] - x_2
        if out.shape != y_1.shape:
            out = repeat(out, len(y_1), axis=0)

        # Implement y_2 = |k*y_1| + x_00 + x_01 - x2 in the case of complex numbers.
        y_2 = where(
            y_1.real > 0,
            self.__k * y_1 + out,
            # The complex step technique expects "out" at y_1=0+aj where a is a real.
            # This is the reason of this second check:
            # -> neither "self.__k * y_1 + out" nor "-self.__k * y_1 + out" can be used,
            # even if they are equal to "out" when considering real numbers.
            where(y_1.real < 0, -self.__k * y_1 + out, out),
        )
        return {Y_2: y_2.ravel()}

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        input_data = self.io.data
        x_shared = input_data[X_SHARED]
        x_2 = input_data[X_2]
        y_1 = input_data[Y_1]
        n_samples = 1
        if not WITH_2D_ARRAY:  # pragma: no branch
            defaults = self.io.input_grammar.defaults
            x_shared = x_shared.reshape((-1, defaults[X_SHARED].size))
            x_2 = x_2.reshape((-1, defaults[X_2].size))
            y_1 = y_1.reshape((-1, defaults[Y_1].size))
            n_samples = self._get_n_samples(x_shared, x_2, y_1)

        self.jac = {Y_2: {}}
        jac = self.jac[Y_2]
        if n_samples > 1 and not WITH_2D_ARRAY:
            jac[X_1] = csr_array((n_samples * self._n, n_samples * self._n))
            jac[X_2] = -block_diag([eye(self._n)] * n_samples, format="csr")
            jac[X_SHARED] = block_diag([ones((self._n, 2))] * n_samples, format="csr")
            matrices = []
            for y_1_i in y_1:
                matrix = self.__k * eye(self._n).tocsr()
                matrix[y_1_i.real < 0] *= -1.0
                matrix[y_1_i.real == 0] = 0.0
                matrices.append(matrix)

            self.jac[Y_2][Y_1] = block_diag(matrices, format="csr")
        else:
            jac[X_1] = self.__zeros_n
            jac[X_2] = -self.__eye_n
            jac[X_SHARED] = self.__ones_n
            dy_2_dy_1 = self.__k_eye_n.tocsr().copy()
            y_1_real = y_1.real.ravel()
            inds_negative = y_1_real < 0
            dy_2_dy_1[inds_negative] *= -1.0
            dy_2_dy_1[y_1_real == 0] = 0.0
            self.jac[Y_2][Y_1] = dy_2_dy_1
