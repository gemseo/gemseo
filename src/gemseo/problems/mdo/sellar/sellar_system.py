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
"""The system discipline of the customizable Sellar MDO problem."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import array
from numpy import exp
from numpy import ones
from numpy import repeat
from scipy.sparse import block_diag
from scipy.sparse import diags
from scipy.sparse import eye

from gemseo.problems.mdo.sellar import WITH_2D_ARRAY
from gemseo.problems.mdo.sellar.base_sellar import BaseSellar
from gemseo.problems.mdo.sellar.variables import ALPHA
from gemseo.problems.mdo.sellar.variables import BETA
from gemseo.problems.mdo.sellar.variables import C_1
from gemseo.problems.mdo.sellar.variables import C_2
from gemseo.problems.mdo.sellar.variables import OBJ
from gemseo.problems.mdo.sellar.variables import X_1
from gemseo.problems.mdo.sellar.variables import X_2
from gemseo.problems.mdo.sellar.variables import X_SHARED
from gemseo.problems.mdo.sellar.variables import Y_1
from gemseo.problems.mdo.sellar.variables import Y_2

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping


class SellarSystem(BaseSellar):
    """The discipline to compute the objective and constraints of the Sellar problem."""

    _INPUT_NAMES: ClassVar[tuple[str, ...]] = (
        X_SHARED,
        X_1,
        X_2,
        Y_1,
        Y_2,
        ALPHA,
        BETA,
    )

    _OUTPUT_NAMES: ClassVar[tuple[str, ...]] = (OBJ, C_1, C_2)

    __eye_n: RealArray
    """The identity matrix of dimension n."""

    __inv_n: float
    """The inverse of the size of the local and coupling variables."""

    __inv_n_double: float
    """The double of the inverse of the size of the local and coupling variables."""

    __ones_n: RealArray
    """The one vector."""

    def __init__(self, n: int = 1) -> None:
        """
        Args:
            n: The size of the local design variables and coupling variables.
        """  # noqa: D107 D205 D205 D212 D415
        super().__init__(n)
        self.io.output_grammar.update_from_names(self._OUTPUT_NAMES)
        self.__inv_n = 1.0 / n
        self.__inv_n_double = self.__inv_n * 2.0
        self.__eye_n = eye(n)
        self.__ones_n = ones((n, 1))

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x_shared = input_data[X_SHARED]
        x_1 = input_data[X_1]
        x_2 = input_data[X_2]
        y_1 = input_data[Y_1]
        y_2 = input_data[Y_2]
        alpha = input_data[ALPHA]
        beta = input_data[BETA]
        if WITH_2D_ARRAY:  # pragma: no cover
            x_shared = x_shared[0]
        else:
            defaults = self.io.input_grammar.defaults
            x_shared = x_shared.reshape((-1, defaults[X_SHARED].size))
            x_1 = x_1.reshape((-1, defaults[X_1].size))
            x_2 = x_2.reshape((-1, defaults[X_2].size))
            y_1 = y_1.reshape((-1, defaults[Y_1].size))
            y_2 = y_2.reshape((-1, defaults[Y_2].size))
            alpha = alpha.reshape((-1, defaults[ALPHA].size))
            beta = beta.reshape((-1, defaults[BETA].size))

        obj = (
            ((x_1**2).sum(-1) + (x_2**2).sum(-1) + (y_1**2).sum(-1)) * self.__inv_n
            + x_shared[..., 1]
            + exp(-y_2.mean(-1))
        )
        return {
            "obj": obj.ravel(),
            "c_1": (alpha - y_1**2).ravel(),
            "c_2": (y_2 - beta).ravel(),
        }

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        input_data = self.io.data
        x_1 = input_data[X_1]
        x_2 = input_data[X_2]
        y_1 = input_data[Y_1]
        y_2 = input_data[Y_2]
        alpha = input_data[ALPHA]
        beta = input_data[BETA]
        n_samples = 1
        if not WITH_2D_ARRAY:  # pragma: no cover
            defaults = self.io.input_grammar.defaults
            x_1 = x_1.reshape((-1, defaults[X_1].size))
            x_2 = x_2.reshape((-1, defaults[X_2].size))
            y_1 = y_1.reshape((-1, defaults[Y_1].size))
            y_2 = y_2.reshape((-1, defaults[Y_2].size))
            alpha = alpha.reshape((-1, defaults[ALPHA].size))
            beta = beta.reshape((-1, defaults[BETA].size))
            n_samples = self._get_n_samples(x_1, x_2, y_1, y_2, alpha, beta)

        self._init_jacobian(input_names, output_names)
        if n_samples > 1 and not WITH_2D_ARRAY:
            jac = self.jac[C_1]
            ones_m_n_1 = block_diag([ones((self._n, 1))] * n_samples, format="csr")
            jac[Y_1] = block_diag([diags(-2.0 * y_1_i) for y_1_i in y_1], format="csr")
            jac[ALPHA] = ones_m_n_1
            jac = self.jac[C_2]
            jac[Y_2] = block_diag([eye(self._n)] * n_samples, format="csr")
            jac[BETA] = -ones_m_n_1
            jac = self.jac[OBJ]
            jac[X_1] = block_diag(
                [array([x * self.__inv_n_double]) for x in x_1], format="csr"
            )
            jac[X_2] = block_diag(
                [array([x * self.__inv_n_double]) for x in x_2], format="csr"
            )
            jac[X_SHARED] = block_diag([array([[0.0, 1.0]])] * n_samples, format="csr")
            jac[Y_1] = block_diag(
                [array([y * self.__inv_n_double]) for y in y_1], format="csr"
            )
            jac[Y_2] = block_diag(
                [
                    array([
                        repeat(-exp(-y.sum() * self.__inv_n) * self.__inv_n, self._n)
                    ])
                    for y in y_2
                ],
                format="csr",
            )
        else:
            jac = self.jac[C_1]
            jac[Y_1] = diags(-2.0 * y_1.ravel(), format="csr")
            jac[ALPHA] = self.__ones_n
            jac = self.jac[C_2]
            jac[Y_2] = self.__eye_n
            jac[BETA] = -self.__ones_n
            jac = self.jac[OBJ]
            jac[X_1] = array([x_1.ravel() * self.__inv_n_double])
            jac[X_2] = array([x_2.ravel() * self.__inv_n_double])
            jac[X_SHARED] = array([[0.0, 1.0]])
            jac[Y_1] = array([y_1.ravel() * self.__inv_n_double])
            exp_sum_y2 = -exp(-y_2.sum() * self.__inv_n) * self.__inv_n
            jac[Y_2] = array([repeat(exp_sum_y2, self._n)])
