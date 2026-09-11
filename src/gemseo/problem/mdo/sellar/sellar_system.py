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

from gemseo.problem.mdo.sellar import variable
from gemseo.problem.mdo.sellar import with_2d_array
from gemseo.problem.mdo.sellar.base_sellar import BaseSellar

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.util.typing import RealArray
    from gemseo.util.typing import StrKeyMapping


class SellarSystem(BaseSellar):
    """The discipline to compute the objective and constraints of the Sellar problem."""

    _input_names: ClassVar[tuple[str, ...]] = (
        variable.x_shared,
        variable.x_1,
        variable.x_2,
        variable.y_1,
        variable.y_2,
        variable.alpha,
        variable.beta,
    )

    _output_names: ClassVar[tuple[str, ...]] = (
        variable.obj,
        variable.c_1,
        variable.c_2,
    )

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
        self.io.output_grammar.update_from_names(self._output_names)
        self.__inv_n = 1.0 / n
        self.__inv_n_double = self.__inv_n * 2.0
        self.__eye_n = eye(n)
        self.__ones_n = ones((n, 1))

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x_shared = input_data[variable.x_shared]
        x_1 = input_data[variable.x_1]
        x_2 = input_data[variable.x_2]
        y_1 = input_data[variable.y_1]
        y_2 = input_data[variable.y_2]
        alpha = input_data[variable.alpha]
        beta = input_data[variable.beta]
        if with_2d_array:  # pragma: no cover
            x_shared = x_shared[0]
        else:
            defaults = self.io.input_grammar.defaults
            x_shared = x_shared.reshape((-1, defaults[variable.x_shared].size))
            x_1 = x_1.reshape((-1, defaults[variable.x_1].size))
            x_2 = x_2.reshape((-1, defaults[variable.x_2].size))
            y_1 = y_1.reshape((-1, defaults[variable.y_1].size))
            y_2 = y_2.reshape((-1, defaults[variable.y_2].size))
            alpha = alpha.reshape((-1, defaults[variable.alpha].size))
            beta = beta.reshape((-1, defaults[variable.beta].size))

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
        input_data = self.io.input_data
        x_1 = input_data[variable.x_1]
        x_2 = input_data[variable.x_2]
        y_1 = input_data[variable.y_1]
        y_2 = input_data[variable.y_2]
        alpha = input_data[variable.alpha]
        beta = input_data[variable.beta]
        n_samples = 1
        if not with_2d_array:  # pragma: no cover
            defaults = self.io.input_grammar.defaults
            x_1 = x_1.reshape((-1, defaults[variable.x_1].size))
            x_2 = x_2.reshape((-1, defaults[variable.x_2].size))
            y_1 = y_1.reshape((-1, defaults[variable.y_1].size))
            y_2 = y_2.reshape((-1, defaults[variable.y_2].size))
            alpha = alpha.reshape((-1, defaults[variable.alpha].size))
            beta = beta.reshape((-1, defaults[variable.beta].size))
            n_samples = self._get_n_samples(x_1, x_2, y_1, y_2, alpha, beta)

        self._init_jacobian(input_names, output_names)
        if n_samples > 1 and not with_2d_array:
            jac = self.jac[variable.c_1]
            ones_m_n_1 = block_diag([ones((self._n, 1))] * n_samples, format="csr")
            jac[variable.y_1] = block_diag(
                [diags(-2.0 * y_1_i) for y_1_i in y_1], format="csr"
            )
            jac[variable.alpha] = ones_m_n_1
            jac = self.jac[variable.c_2]
            jac[variable.y_2] = block_diag([eye(self._n)] * n_samples, format="csr")
            jac[variable.beta] = -ones_m_n_1
            jac = self.jac[variable.obj]
            jac[variable.x_1] = block_diag(
                [array([x * self.__inv_n_double]) for x in x_1], format="csr"
            )
            jac[variable.x_2] = block_diag(
                [array([x * self.__inv_n_double]) for x in x_2], format="csr"
            )
            jac[variable.x_shared] = block_diag(
                [array([[0.0, 1.0]])] * n_samples, format="csr"
            )
            jac[variable.y_1] = block_diag(
                [array([y * self.__inv_n_double]) for y in y_1], format="csr"
            )
            jac[variable.y_2] = block_diag(
                [
                    array([
                        repeat(-exp(-y.sum() * self.__inv_n) * self.__inv_n, self._n)
                    ])
                    for y in y_2
                ],
                format="csr",
            )
        else:
            jac = self.jac[variable.c_1]
            jac[variable.y_1] = diags(-2.0 * y_1.ravel(), format="csr")
            jac[variable.alpha] = self.__ones_n
            jac = self.jac[variable.c_2]
            jac[variable.y_2] = self.__eye_n
            jac[variable.beta] = -self.__ones_n
            jac = self.jac[variable.obj]
            jac[variable.x_1] = array([x_1.ravel() * self.__inv_n_double])
            jac[variable.x_2] = array([x_2.ravel() * self.__inv_n_double])
            jac[variable.x_shared] = array([[0.0, 1.0]])
            jac[variable.y_1] = array([y_1.ravel() * self.__inv_n_double])
            exp_sum_y2 = -exp(-y_2.sum() * self.__inv_n) * self.__inv_n
            jac[variable.y_2] = array([repeat(exp_sum_y2, self._n)])
