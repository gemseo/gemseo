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

from collections.abc import Iterable
from typing import ClassVar

from numpy import array
from numpy import exp
from numpy import ones
from numpy import repeat
from numpy import sum as np_sum
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
from gemseo.typing import RealArray


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

    __n: float
    """The size of the local and coupling variables."""

    __ones_n: RealArray
    """The one vector."""

    def __init__(self, n: int = 1) -> None:
        """
        Args:
            n: The size of the local design variables and coupling variables.
        """  # noqa: D107 D205 D205 D212 D415
        super().__init__(n)
        self.output_grammar.update_from_names(self._OUTPUT_NAMES)
        self.re_exec_policy = self.ReExecutionPolicy.DONE
        self.__n = n
        self.__inv_n = 1.0 / n
        self.__inv_n_double = self.__inv_n * 2.0
        self.__eye_n = eye(n)
        self.__ones_n = ones((n, 1))

    def _run(self) -> None:
        x_1 = self._local_data[X_1]
        x_2 = self._local_data[X_2]
        x_shared = self._local_data[X_SHARED]
        y_1 = self._local_data[Y_1]
        y_2 = self._local_data[Y_2]
        alpha = self._local_data[ALPHA]
        beta = self._local_data[BETA]
        if WITH_2D_ARRAY:  # pragma: no cover
            x_shared = x_shared[0]

        obj = array([
            (x_1.dot(x_1) + x_2.dot(x_2) + y_1.dot(y_1)) * self.__inv_n
            + x_shared[1]
            + exp(-y_2.mean())
        ])
        self.store_local_data(
            obj=obj,
            c_1=alpha - y_1**2,
            c_2=y_2 - beta,
        )

    def _compute_jacobian(
        self,
        inputs: Iterable[str] = (),
        outputs: Iterable[str] = (),
    ) -> None:
        self._init_jacobian(inputs, outputs)
        x_1 = self._local_data[X_1]
        x_2 = self._local_data[X_2]
        y_1 = self._local_data[Y_1]
        y_2 = self._local_data[Y_2]
        jac = self.jac[C_1]
        jac[Y_1] = diags(-2.0 * y_1)
        jac[ALPHA] = self.__ones_n
        jac = self.jac[C_2]
        jac[Y_2] = self.__eye_n
        jac[BETA] = -self.__ones_n
        jac = self.jac[OBJ]
        jac[X_1] = array([x_1 * self.__inv_n_double])
        jac[X_2] = array([x_2 * self.__inv_n_double])
        jac[X_SHARED] = array([[0.0, 1.0]])
        jac[Y_1] = array([y_1 * self.__inv_n_double])
        exp_sum_y2 = -exp(-np_sum(y_2) * self.__inv_n) * self.__inv_n
        jac[Y_2] = array([repeat(exp_sum_y2, self.__n)])
