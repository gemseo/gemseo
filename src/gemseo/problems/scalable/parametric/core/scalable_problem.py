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
"""The scalable problem."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from numpy import concatenate
from numpy import eye
from numpy import full
from numpy import newaxis
from numpy import ones
from numpy import quantile
from numpy import vstack
from numpy import zeros
from numpy.linalg import inv
from numpy.random import default_rng

from gemseo import SEED
from gemseo.problems.scalable.parametric.core.default_settings import DEFAULT_D_0
from gemseo.problems.scalable.parametric.core.disciplines.main_discipline import (
    MainDiscipline,
)
from gemseo.problems.scalable.parametric.core.disciplines.scalable_discipline import (
    ScalableDiscipline,
)
from gemseo.problems.scalable.parametric.core.quadratic_programming_problem import (
    QuadraticProgrammingProblem,
)
from gemseo.problems.scalable.parametric.core.scalable_design_space import (
    ScalableDesignSpace,
)
from gemseo.problems.scalable.parametric.core.scalable_discipline_settings import (
    DEFAULT_SCALABLE_DISCIPLINE_SETTINGS,
)
from gemseo.problems.scalable.parametric.core.scalable_discipline_settings import (
    ScalableDisciplineSettings,
)
from gemseo.problems.scalable.parametric.core.variable_names import (
    SHARED_DESIGN_VARIABLE_NAME,
)
from gemseo.problems.scalable.parametric.core.variable_names import get_constraint_name
from gemseo.problems.scalable.parametric.core.variable_names import get_coupling_name
from gemseo.problems.scalable.parametric.core.variable_names import get_u_local_name
from gemseo.problems.scalable.parametric.core.variable_names import get_x_local_name
from gemseo.utils.string_tools import MultiLineString

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class ScalableProblem:
    r"""The scalable problem.

    It builds a set of strongly coupled scalable disciplines completed by a system
    discipline computing the objective function and the constraints.

    These disciplines are defined on a unit design space, i.e. design variables belongs
    to :math:`[0, 1]`.
    """

    _MAIN_DISCIPLINE_CLASS: ClassVar[type] = MainDiscipline
    """The class of the main discipline."""

    _SCALABLE_DISCIPLINE_CLASS: ClassVar[type] = ScalableDiscipline
    """The class of the scalable discipline."""

    _DESIGN_SPACE_CLASS: ClassVar[type] = ScalableDesignSpace
    """The class of the design space."""

    disciplines: list[_MAIN_DISCIPLINE_CLASS | _SCALABLE_DISCIPLINE_CLASS]
    """The disciplines."""

    design_space: _DESIGN_SPACE_CLASS
    """The design space."""

    qp_problem: QuadraticProgrammingProblem
    """The quadratic programming problem."""

    __N_SAMPLES: Final[int] = 100000
    """The number of samples to estimate the quantile-based constraint threshold."""

    __alpha: NDArray[float]
    r"""The matrix :math:`\alpha` to compute :math:`y=\alpha+\beta x."""

    __beta: NDArray[float]
    r"""The matrix :math:`\beta` to compute :math:`y=\alpha+\beta x."""

    def __init__(
        self,
        discipline_settings: Sequence[
            ScalableDisciplineSettings
        ] = DEFAULT_SCALABLE_DISCIPLINE_SETTINGS,
        d_0: int = DEFAULT_D_0,
        add_random_variables: bool = False,
        alpha: float = 0.50,
        seed: int = SEED,
    ) -> None:
        r"""
        Args:
            discipline_settings: The configurations
                of the different scalable disciplines.
            d_0: The size of the shared design variable :math:`x_0`.
            add_random_variables: Whether to add a centered random variable :math:`u_i`
                on the output of the :math:`i`-th scalable discipline.
            alpha: The proportion of feasible design points.
            seed: The seed for reproducibility.
        """  # noqa: D205 D212
        rng = default_rng(seed)

        # The output sizes of the scalable disciplines.
        p_i = [d.p_i for d in discipline_settings]
        self._p = sum(p_i)

        # The input sizes of the scalable disciplines.
        d_i = [d.d_i for d in discipline_settings]
        d = sum(d_i) + d_0

        # Generate realizations of the random matrices.
        # - a_i  ~ U{[0,1], (p_i,)}
        # - D_i0 ~ U{[0,1], (p_i, d_0)}
        # - D_ii ~ U{[0,1], (p_i, d_i)}
        # - C_ij ~ U{[0,1], (p_i, p_j)}
        a_i = []
        D_i0 = []  # noqa:N806
        D_ii = []  # noqa:N806
        C_ij = []  # noqa:N806
        N = len(discipline_settings)  # noqa:N806
        all_discipline_indices = set(range(N))
        for i, i_th_disc_settings in enumerate(discipline_settings):
            other_discipline_indices = all_discipline_indices - {i}
            _p_i = i_th_disc_settings.p_i
            D_i0.append(rng.random((_p_i, d_0)))
            D_ii.append(rng.random((_p_i, i_th_disc_settings.d_i)))
            C_ij.append({
                get_coupling_name(j + 1): rng.random((_p_i, discipline_settings[j].p_i))
                for j in other_discipline_indices
            })
            a_i.append(rng.random(_p_i))

        # Define the matrix C and compute its inverse.
        C = eye(self._p)  # noqa: N806
        row_start = 0
        for i, _p_i in enumerate(p_i):
            C_ij_i = C_ij[i]  # noqa: N806
            col_start = 0
            row_end = row_start + _p_i
            for j, _p_j in enumerate(p_i):
                col_end = col_start + _p_j
                if j != i:
                    name = get_coupling_name(j + 1)
                    C[row_start:row_end, col_start:col_end] = -C_ij_i[name]

                col_start = col_end

            row_start = row_end

        self._inv_C = inv(C)  # noqa: N806

        # Define the matrices \alpha and \beta.
        D = zeros((self._p, d))  # noqa: N806
        row_start = 0
        col_start = d_0
        for i, _p_i in enumerate(p_i):
            row_end = row_start + _p_i
            col_end = col_start + d_i[i]
            D[row_start:row_end, 0:d_0] = D_i0[i]
            D[row_start:row_end, col_start:col_end] = D_ii[i]
            row_start = row_end
            col_start = col_end

        self.__alpha = self._inv_C @ concatenate(a_i)
        self.__beta = -self._inv_C @ D

        q = quantile(
            [
                self.compute_y(x).min()
                for x in rng.random((self.__N_SAMPLES, sum(d_i) + d_0))
            ],
            1 - alpha,
        )
        t_i = [[q] * n for n in p_i]

        # Define the QP problem with the matrices
        # Q = 2(Q_{x_0}+\beta^T\beta)
        # c = 2*\beta^T\alpha
        # d = \alpha^T\alpha
        # b = [-\beta,\diag(d),-\diag(d)]
        Q_x0 = zeros((d, d))  # noqa: N806
        Q_x0[0:d_0, 0:d_0] = eye(d_0)
        self.qp_problem = QuadraticProgrammingProblem(
            2 * (Q_x0 + self.__beta.T @ self.__beta),
            (2 * self.__beta.T @ self.__alpha)[:, newaxis],
            self.__alpha.T @ self.__alpha,
            vstack([-self.__beta, eye(d), -eye(d)]),
            concatenate([self.__alpha - concatenate(t_i), ones(d), zeros(d)]),
        )

        # Define the default values of the input variables.
        default_input_values = {SHARED_DESIGN_VARIABLE_NAME: zeros(d_0) + 0.5}
        for index, _discipline_settings in enumerate(discipline_settings):
            discipline_index = index + 1
            d_i = _discipline_settings.d_i
            p_i = _discipline_settings.p_i
            default_input_values[get_x_local_name(discipline_index)] = full(d_i, 0.5)
            default_input_values[get_coupling_name(discipline_index)] = full(p_i, 0.5)
            default_input_values[get_constraint_name(discipline_index)] = full(p_i, 0.5)
            default_input_values[get_u_local_name(discipline_index)] = zeros(p_i)

        # Instantiate the main discipline
        names = [SHARED_DESIGN_VARIABLE_NAME]
        names.extend([get_coupling_name(index) for index in range(1, N + 1)])
        self.disciplines = [
            self._MAIN_DISCIPLINE_CLASS(
                *t_i,
                **{k: v.copy() for k, v in default_input_values.items() if k in names},
            )
        ]

        # Instantiate the scalable disciplines
        for discipline_index in range(1, N + 1):
            names = [SHARED_DESIGN_VARIABLE_NAME, get_x_local_name(discipline_index)]
            names.extend([
                get_coupling_name(other_discipline_index)
                for other_discipline_index in range(1, N + 1)
                if other_discipline_index != discipline_index
            ])
            if add_random_variables:
                names.append(get_u_local_name(discipline_index))

            self.disciplines.append(
                self._SCALABLE_DISCIPLINE_CLASS(
                    discipline_index,
                    a_i[discipline_index - 1],
                    D_i0[discipline_index - 1],
                    D_ii[discipline_index - 1],
                    C_ij[discipline_index - 1],
                    **{
                        k: v.copy()
                        for k, v in default_input_values.items()
                        if k in names
                    },
                )
            )

        self.design_space = self._DESIGN_SPACE_CLASS(discipline_settings, d_0)

    def compute_y(
        self, x: NDArray[float], u: NDArray[float] | None = None
    ) -> NDArray[float]:
        r"""Compute the coupling vector :math:`y`.

        Args:
            x: A design point.
            u: An uncertain point, if any.

        Returns:
            The coupling vector associated with the design point :math:`x`
            and the uncertain vector :math:`U` if any.
        """
        y = self.__alpha + self.__beta @ x
        if u is not None:
            y += self._inv_C @ u
        return y

    @property
    def main_discipline(self) -> _MAIN_DISCIPLINE_CLASS:
        """The main discipline."""
        return self.disciplines[0]

    @property
    def scalable_disciplines(self) -> list[_SCALABLE_DISCIPLINE_CLASS]:
        """The scalable disciplines."""
        return self.disciplines[1:]

    @property
    def __string_representation(self) -> MultiLineString:
        """The string representation of the scalable problem."""
        mls = MultiLineString()
        mls.add("Scalable problem")
        mls.indent()
        for discipline in self.disciplines:
            mls.add(discipline.name)
            mls.indent()
            mls.add("Inputs")
            mls.indent()
            for name in discipline.input_names:
                mls.add(f"{name} ({discipline.names_to_sizes[name]})")

            mls.dedent()
            mls.add("Outputs")
            mls.indent()
            for name in discipline.output_names:
                mls.add(f"{name} ({discipline.names_to_sizes[name]})")

            mls.dedent()
            mls.dedent()

        return mls

    def __repr__(self) -> str:
        return str(self.__string_representation)

    def _repr_html_(self) -> str:
        return self.__string_representation._repr_html_()
