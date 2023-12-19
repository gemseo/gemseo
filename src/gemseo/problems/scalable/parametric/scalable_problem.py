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

from statistics import NormalDist
from typing import TYPE_CHECKING
from typing import Any

from numpy import diag
from numpy import trace
from scipy.linalg import block_diag

from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.scalable.parametric.core.scalable_problem import (
    ScalableProblem as _ScalableProblem,
)
from gemseo.problems.scalable.parametric.core.variable_names import OBJECTIVE_NAME
from gemseo.problems.scalable.parametric.core.variable_names import get_constraint_name
from gemseo.problems.scalable.parametric.disciplines.main_discipline import (
    MainDiscipline,
)
from gemseo.problems.scalable.parametric.disciplines.scalable_discipline import (
    ScalableDiscipline,
)
from gemseo.problems.scalable.parametric.scalable_design_space import (
    ScalableDesignSpace,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from gemseo.core.scenario import Scenario


class ScalableProblem(_ScalableProblem):
    r"""The scalable problem.

    It builds a set of strongly coupled scalable disciplines completed by a system
    discipline computing the objective function and the constraints.

    These disciplines are defined on a unit design space, i.e. design variables belongs
    to :math:`[0, 1]`.
    """

    _MAIN_DISCIPLINE_CLASS = MainDiscipline
    _SCALABLE_DISCIPLINE_CLASS = ScalableDiscipline
    _DESIGN_SPACE_CLASS = ScalableDesignSpace

    def create_scenario(
        self,
        use_optimizer: bool = True,
        formulation_name: str = "MDF",
        **formulation_options: Any,
    ) -> Scenario:
        """Create the DOE or MDO scenario associated with this scalable problem.

        Args:
            use_optimizer: Whether to use an optimizer or a design of experiments.
            formulation_name: The name of the formulation.
            **formulation_options: The options of the formulation.

        Returns:
            The scenario to be executed.
        """
        scenario = create_scenario(
            self.disciplines,
            formulation_name,
            OBJECTIVE_NAME,
            self.design_space,
            scenario_type="MDO" if use_optimizer else "DOE",
            **formulation_options,
        )
        for index, _ in enumerate(self.scalable_disciplines):
            scenario.add_constraint(
                get_constraint_name(index + 1),
                constraint_type=MDOFunction.FunctionType.INEQ,
            )

        return scenario

    def create_quadratic_programming_problem(
        self,
        add_coupling: bool = False,
        covariance_matrices: Iterable[NDArray[float]] = (),
        use_margin: bool = True,
        margin_factor: float = 2.0,
        tolerance: float = 0.01,
    ) -> OptimizationProblem:
        r"""Create the quadratic programming (QP) version of the MDO problem.

        This is an optimization problem
        to minimize :math:`0.5x^TQx + c^Tx + d` with respect to :math:`x`
        under the linear constraints :math:`Ax-b\leq 0`,
        where the matrix :math:`Q` is symmetric.

        In the presence of uncertainties,
        offsets are added to the objective and constraint expressions.

        Args:
            add_coupling: Whether to add the coupling variables as an observable.
            use_margin: Whether the statistics used for the constraints
                are margins or probabilities;
                if ``False``, the random vectors are assumed to be Gaussian.
            covariance_matrices: The covariance matrices
                :math:`\Sigma_1,\ldots,\Sigma_N`
                of the random variables :math:`U_1,\ldots,U_N`.
                If empty, do not consider uncertainties.
            margin_factor: The factor :math:`\kappa`
                used in the expression of the margins.
            tolerance: The tolerance level :math:`\varepsilon`
                for the violation probability.

        Returns:
            The quadratic optimization problem.
        """
        Q = self.qp_problem.Q  # noqa: N806
        c = self.qp_problem.c
        d = self.qp_problem.d
        A = self.qp_problem.A[0 : self._p, :]  # noqa: N806
        b = self.qp_problem.b[0 : self._p]

        if covariance_matrices:
            P = self._inv_C  # noqa: N806
            Sigma = block_diag(*covariance_matrices)  # noqa: N806
            P_Sigma_Pt = P @ Sigma @ P.T  # noqa: N806
            f_offset = trace(P_Sigma_Pt)
            if use_margin:
                g_offset = margin_factor * diag(P_Sigma_Pt) ** 0.5
            else:
                g_offset = -(diag(P_Sigma_Pt) ** 0.5) * NormalDist().inv_cdf(tolerance)
        else:
            g_offset = f_offset = 0

        f = lambda x: (0.5 * x @ Q @ x + c.T @ x + d + f_offset)[0]  # noqa: E731
        df = lambda x: Q @ x + c.T  # noqa: E731
        g = lambda x: A @ x - b + g_offset  # noqa: E731
        dg = lambda x: A  # noqa: E731

        design_space = DesignSpace()
        design_space.add_variable("x", size=Q.shape[0], l_b=0.0, u_b=1.0, value=0.5)

        problem = OptimizationProblem(design_space)
        problem.objective = MDOFunction(f, "f", expr="0.5x'Qx + c'x + d", jac=df)
        problem.add_constraint(
            MDOFunction(
                g, "g", f_type=MDOFunction.FunctionType.INEQ, expr="Ax-b <= 0", jac=dg
            )
        )
        if add_coupling:
            problem.add_observable(MDOFunction(self.compute_y, "coupling"))
        return problem
