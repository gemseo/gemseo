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

"""Settings for the scikit-learn ridge algorithm."""

from __future__ import annotations

from typing import ClassVar

from numpy import ndarray  # noqa: TC002
from numpy.random import RandomState  # noqa: TC002
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveInt
from strenum import StrEnum

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter_settings import (
    BaseLinearModelFitter_Settings,
)


class Solver(StrEnum):
    """A solver type."""

    AUTO = "auto"
    SVD = "svd"
    CHOLESKY = "cholesky"
    LSQR = "lsqr"
    SPARSE_CG = "sparse_cg"
    SAG = "sag"
    SAGE = "saga"
    LBFGS = "lbfgs"


class Ridge_Settings(BaseLinearModelFitter_Settings):  # noqa: N801
    """Settings for the scikit-learn ridge algorithm."""

    _TARGET_CLASS_NAME: ClassVar[str] = "Ridge"

    alpha: NonNegativeFloat | ndarray = Field(
        default=1.0,
        description=r"""The constant :math:`\alpha` that multiplies the L2 term,
controlling regularization strength.
If an array is passed, penalties are assumed to be specific to the targets.""",
    )

    copy_X: bool = Field(  # noqa: N815
        default=True,
        description="""If ``True``, input data will be copied;
else, it may be overwritten""",
    )

    max_iter: PositiveInt | None = Field(
        default=None,
        description="""The maximum number of iterations for conjugate gradient solver.
For "sparse_cg" and "lsqr" solvers,
the default value is determined by ``scipy.sparse.linalg``.
For "sag" solver, the default value is 1000.
For "lbfgs" solver, the default value is 15000.""",
    )

    positive: bool = Field(
        default=False,
        description="""When set to ``True``, forces the coefficients to be positive.
Only "lbfgs" solver is supported in this case.""",
    )

    random_state: int | RandomState | None = Field(
        default=None,
        description="Used when ``solver == 'sag'`` or ``'saga'`` to shuffle the data.",
    )

    solver: Solver = Field(
        default=Solver.AUTO,
        description="""The solver to use in the computational routines.
If ``"auto"``, the solver is automatically chosen based on the type of data.""",
    )

    tol: NonNegativeFloat = Field(
        default=1e-4,
        description="""The precision of the solution is determined by ``tol``
which specifies a different convergence criterion for each solver:

- "svd": ``tol`` has no impact.
- "cholesky": ``tol`` has no impact.
- "sparse_cg": norm of residuals smaller than ``tol``.
- "lsqr": ``tol`` is set as ``atol`` and ``btol`` of scipy.sparse.linalg.lsqr``,
  which control the norm of the residual vector
  in terms of the norms of matrix and coefficients.
- "sag" and "saga": relative change of coef smaller than ``tol``.
- "lbfgs": maximum of the absolute (projected) ``gradient=max|residuals|``
  smaller than tol.""",
    )
