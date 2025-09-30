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

"""Settings for the SPGL1 (Spectral Projected Gradient for L1 minimization) algorithm."""  # noqa: E501

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar
from typing import Literal

from numpy import inf
from numpy import ndarray  # noqa: TC002
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveFloat
from pydantic import PositiveInt
from spgl1.spgl1 import _norm_l1_dual
from spgl1.spgl1 import _norm_l1_primal
from spgl1.spgl1 import _norm_l1_project

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter_settings import (
    BaseLinearModelFitter_Settings,
)


class SPGL1_Settings(BaseLinearModelFitter_Settings):  # noqa: N801
    """Settings for the SPGL1 (Spectral Projected Gradient for L1 minimization) algorithm."""  # noqa: E501

    _TARGET_CLASS_NAME: ClassVar[str] = "SPGL1"

    tau: NonNegativeFloat = Field(
        default=0.0,
        description="""The Lasso threshold.
If ``0`` and ``sigma`` is ``0``, spgl1 solves a BP problem.
If different from ``0``, spgl1 solves a Lasso problem.
``tau`` and ``sigma`` cannot both be positive.""",
    )

    sigma: NonNegativeFloat = Field(
        default=0.0,
        description="""The BPDN threshold.
If ``0`` and ``sigma`` is ``0``, spgl1 solves a BP problem.
If different from ``0``, spgl1 solves a BPDN problem.
``tau`` and ``sigma`` cannot both be positive.""",
    )

    x0: ndarray | None = Field(
        default=None, description="The initial guess of x; if None zeros are used."
    )

    # fid : file, optional
    #     File ID to direct log output, if None print on screen.

    verbosity: Literal[0, 1, 2] = Field(
        default=0,
        description="The verbosity level: 0=quiet, 1=some output, 2=more output.",
    )

    iter_lim: PositiveInt | None = Field(
        default=None,
        description="The maximum number of iterations (default if ``10*m``).",
    )

    n_prev_vals: PositiveInt = Field(
        default=3,
        description="The line-search history length.",
    )

    bp_tol: PositiveFloat = Field(
        default=1e-6,
        description="The tolerance for identifying a basis pursuit solution.",
    )

    ls_tol: PositiveFloat = Field(
        default=1e-6,
        description="""The tolerance for least-squares solution.
Iterations are stopped when the ratio
between the dual norm of the gradient and the L2 norm of the residual
becomes smaller or equal to ``ls_tol``.""",
    )

    opt_tol: PositiveFloat = Field(
        default=1e-4,
        description="""The optimality tolerance.
More specifically,
when using basis pursuit denoise,
the optimality condition is met when the absolute difference
between the L2 norm of the residual and the ``sigma`` is smaller than``opt_tol``.""",
    )

    dec_tol: PositiveFloat = Field(
        default=1e-4,
        description="""The required relative change in primal objective for Newton.
Larger ``decTol`` means more frequent Newton updates.""",
    )

    step_min: PositiveFloat = Field(
        default=1e-16,
        description="The minimum spectral step.",
    )

    step_max: PositiveFloat = Field(
        default=1e5,
        description="The maximum spectral step.",
    )

    active_set_niters: PositiveInt | Literal[inf] = Field(
        default=inf,
        description="""The maximum number of iterations
where no change in support is tolerated.
Exit with EXIT_ACTIVE_SET if no change is observed for ``activeSetIt`` iterations""",
    )

    subspace_min: bool = Field(
        default=False,
        description="Subspace minimization.",
    )

    iscomplex: bool = Field(
        default=False,
        description="Whether the problem has complex variables.",
    )

    max_matvec: PositiveInt | Literal[inf] = Field(
        default=inf,
        description="The maximum matrix-vector multiplies allowed.",
    )

    weights: float | ndarray | None = Field(
        default=None,
        description="The weights ``W`` in ``||Wx||_1``. If ``None``, use 1.",
    )

    project: Callable[[ndarray, float | ndarray, float], ndarray] = Field(
        default=_norm_l1_project, description="The projection function."
    )

    primal_norm: Callable[[ndarray, float | ndarray], float] = Field(
        default=_norm_l1_primal, description="The primal norm evaluation function."
    )

    dual_norm: Callable[[ndarray, float | ndarray], float] = Field(
        default=_norm_l1_dual, description="The primal norm evaluation function."
    )
