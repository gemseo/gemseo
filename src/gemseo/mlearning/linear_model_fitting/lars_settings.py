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

"""Settings for the scikit-lear least angle regression (LARS) algorithm."""

from __future__ import annotations

from typing import ClassVar
from typing import Literal

from numpy import finfo
from numpy import ndarray  # noqa: TC002
from numpy.random import RandomState  # noqa: TC002
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveInt

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter_settings import (
    BaseLinearModelFitter_Settings,
)


class LARS_Settings(BaseLinearModelFitter_Settings):  # noqa: N801
    """Settings for the scikit-learn least angle regression (LARS) algorithm."""

    _TARGET_CLASS_NAME: ClassVar[str] = "LARS"

    copy_X: bool = Field(  # noqa: N815
        default=True,
        description="""If ``True``, input data will be copied;
else, it may be overwritten""",
    )

    eps: NonNegativeFloat = Field(
        default=finfo(float).eps,
        description="""The machine-precision regularization
in the computation of the Cholesky diagonal factors.
Increase this for very ill-conditioned systems.
Unlike the ``tol`` parameter in some iterative optimization-based algorithms,
this parameter does not control the tolerance of the optimization.""",
    )

    fit_path: bool = Field(
        default=False,
        description="""If ``True`` the full path is stored
in the ``coef_path_`` attribute.
If you compute the solution for a large problem or many targets,
setting ``fit_path`` to ``False`` will lead to a speedup,
especially with a small alpha.""",
    )

    jitter: float | None = Field(
        default=None,
        description="""Upper bound on a uniform noise parameter
to be added to the output values,
to satisfy the model's assumption of one-at-a-time computations.
Might help with stability.""",
    )

    n_nonzero_coefs: PositiveInt = Field(
        default=500,
        description="""Target number of non-zero coefficients.
Use ``np.inf`` for no limit.""",
    )

    precompute: Literal["auto"] | ndarray = Field(
        default="auto",
        description="""Whether to use a precomputed Gram matrix
to speed up calculations. If set to "auto" let us decide.
The Gram matrix can also be passed as argument.""",
    )

    random_state: int | RandomState | None = Field(
        default=None,
        description="""Determines random number generation for jittering.
Pass an int for reproducible output across multiple function calls.
Ignored if jitter is ``None``.""",
    )

    verbose: bool = Field(default=False, description="Sets the verbosity amount.")
