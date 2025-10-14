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

"""Settings for the linear regression algorithm."""

from __future__ import annotations

from importlib.metadata import version
from typing import ClassVar

from packaging.version import parse
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveInt

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter_settings import (
    BaseLinearModelFitter_Settings,
)


class LinearRegression_Settings(BaseLinearModelFitter_Settings):  # noqa: N801
    """Settings for the scikit-learn linear regression algorithm."""

    _TARGET_CLASS_NAME: ClassVar[str] = "LinearRegression"

    copy_X: bool = Field(  # noqa: N815
        default=True,
        description="""If ``True``, input data will be copied;
else, it may be overwritten""",
    )

    n_jobs: PositiveInt | None = Field(
        default=None,
        description="""The number of jobs to use for the computation.
This will only provide speedup in case of sufficiently large problems,
that is if firstly n_targets > 1 and secondly X is sparse or if positive is set to True.
``None`` means 1 unless in a ``joblib.parallel_backend`` context.
-1 means using all processors.""",
    )

    positive: bool = Field(
        default=False,
        description="""When set to ``True``, forces the coefficients to be positive.
This option is only supported for dense arrays.""",
    )

    if parse("1.7") <= parse(version("scikit-learn")):  # pragma: no cover
        tol: NonNegativeFloat = Field(
            default=1e-6,
            description="""The precision of the solution is determined by ``tol``
which specifies a different convergence criterion for the lsqr solver.
``tol`` is set as ``atol`` and ``btol`` of ``scipy.sparse.linalg.lsqr``
when fitting on sparse training data.
This parameter has no effect when fitting on dense data.""",
        )
