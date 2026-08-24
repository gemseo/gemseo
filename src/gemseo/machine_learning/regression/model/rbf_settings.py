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
"""Settings of the RBF network for regression."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from enum import auto

from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveFloat
from pydantic import PositiveInt
from strenum import LowercaseStrEnum

from gemseo.machine_learning.regression.core.base_regressor_settings import (
    BaseRegressorSettings,
)


class RBF(LowercaseStrEnum):
    r"""The radial basis functions.

    These functions take the scaled radius $\epsilon r$ as input,
    where $r$ is a distance between two points
    and $\epsilon$ is the shape parameter defined by the setting `epsilon`.
    """

    CUBIC = auto()
    r"""The cubic RBF $(\epsilon r)^3$."""

    GAUSSIAN = auto()
    r"""The Gaussian RBF $\exp(-(\epsilon r)^2)$."""

    INVERSE_MULTIQUADRIC = auto()
    r"""The inverse multiquadric RBF $1/\sqrt{1 + (\epsilon r)^2}$."""

    INVERSE_QUADRATIC = auto()
    r"""The inverse quadratic RBF $1/(1 + (\epsilon r)^2)$."""

    LINEAR = auto()
    r"""The linear RBF $-\epsilon r$."""

    MULTIQUADRIC = auto()
    r"""The multiquadric RBF $-\sqrt{1 + (\epsilon r)^2}$."""

    QUINTIC = auto()
    r"""The quintic RBF $-(\epsilon r)^5$."""

    THIN_PLATE_SPLINE = auto()
    r"""The thin plate spline RBF $(\epsilon r)^2\log(\epsilon r)$."""


class BaseRBFRegressorSettings(BaseRegressorSettings, ABC):
    """The base class for the settings of the RBF regressors."""

    @property
    @abstractmethod
    def kernel_(self) -> RBF:
        """The radial basis function."""

    @property
    @abstractmethod
    def epsilon_(self) -> PositiveFloat | None:
        r"""The shape parameter scaling the radius, as $\epsilon r$."""

    smoothing: NonNegativeFloat = Field(
        default=0.0,
        description="""The smoothing parameter.

`0` involves an interpolation of the learning points.""",
    )

    degree: int | None = Field(
        default=None,
        ge=-1,
        description="""The degree of the added polynomial.

If `None`,
use the minimum degree required by the kernel, at least `0`.
If `-1`,
no polynomial is added.""",
    )

    neighbors: PositiveInt | None = Field(
        default=None,
        description="""The number of nearest learning points used for prediction.

If `None`, use all the learning points.
Otherwise, the model is a local interpolant,
whose coefficients are recomputed for each set of nearest learning points;
it is thus discontinuous where this set changes
and its Jacobian is not implemented.""",
    )


class RBFRegressor_Settings(BaseRBFRegressorSettings):  # noqa: N801
    """The settings of the RBF network for regression."""

    @property
    def kernel_(self) -> RBF:  # ruff: ignore[undocumented-public-method]
        return self.kernel

    @property
    def epsilon_(self) -> PositiveFloat | None:  # ruff: ignore[undocumented-public-method]
        return self.epsilon

    kernel: RBF = Field(
        default=RBF.MULTIQUADRIC, description="The radial basis function."
    )

    epsilon: PositiveFloat | None = Field(
        default=None,
        description=r"""The shape parameter scaling the radius, as $\epsilon r$.

The greater $\epsilon$, the narrower the radial basis function.
If `None`,
use $\left(\frac{1}{n}\prod_{i\in\mathcal{D}}\Delta_i\right)^{-1/d}$
for the kernels `"multiquadric"`, `"inverse_multiquadric"`,
`"inverse_quadratic"` and `"gaussian"`,
where $\mathcal{D}$ is the set of the $d$ input dimensions
whose range $\Delta_i$ is not zero
and $n$ is the number of learning points;
use `1.0` for the other kernels, which are scale-invariant,
and when the data of all the input dimensions are constant.""",
    )
