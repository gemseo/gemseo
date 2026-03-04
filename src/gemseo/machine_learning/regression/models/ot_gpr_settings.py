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
"""Settings of the Gaussian process regressor from OpenTURNS."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Final

from openturns import TNC as OT_TNC
from openturns import CovarianceModelImplementation
from openturns import OptimizationAlgorithmImplementation
from pydantic import Field
from pydantic import model_validator
from strenum import StrEnum

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.base_doe_settings import BaseDOESettings
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.machine_learning.regression.models.base_regressor_settings import (
    BaseRegressorSettings,
)

if TYPE_CHECKING:
    from typing_extensions import Self


class Trend(StrEnum):
    """The name of a trend."""

    CONSTANT = "constant"
    LINEAR = "linear"
    QUADRATIC = "quadratic"


class CovarianceModel(StrEnum):
    """The name of a covariance model."""

    ABSOLUTE_EXPONENTIAL = "AbsoluteExponential"
    """The absolute exponential kernel."""

    EXPONENTIAL = "Exponential"
    """The exponential kernel."""

    MATERN12 = "Matern12"
    """The Matérn 1/2 kernel."""

    MATERN32 = "Matern32"
    """The Matérn 3/2 kernel."""

    MATERN52 = "Matern52"
    """The Matérn 5/2 kernel."""

    SQUARED_EXPONENTIAL = "SquaredExponential"
    """The squared exponential kernel."""


CovarianceModelType = (
    CovarianceModelImplementation
    | type[CovarianceModelImplementation]
    | CovarianceModel
)

TNC: Final[TNC] = OT_TNC()
"""The TNC algorithm."""


class OTGaussianProcessRegressor_Settings(BaseRegressorSettings):  # noqa: N801
    """The settings of the Gaussian process regressor from OpenTURNS."""

    use_hmat: bool | None = Field(
        default=None,
        description="""Whether to use the HMAT or LAPACK as linear algebra method.

If `None`,
use HMAT when the learning size is greater than
[OTGaussianProcessRegressor.MAX_SIZE_FOR_LAPACK][gemseo.machine_learning.regression.models.ot_gpr.OTGaussianProcessRegressor.MAX_SIZE_FOR_LAPACK].""",
    )

    trend: Trend = Field(default=Trend.CONSTANT, description="The name of the trend.")

    optimizer: OptimizationAlgorithmImplementation = Field(
        default=TNC,
        description="The solver used to optimize the covariance model parameters.",
    )

    optimization_space: DesignSpace | None = Field(
        default=None,
        description="""The covariance model parameter space.

The size of a variable must take into account the size of the output space.
If `None`,
the model will use a design space with bounds defined by OpenTURNS.""",
    )

    covariance_model: Sequence[CovarianceModelType] | CovarianceModelType = Field(
        default=CovarianceModel.MATERN52,
        description="""The covariance model of the Gaussian process.

Either an OpenTURNS covariance model class,
an OpenTURNS covariance model class instance,
a name of covariance model,
or a list of OpenTURNS covariance model classes,
OpenTURNS class instances and covariance model names,
whose size is equal to the output dimension.""",
    )

    multi_start_algo_settings: BaseDOESettings | None = Field(
        default_factory=OT_OPT_LHS_Settings,
        description="""The settings of the DOE algorithm.

This DOE algorithm is used for the multi-start optimization
of the covariance model parameters.

The number of samples corresponds to the number of starting points.
If not set explicitly, its value will be 10.
If `None`, do not use multi-start optimization.
""",
    )

    @model_validator(mode="after")
    def __validate_n_samples(self) -> Self:
        """Validate the number of starting points."""
        if (
            self.multi_start_algo_settings is not None
            and "n_samples" not in self.multi_start_algo_settings.model_fields_set
        ):
            self.multi_start_algo_settings.n_samples = 10

        return self
