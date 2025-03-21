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

from collections.abc import Sequence  # noqa: TC003
from typing import Final
from typing import Union

from openturns import TNC as OT_TNC
from openturns import CovarianceModelImplementation
from openturns import OptimizationAlgorithmImplementation
from pydantic import Field
from pydantic import NonNegativeInt
from strenum import StrEnum

from gemseo.algos.design_space import DesignSpace  # noqa: TC001
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.mlearning.regression.algos.base_regressor_settings import (
    BaseRegressorSettings,
)
from gemseo.typing import StrKeyMapping  # noqa: TC001

DOEAlgorithmName = StrEnum("DOEAlgorithmName", DOELibraryFactory().algorithms)
"""The name of a DOE algorithm."""


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


CovarianceModelType = Union[
    CovarianceModelImplementation,
    type[CovarianceModelImplementation],
    CovarianceModel,
]

TNC: Final[TNC] = OT_TNC()
"""The TNC algorithm."""


class OTGaussianProcessRegressor_Settings(BaseRegressorSettings):  # noqa: N801
    """The settings of the Gaussian process regressor from OpenTURNS."""

    use_hmat: bool | None = Field(
        default=None,
        description="""Whether to use the HMAT or LAPACK as linear algebra method.

If ``None``,
use HMAT when the learning size is greater than
:attr:`~.OTGaussianProcessRegressor.MAX_SIZE_FOR_LAPACK`.""",
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
If ``None``,
the algorithm will use a design space with bounds defined by OpenTURNS.""",
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

    multi_start_n_samples: NonNegativeInt = Field(
        default=10,
        description="""The number of starting points of the multi-start optimizer.

This optimizer is used for the covariance model parameters.""",
    )

    multi_start_algo_name: DOEAlgorithmName = Field(
        default=DOEAlgorithmName.OT_OPT_LHS,
        description="""The name of the DOE algorithm.

This DOE is used for the multi-start optimization
of the covariance model parameters.""",
    )

    multi_start_algo_settings: StrKeyMapping = Field(
        default_factory=dict, description="The settings of the DOE algorithm."
    )
