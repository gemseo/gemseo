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
"""Compatibility between different versions of openturns."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING
from typing import Final

from packaging.version import parse as parse_version

if TYPE_CHECKING:
    from openturns import CovarianceMatrix
    from openturns import CovarianceModel
    from openturns import Function
    from openturns import Point
    from openturns import Sample
    from packaging.version import Version

ot_version: Final[Version] = parse_version(version("openturns"))

ot_1_27: Final[Version] = parse_version("1.27")

if ot_version >= ot_1_27:
    from openturns import GaussianProcessConditionalCovariance
    from openturns import GaussianProcessFitter
    from openturns import GaussianProcessRegression
    from openturns import GaussianProcessRegressionResult as OTGPRResult

    gpr_algo_class = GaussianProcessFitter
    """The OpenTURNS class fitting the covariance model of a Gaussian process."""

    gpr_conditional_covariance_class = GaussianProcessConditionalCovariance
    """The OpenTURNS class owning `getConditionalCovariance`."""

    linear_algebra_resource_key = "GaussianProcessFitter-LinearAlgebra"
    """The `ResourceMap` key setting the linear algebra method of the GP fitter."""

    class GaussianProcessRegressionResult:
        """The result of the Gaussian process regression (GPR).

        The API of OpenTURNS 1.27 splits the conditional mean and covariance off the
        result object onto a separate `GaussianProcessConditionalCovariance` object.
        This adapter recombines them
        so that the rest of GEMSEO can keep calling the same methods
        (`getConditionalMean`, `getConditionalCovariance`, `getMetaModel`,
        `getCovarianceModel` and `getTrendCoefficients`)
        regardless of the OpenTURNS version.
        """

        result: OTGPRResult
        """The Gaussian process regression result from OpenTURNS."""

        __conditional: GaussianProcessConditionalCovariance
        """The conditional covariance post-processing."""

        def __init__(self, regression_result: OTGPRResult) -> None:
            """
            Args:
                regression_result: The Gaussian process regression result
                    from OpenTURNS.
            """  # noqa: D205, D212
            self.result = regression_result
            self.__conditional = GaussianProcessConditionalCovariance(regression_result)

        def getConditionalMean(self, input_data: Sample) -> Sample:  # noqa: N802
            """Compute the conditional mean.

            Args:
                input_data: The input point.

            Returns:
                The conditional mean at the input point.
            """
            return self.__conditional.getConditionalMean(input_data)

        def getConditionalCovariance(  # noqa: N802
            self, input_data: Sample
        ) -> CovarianceMatrix:
            """Compute the conditional covariance.

            Args:
                input_data: The input point.

            Returns:
                The conditional covariance at the input point.
            """
            return self.__conditional.getConditionalCovariance(input_data)

        def getMetaModel(self) -> Function:  # noqa: N802
            """Get the metamodel.

            Returns:
                The metamodel.
            """
            return self.result.getMetaModel()

        def getCovarianceModel(self) -> CovarianceModel:  # noqa: N802
            """Get the covariance model.

            Returns:
                The covariance model.
            """
            return self.result.getCovarianceModel()

        def getTrendCoefficients(self) -> Point:  # noqa: N802
            """Get the coefficients of the trend.

            Returns:
                The coefficients of the trend.
            """
            return self.result.getTrendCoefficients()

    def build_gpr_result(  # noqa: D103
        algo: GaussianProcessFitter,
    ) -> GaussianProcessRegressionResult:
        regression = GaussianProcessRegression(algo.getResult())
        regression.run()
        return GaussianProcessRegressionResult(regression.getResult())

else:
    from openturns import KrigingAlgorithm
    from openturns import KrigingResult

    gpr_algo_class = KrigingAlgorithm
    """The OpenTURNS class fitting the covariance model of a Gaussian process."""

    gpr_conditional_covariance_class = KrigingResult
    """The OpenTURNS class owning `getConditionalCovariance`."""

    linear_algebra_resource_key = "KrigingAlgorithm-LinearAlgebra"
    """The `ResourceMap` key setting the linear algebra method of the GP fitter."""

    def build_gpr_result(algo: KrigingAlgorithm) -> KrigingResult:  # noqa: D103
        return algo.getResult()
