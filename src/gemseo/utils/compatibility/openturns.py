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

import openturns
from openturns import AggregatedFunction
from openturns import Basis
from openturns import BasisFactory
from packaging.version import Version
from packaging.version import parse as parse_version

if TYPE_CHECKING:
    from numpy import ndarray

OT_VERSION: Final[Version] = parse_version(version("openturns"))

OT_1_23: Final[Version] = parse_version("1.21")
IS_OT_LOWER_THAN_1_20: Final[bool] = parse_version("1.20") > OT_VERSION

if parse_version("1.21") > OT_VERSION:  # pragma: no cover

    def create_trend_basis(  # noqa: D103
        basis_factory: type(BasisFactory), input_dimension: int, output_dimension: int
    ) -> Basis:
        return basis_factory(input_dimension).build()

else:

    def create_trend_basis(  # noqa: D103
        basis_factory: type(BasisFactory), input_dimension: int, output_dimension: int
    ) -> Basis:
        basis = basis_factory(input_dimension).build()
        return Basis([
            AggregatedFunction([basis.build(k)] * output_dimension)
            for k in range(basis.getSize())
        ])


if not IS_OT_LOWER_THAN_1_20:

    def compute_pcc(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis(x, y).computePCC()

    def compute_prcc(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis(x, y).computePRCC()

    if OT_VERSION > OT_1_23:

        def compute_pearson_correlation(  # noqa: D103
            x: ndarray, y: ndarray
        ) -> openturns.Point:
            return openturns.CorrelationAnalysis(x, y).computeLinearCorrelation()

    else:

        def compute_pearson_correlation(  # noqa: D103
            x: ndarray, y: ndarray
        ) -> openturns.Point:
            return openturns.CorrelationAnalysis(x, y).computePearsonCorrelation()

    def compute_spearman_correlation(  # noqa: D103
        x: ndarray, y: ndarray
    ) -> openturns.Point:
        return openturns.CorrelationAnalysis(x, y).computeSpearmanCorrelation()

    def compute_src(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis(x, y).computeSRC()

    def compute_srrc(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis(x, y).computeSRRC()

    def compute_kendall_tau(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis(x, y).computeKendallTau()

    def compute_squared_src(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis(x, y).computeSquaredSRC()

elif parse_version("1.19") <= OT_VERSION:

    def compute_pcc(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis.PCC(x, y)

    def compute_prcc(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis.PRCC(x, y)

    def compute_pearson_correlation(  # noqa: D103
        x: ndarray, y: ndarray
    ) -> openturns.Point:
        return openturns.CorrelationAnalysis.PearsonCorrelation(x, y)

    def compute_spearman_correlation(  # noqa: D103
        x: ndarray, y: ndarray
    ) -> openturns.Point:
        return openturns.CorrelationAnalysis.SpearmanCorrelation(x, y)

    def compute_src(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis.SRC(x, y)

    def compute_srrc(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis.SRRC(x, y)

    def compute_kendall_tau(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        msg = "Requires openturns>=1.20"
        raise NotImplementedError(msg)

    def compute_squared_src(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        msg = "Requires openturns>=1.20"
        raise NotImplementedError(msg)

else:

    def compute_pcc(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis_PCC(x, y)

    def compute_prcc(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis_PRCC(x, y)

    def compute_pearson_correlation(  # noqa: D103
        x: ndarray, y: ndarray
    ) -> openturns.Point:
        return openturns.CorrelationAnalysis_PearsonCorrelation(x, y)

    def compute_spearman_correlation(  # noqa: D103
        x: ndarray, y: ndarray
    ) -> openturns.Point:
        return openturns.CorrelationAnalysis_SpearmanCorrelation(x, y)

    def compute_src(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis_SRC(x, y)

    def compute_srrc(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis_SRRC(x, y)

    def compute_kendall_tau(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        msg = "Requires openturns>=1.20"
        raise NotImplementedError(msg)

    def compute_squared_src(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        msg = "Requires openturns>=1.20"
        raise NotImplementedError(msg)
