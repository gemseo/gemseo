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

from typing import Final

import openturns
from numpy import ndarray
from packaging import version

OT_VERSION: Final[version.Version] = version.parse(openturns.__version__)

IS_OT_LOWER_THAN_1_20: Final[bool] = OT_VERSION < version.parse("1.20")

if OT_VERSION < version.parse("1.18"):

    def get_eigenvalues(  # noqa:D103
        result: openturns.KarhunenLoeveResult,
    ) -> openturns.Point:
        return result.getEigenValues()

else:

    def get_eigenvalues(  # noqa:D103
        result: openturns.KarhunenLoeveResult,
    ) -> openturns.Point:
        return result.getEigenvalues()


if version.parse(openturns.__version__) >= version.parse("1.20"):

    def compute_pcc(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis(x, y).computePCC()

    def compute_prcc(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        return openturns.CorrelationAnalysis(x, y).computePRCC()

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

elif version.parse(openturns.__version__) >= version.parse("1.19"):

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
        raise NotImplementedError("Requires openturns>=1.20")

    def compute_squared_src(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        raise NotImplementedError("Requires openturns>=1.20")

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
        raise NotImplementedError("Requires openturns>=1.20")

    def compute_squared_src(x: ndarray, y: ndarray) -> openturns.Point:  # noqa: D103
        raise NotImplementedError("Requires openturns>=1.20")
