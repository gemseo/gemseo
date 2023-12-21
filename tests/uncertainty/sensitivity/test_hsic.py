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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Olivier Sapin
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the class HSICAnalysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import newaxis
from openturns import HSICEstimatorGlobalSensitivity
from openturns import HSICUStat
from openturns import Sample
from openturns import SquaredExponential

from gemseo import create_discipline
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.uncertainty.sensitivity.hsic.analysis import HSICAnalysis

if TYPE_CHECKING:
    from gemseo.uncertainty.sensitivity.analysis import FirstOrderIndicesType


@pytest.fixture(scope="module")
def hsic_analysis() -> HSICAnalysis:
    """A sensitivity analysis based on normalized HSIC."""
    discipline = create_discipline(
        "AnalyticDiscipline", expressions={"y1": "x1+2*x2", "y2": "x1-2*x2"}
    )
    space = ParameterSpace()
    for name in ["x1", "x2"]:
        space.add_random_variable(name, "OTNormalDistribution")
    return HSICAnalysis([discipline], space, 100)


@pytest.fixture(scope="module")
def hsic_analysis_2(hsic_analysis) -> HSICAnalysis:
    """A sensitivity analysis based on normalized HSIC."""
    hsic_analysis.compute_indices()
    return hsic_analysis


@pytest.fixture(scope="module")
def openturns_hsic_indices(
    hsic_analysis: HSICAnalysis,
) -> dict[str, FirstOrderIndicesType]:
    """Compute the HSIC and R2HSIC indices using a HSIC estimator based on U-statistics
    with a Gaussian covariance model for global analysis."""
    input_samples = Sample(
        hsic_analysis.dataset.get_view(
            group_names=hsic_analysis.dataset.INPUT_GROUP
        ).to_numpy()
    )
    x1_covariance_model = SquaredExponential(1)
    x1_covariance_model.setScale(
        input_samples.getMarginal(0).computeStandardDeviation()
    )
    x2_covariance_model = SquaredExponential(1)
    x2_covariance_model.setScale(
        input_samples.getMarginal(1).computeStandardDeviation()
    )
    y1_samples = Sample(
        hsic_analysis.dataset.get_view(
            group_names=hsic_analysis.dataset.OUTPUT_GROUP,
            variable_names="y1",
        )
        .to_numpy()
        .T[0][:, newaxis]
    )
    y1_covariance_model = SquaredExponential(1)
    y1_covariance_model.setScale(y1_samples.computeStandardDeviation())
    y2_samples = Sample(
        hsic_analysis.dataset.get_view(
            group_names=hsic_analysis.dataset.OUTPUT_GROUP,
            variable_names="y2",
        )
        .to_numpy()
        .T[0][:, newaxis]
    )
    y2_covariance_model = SquaredExponential(1)
    y2_covariance_model.setScale(y2_samples.computeStandardDeviation())

    y1_hsic_class = HSICEstimatorGlobalSensitivity(
        [x1_covariance_model, x2_covariance_model, y1_covariance_model],
        input_samples,
        y1_samples,
        HSICUStat(),
    )

    y2_hsic_class = HSICEstimatorGlobalSensitivity(
        [x1_covariance_model, x2_covariance_model, y2_covariance_model],
        input_samples,
        y2_samples,
        HSICUStat(),
    )

    y1_hsic_indices = y1_hsic_class.getHSICIndices()
    y1_r2hsic_indices = y1_hsic_class.getR2HSICIndices()
    y2_hsic_indices = y2_hsic_class.getHSICIndices()
    y2_r2hsic_indices = y2_hsic_class.getR2HSICIndices()

    hsic_indices = {
        "y1": [{"x1": y1_hsic_indices[0], "x2": y1_hsic_indices[1]}],
        "y2": [{"x1": y2_hsic_indices[0], "x2": y2_hsic_indices[1]}],
    }
    r2_hsic_indices = {
        "y1": [{"x1": y1_r2hsic_indices[0], "x2": y1_r2hsic_indices[1]}],
        "y2": [{"x1": y2_r2hsic_indices[0], "x2": y2_r2hsic_indices[1]}],
    }

    return {"HSIC": hsic_indices, "R2-HSIC": r2_hsic_indices}


@pytest.mark.parametrize("outputs", [{}, {"outputs": ["y1", "y2"]}, {"outputs": "y2"}])
def test_outputs(hsic_analysis: HSICAnalysis, outputs):
    """Check that outputs are taken into account."""
    hsic_analysis.compute_indices(**outputs)
    output_names = outputs.get("outputs", hsic_analysis.default_output)
    if isinstance(output_names, str):
        output_names = [output_names]
    assert set(output_names) == set(
        hsic_analysis.indices[hsic_analysis.Method.HSIC].keys()
    )


def test_methods(hsic_analysis_2: HSICAnalysis):
    """Check the methods for which the indices have been computed."""
    all_methods = set(hsic_analysis_2.Method)
    available_methods = set(hsic_analysis_2.indices.keys())
    assert available_methods == all_methods


def test_outputs_names_and_size(
    hsic_analysis_2: HSICAnalysis, openturns_hsic_indices: FirstOrderIndicesType
):
    """Check the names and sizes of the outputs."""
    indices = hsic_analysis_2.indices
    hsic_index = indices["HSIC"]
    output_names = {"y1", "y2"}
    assert set(hsic_index.keys()) == output_names
    for output_name in output_names:
        assert len(hsic_index[output_name]) == 1


def test_inputs_names_and_size(
    hsic_analysis_2: HSICAnalysis, openturns_hsic_indices: FirstOrderIndicesType
):
    """Check the names and sizes of the inputs."""
    indices = hsic_analysis_2.indices
    hsic_index = indices["HSIC"]
    input_names = {"x1", "x2"}
    assert set(hsic_index["y1"][0].keys()) == input_names
    for input_name in input_names:
        assert hsic_index["y1"][0][input_name].shape == (1,)


def test_method_names(
    hsic_analysis_2: HSICAnalysis, openturns_hsic_indices: FirstOrderIndicesType
):
    """Check that the property ``method`` is ``indices[algo.lower()]``."""
    indices = hsic_analysis_2.indices
    for method_name in hsic_analysis_2.Method:
        assert (
            getattr(hsic_analysis_2, method_name.lower().replace("-", "_"))
            == indices[method_name]
        )


def test_hsic_indices_values(
    hsic_analysis_2: HSICAnalysis, openturns_hsic_indices: FirstOrderIndicesType
):
    """Check that the global HSIC indices are equal to the indices computed with
    OpenTURNS."""
    indices = hsic_analysis_2.indices
    assert indices == openturns_hsic_indices
