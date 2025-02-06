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

from dataclasses import fields
from pathlib import Path

import pytest
from numpy import newaxis
from openturns import HSICEstimatorConditionalSensitivity
from openturns import HSICEstimatorGlobalSensitivity
from openturns import HSICEstimatorTargetSensitivity
from openturns import HSICUStat
from openturns import IndicatorFunction
from openturns import Interval
from openturns import RandomGenerator
from openturns import Sample
from openturns import SquaredExponential

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.uncertainty.sensitivity.hsic_analysis import HSICAnalysis


@pytest.fixture(params=HSICAnalysis.AnalysisType, scope="module")
def analysis_type(request) -> HSICAnalysis.AnalysisType:
    """Return a sensitivity analysis type."""
    return request.param


@pytest.fixture(scope="module")
def hsic_analysis() -> HSICAnalysis:
    """An HSIC sensitivity analysis before calling the compute_indices method."""
    discipline = AnalyticDiscipline({"y1": "x1+2*x2", "y2": "x1-2*x2"})

    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable("x1", "OTNormalDistribution")
    uncertain_space.add_random_variable("x2", "OTNormalDistribution")

    analysis = HSICAnalysis()
    analysis.compute_samples([discipline], uncertain_space, 100)
    return analysis


@pytest.fixture(scope="module")
def hsic_analysis_2(hsic_analysis, analysis_type) -> HSICAnalysis:
    """A HSIC sensitivity analysis after calling the compute_indices method."""
    hsic_analysis.compute_indices(
        output_bounds={"y1": ([0], [1]), "y2": ([1], [float("inf")])},
        analysis_type=analysis_type,
        n_permutations=90,
        seed=3,
    )
    return hsic_analysis


@pytest.fixture(scope="module")
def openturns_hsic_indices(
    hsic_analysis, analysis_type
) -> HSICAnalysis.SensitivityIndices:
    """The HSIC and R2-HSIC indices calculated directly from OpenTURNS."""
    RandomGenerator.SetSeed(3)
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

    if analysis_type == analysis_type.GLOBAL:
        y1_estimator = HSICEstimatorGlobalSensitivity(
            [x1_covariance_model, x2_covariance_model, y1_covariance_model],
            input_samples,
            y1_samples,
            HSICUStat(),
        )
        y2_estimator = HSICEstimatorGlobalSensitivity(
            [x1_covariance_model, x2_covariance_model, y2_covariance_model],
            input_samples,
            y2_samples,
            HSICUStat(),
        )
    elif analysis_type == analysis_type.TARGET:
        y1_estimator = HSICEstimatorTargetSensitivity(
            [x1_covariance_model, x2_covariance_model, y1_covariance_model],
            input_samples,
            y1_samples,
            HSICUStat(),
            IndicatorFunction(Interval(0, 1)),
        )
        y2_estimator = HSICEstimatorTargetSensitivity(
            [x1_covariance_model, x2_covariance_model, y2_covariance_model],
            input_samples,
            y2_samples,
            HSICUStat(),
            IndicatorFunction(Interval(1, float("inf"))),
        )
    else:
        y1_estimator = HSICEstimatorConditionalSensitivity(
            [x1_covariance_model, x2_covariance_model, y1_covariance_model],
            input_samples,
            y1_samples,
            IndicatorFunction(Interval(0, 1)),
        )
        y2_estimator = HSICEstimatorConditionalSensitivity(
            [x1_covariance_model, x2_covariance_model, y2_covariance_model],
            input_samples,
            y2_samples,
            IndicatorFunction(Interval(1, float("inf"))),
        )

    y1_estimator.setPermutationSize(90)
    y2_estimator.setPermutationSize(90)

    y1_hsic_indices = y1_estimator.getHSICIndices()
    y1_r2hsic_indices = y1_estimator.getR2HSICIndices()
    y1_p_value_p = y1_estimator.getPValuesPermutation()
    y2_hsic_indices = y2_estimator.getHSICIndices()
    y2_r2hsic_indices = y2_estimator.getR2HSICIndices()
    y2_p_value_p = y2_estimator.getPValuesPermutation()

    if analysis_type == analysis_type.CONDITIONAL:
        p_value_asymptotic = {}
    else:
        y1_p_value_a = y1_estimator.getPValuesAsymptotic()
        y2_p_value_a = y2_estimator.getPValuesAsymptotic()
        p_value_asymptotic = {
            "y1": [{"x1": y1_p_value_a[0], "x2": y1_p_value_a[1]}],
            "y2": [{"x1": y2_p_value_a[0], "x2": y2_p_value_a[1]}],
        }

    return HSICAnalysis.SensitivityIndices(
        hsic={
            "y1": [{"x1": y1_hsic_indices[0], "x2": y1_hsic_indices[1]}],
            "y2": [{"x1": y2_hsic_indices[0], "x2": y2_hsic_indices[1]}],
        },
        p_value_asymptotic=p_value_asymptotic,
        p_value_permutation={
            "y1": [{"x1": y1_p_value_p[0], "x2": y1_p_value_p[1]}],
            "y2": [{"x1": y2_p_value_p[0], "x2": y2_p_value_p[1]}],
        },
        r2_hsic={
            "y1": [{"x1": y1_r2hsic_indices[0], "x2": y1_r2hsic_indices[1]}],
            "y2": [{"x1": y2_r2hsic_indices[0], "x2": y2_r2hsic_indices[1]}],
        },
    )


@pytest.mark.parametrize(
    "outputs", [{}, {"output_names": ["y1", "y2"]}, {"output_names": "y2"}]
)
def test_outputs(hsic_analysis, outputs) -> None:
    """Check that outputs are taken into account."""
    hsic_analysis.compute_indices(**outputs)
    output_names = outputs.get("output_names", hsic_analysis.default_output_names)
    if isinstance(output_names, str):
        output_names = [output_names]

    assert list(hsic_analysis.indices.hsic) == output_names


def test_sort_input_variables(hsic_analysis):
    """Check that sort_input_variables works correctly."""
    hsic_analysis.compute_indices()
    assert hsic_analysis.sort_input_variables("y1") == ["x2", "x1"]


def test_methods(hsic_analysis_2) -> None:
    """Check the methods for which the indices have been computed."""
    assert {f.name for f in fields(hsic_analysis_2.indices)} == {
        str(m).lower().replace("-", "_") for m in hsic_analysis_2.Method
    }


def test_outputs_names_and_size(hsic_analysis_2) -> None:
    """Check the names and sizes of the outputs."""
    hsic_index = hsic_analysis_2.indices.hsic
    output_names = ["y1", "y2"]
    assert list(hsic_index) == output_names
    for output_name in output_names:
        assert len(hsic_index[output_name]) == 1


def test_inputs_names_and_size(hsic_analysis_2) -> None:
    """Check the names and sizes of the inputs."""
    hsic_index = hsic_analysis_2.indices.hsic
    input_names = ["x1", "x2"]
    assert list(hsic_index["y1"][0]) == input_names
    for input_name in input_names:
        assert hsic_index["y1"][0][input_name].shape == (1,)


def test_hsic_indices_values(hsic_analysis_2, openturns_hsic_indices) -> None:
    """Check that the global HSIC indices are equal to the indices computed with
    OpenTURNS."""
    assert hsic_analysis_2.indices == openturns_hsic_indices


def test_from_samples(hsic_analysis, tmp_wd):
    """Check the instantiation from samples."""
    file_path = Path("samples.pkl")
    hsic_analysis.compute_indices()
    hsic_analysis.dataset.to_pickle(file_path)
    new_hsic_analysis = HSICAnalysis(samples=file_path)
    new_hsic_analysis.compute_indices()
    assert new_hsic_analysis.indices == hsic_analysis.indices
