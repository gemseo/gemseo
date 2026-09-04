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
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import newaxis
from numpy.testing import assert_equal
from openturns import HSICEstimatorConditionalSensitivity
from openturns import HSICEstimatorGlobalSensitivity
from openturns import HSICEstimatorTargetSensitivity
from openturns import HSICUStat
from openturns import IndicatorFunction
from openturns import Interval
from openturns import RandomGenerator
from openturns import Sample
from openturns import SquaredExponential

from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.space.parameter import ParameterSpace
from gemseo.uncertainty.distribution.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.sensitivity.hsic import HSICAnalysis
from gemseo.uncertainty.sensitivity.hsic import HSICAnalysisMethod
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from gemseo.dataset.io_dataset import IODataset
    from gemseo.util.typing import IntegerArray


@pytest.fixture(params=HSICAnalysis.AnalysisType, scope="module")
def analysis_type(request) -> HSICAnalysis.AnalysisType:
    """Return a sensitivity analysis type."""
    return request.param


@pytest.fixture(scope="module")
def samples() -> IODataset:
    """The samples shared by the HSIC sensitivity analyses."""
    discipline = AnalyticDiscipline({"y1": "x1+2*x2", "y2": "x1-2*x2"})

    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable("x1", OTNormalDistribution_Settings())
    uncertain_space.add_random_variable("x2", OTNormalDistribution_Settings())

    analysis = HSICAnalysis()
    analysis.compute_samples([discipline], uncertain_space, 100)
    return analysis.dataset


@pytest.fixture
def hsic_analysis(samples) -> HSICAnalysis:
    """An HSIC sensitivity analysis before calling the compute_indices method."""
    return HSICAnalysis(samples=samples)


@pytest.fixture
def hsic_analysis_2(hsic_analysis, analysis_type) -> HSICAnalysis:
    """A HSIC sensitivity analysis after calling the compute_indices method."""
    hsic_analysis.compute_indices(
        output_bounds={"y1": ([0], [1]), "y2": ([1], [float("inf")])},
        analysis_type=analysis_type,
        use_permutations=True,
        n_permutations=90,
        seed=3,
    )
    return hsic_analysis


@pytest.fixture(scope="module")
def significant_variables(analysis_type) -> dict[str, list[dict[str, IntegerArray]]]:
    """The significant variables."""
    if analysis_type == HSICAnalysis.AnalysisType.GLOBAL:
        return {
            "y1": [{"x1": array([0]), "x2": array([0])}],
            "y2": [{"x1": array([0]), "x2": array([0])}],
        }

    if analysis_type == HSICAnalysis.AnalysisType.TARGET:
        return {
            "y1": [{"x2": array([0])}],
            "y2": [{"x1": array([0]), "x2": array([0])}],
        }

    return {"y1": [{}], "y2": [{"x1": array([0]), "x2": array([0])}]}


@pytest.fixture(scope="module")
def openturns_hsic_indices(samples, analysis_type) -> HSICAnalysis.SensitivityIndices:
    """The HSIC and R2-HSIC indices calculated directly from OpenTURNS."""
    RandomGenerator.SetSeed(3)
    input_samples = Sample(samples.get_view(group_names=samples.INPUT_GROUP).to_numpy())
    x1_covariance_model = SquaredExponential(1)
    x1_covariance_model.setScale(
        input_samples.getMarginal(0).computeStandardDeviation()
    )
    x2_covariance_model = SquaredExponential(1)
    x2_covariance_model.setScale(
        input_samples.getMarginal(1).computeStandardDeviation()
    )
    y1_samples = Sample(
        samples
        .get_view(
            group_names=samples.OUTPUT_GROUP,
            variable_names="y1",
        )
        .to_numpy()
        .T[0][:, newaxis]
    )
    y1_covariance_model = SquaredExponential(1)
    y1_covariance_model.setScale(y1_samples.computeStandardDeviation())
    y2_samples = Sample(
        samples
        .get_view(
            group_names=samples.OUTPUT_GROUP,
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
        str(m).lower().replace("-", "_") for m in HSICAnalysisMethod
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


@pytest.mark.parametrize("kwargs", [{}, {"level": 0.06}, {"use_asymptotic": False}])
def test_filter(
    hsic_analysis_2, analysis_type, significant_variables, kwargs, snapshot
) -> None:
    """Check the filter method."""
    if analysis_type == HSICAnalysis.AnalysisType.CONDITIONAL and kwargs.get(
        "use_asymptotic", True
    ):
        with assert_exception(ValueError, snapshot):
            hsic_analysis_2.filter(**kwargs)
    else:
        assert_equal(hsic_analysis_2.filter(**kwargs), significant_variables)


def test_from_samples(hsic_analysis, tmp_wd):
    """Check the instantiation from samples."""
    file_path = Path("samples.pkl")
    hsic_analysis.compute_indices()
    hsic_analysis.dataset.to_pickle(file_path)
    new_hsic_analysis = HSICAnalysis(samples=file_path)
    new_hsic_analysis.compute_indices()
    assert new_hsic_analysis.indices == hsic_analysis.indices


def test_constant_output(discipline_with_constant_output_and_space):
    """Check that HSICAnalysis supports constant outputs."""
    discipline, uncertain_space = discipline_with_constant_output_and_space
    analysis = HSICAnalysis()
    analysis.compute_samples([discipline], uncertain_space, 100)
    indices = analysis.compute_indices()
    assert indices.hsic["constant"][0] is None
    assert indices.hsic["varying"][0] is not None


def test_plot(hsic_analysis, tmp_wd):
    """Check that HSICAnalysis.plot returns a bar plot of the main indices."""
    hsic_analysis.compute_indices()
    plot = hsic_analysis.plot("y1", save=False)
    assert plot.__class__.__name__ == "BarPlot"


def test_permutation_p_values_not_computed_by_default(hsic_analysis) -> None:
    """Check that the p-values are not estimated through permutations by default."""
    indices = hsic_analysis.compute_indices()
    assert not indices.p_value_permutation
    assert indices.hsic
    assert indices.r2_hsic
    assert indices.p_value_asymptotic


def test_n_permutations_ignored_without_use_permutations(hsic_analysis) -> None:
    """Check that n_permutations is ignored when use_permutations is False."""
    indices = hsic_analysis.compute_indices(n_permutations=1)
    other_indices = hsic_analysis.compute_indices(n_permutations=1000)
    assert not indices.p_value_permutation
    assert not other_indices.p_value_permutation
    assert indices == other_indices


def test_use_permutations(hsic_analysis) -> None:
    """Check that use_permutations enables the p-values estimated by permutations."""
    indices = hsic_analysis.compute_indices(use_permutations=True, n_permutations=10)
    assert indices.p_value_permutation
    for output_name in ["y1", "y2"]:
        for input_name in ["x1", "x2"]:
            assert indices.p_value_permutation[output_name][0][input_name].shape == (1,)


def test_filter_without_permutation_p_values(hsic_analysis, snapshot) -> None:
    """Check the error raised when the permutation p-values have not been computed."""
    hsic_analysis.compute_indices()
    with assert_exception(ValueError, snapshot):
        hsic_analysis.filter(use_asymptotic=False)


@pytest.mark.parametrize("use_asymptotic", [False, True])
def test_filter_conditional_without_permutation_p_values(
    hsic_analysis, use_asymptotic, snapshot
) -> None:
    """Check that filtering a conditional analysis requires the permutation p-values."""
    hsic_analysis.compute_indices(
        output_bounds={"y1": ([0], [1]), "y2": ([1], [float("inf")])},
        analysis_type=HSICAnalysis.AnalysisType.CONDITIONAL,
    )
    with assert_exception(ValueError, snapshot):
        hsic_analysis.filter(use_asymptotic=use_asymptotic)


def test_main_indices_before_compute_indices(snapshot) -> None:
    """Check the error raised when reading main_indices before computing them."""
    with assert_exception(ValueError, snapshot):
        HSICAnalysis().main_indices  # noqa: B018


@pytest.mark.parametrize(
    ("main_method", "analysis_type"),
    [
        (
            HSICAnalysisMethod.P_VALUE_PERMUTATION,
            HSICAnalysis.AnalysisType.GLOBAL,
        ),
        (
            HSICAnalysisMethod.P_VALUE_ASYMPTOTIC,
            HSICAnalysis.AnalysisType.CONDITIONAL,
        ),
    ],
)
def test_main_indices_without_p_values(
    hsic_analysis, main_method, analysis_type, snapshot, monkeypatch
) -> None:
    """Check the error raised when the p-values of main_method are not available."""
    hsic_analysis.compute_indices(
        output_bounds={"y1": ([0], [1]), "y2": ([1], [float("inf")])},
        analysis_type=analysis_type,
    )
    monkeypatch.setattr(hsic_analysis, "main_method", main_method)
    with assert_exception(ValueError, snapshot):
        hsic_analysis.main_indices  # noqa: B018
