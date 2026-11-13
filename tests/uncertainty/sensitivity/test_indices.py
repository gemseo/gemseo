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
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the class BaseSensitivityAnalysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import NamedTuple

import pytest
from numpy import array
from numpy import hstack
from numpy import isnan
from numpy import linspace
from numpy import nan
from numpy import pi
from numpy import sin
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from gemseo import create_discipline
from gemseo.core.discipline import Discipline
from gemseo.dataset.dataset import Dataset
from gemseo.dataset.io_dataset import IODataset
from gemseo.discipline import AutoPyDiscipline
from gemseo.doe.openturns.settings.ot_monte_carlo import OT_MONTE_CARLO_Settings
from gemseo.post.dataset.bar_plot_settings import BarPlot_Settings
from gemseo.problem.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problem.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.scenario.backup_settings import BackupSettings
from gemseo.space.random import RandomSpace
from gemseo.uncertainty.distribution.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.uncertainty.sensitivity.core.base import BaseSensitivityAnalysis
from gemseo.uncertainty.sensitivity.correlation import CorrelationAnalysis
from gemseo.uncertainty.sensitivity.correlation import CorrelationAnalysisMethod
from gemseo.uncertainty.sensitivity.morris import MorrisAnalysis
from gemseo.uncertainty.sensitivity.sobol import SobolAnalysis
from gemseo.util.testing.helper import assert_exception
from gemseo.util.testing.helper import concretize_classes

if TYPE_CHECKING:
    from gemseo.discipline.analytic import AnalyticDiscipline
    from gemseo.uncertainty.sensitivity.core.base import FirstOrderIndicesType
    from gemseo.util.string import VariableOrComponent
    from gemseo.util.typing import StrKeyMapping


@pytest.fixture(scope="module")
def discipline() -> AnalyticDiscipline:
    """Return a discipline of interest."""
    return create_discipline("AnalyticDiscipline", {"out": "x1+2*x2+3*x3"})


@pytest.fixture(scope="module")
def parameter_space() -> RandomSpace:
    """Return the parameter space on which to evaluate the discipline."""
    space = RandomSpace()
    for name in ["x1", "x2", "x3"]:
        space.add_variable(name, OTNormalDistribution_Settings())
    return space


class Ishigami1D(Discipline):
    """A version of the Ishigami function indexed by a 1D variable."""

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(["x1", "x2", "x3"])
        self.io.output_grammar.update_from_names(["out"])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x_1 = input_data["x1"]
        x_2 = input_data["x2"]
        x_3 = input_data["x3"]
        time = linspace(0, 1, 100)
        output = sin(x_1) + 7 * sin(x_2) ** 2 + 0.1 * x_3**4 * sin(x_1) * time
        return {"out": output}


class MockSensitivityAnalysis(BaseSensitivityAnalysis):
    """This sensitivity analysis returns dummy indices with dummy methods m1 and m2.

    It relies on a dummy dataset with inputs x1 and x2 and outputs y1 and y2. xi and yi
    of dimension i.
    """

    main_method = "m1"

    @dataclass
    class SensitivityIndices:
        m1: FirstOrderIndicesType
        m2: FirstOrderIndicesType

    def __init__(self) -> None:
        self.dataset = IODataset()
        data = array([[1, 2, 3]])
        self._input_names = ["x1", "x2"]
        self._output_names = ["y1", "y2"]
        self.dataset.add_group(
            self.dataset.input_group, data, self._input_names, {"x1": 1, "x2": 2}
        )
        self.dataset.add_group(
            self.dataset.output_group, data, ["y1", "y2"], {"y1": 1, "y2": 2}
        )

    @property
    def indices(self):
        return self.SensitivityIndices(
            m1={
                "y1": [{"x1": array([1.25]), "x2": array([1.5, 1.75])}],
                "y2": [
                    {"x1": array([2.25]), "x2": array([2.5, 2.75])},
                    {"x1": array([3.25]), "x2": array([3.5, 3.75])},
                ],
            },
            m2={
                "y1": [{"x1": array([1]), "x2": array([1, 1])}],
                "y2": [
                    {"x1": array([2]), "x2": array([2, 2])},
                    {"x1": array([3]), "x2": array([3, 3])},
                ],
            },
        )

    @property
    def main_indices(self):
        return self.indices.m1


class SecondMockSensitivityAnalysis(MockSensitivityAnalysis):
    """This sensitivity analysis uses the values of m2 as main indices."""

    @property
    def main_indices(self):
        return self.indices.m2


class NaNMockSensitivityAnalysis(MockSensitivityAnalysis):
    """A sensitivity analysis whose indices are partly `nan`.

    The input `x1` has no index at all,
    and the first component of the input `x2` has no index either.
    """

    @property
    def main_indices(self):
        return {
            "y1": [{"x1": array([nan]), "x2": array([nan, 1.75])}],
            "y2": [
                {"x1": array([nan]), "x2": array([nan, 2.75])},
                {"x1": array([nan]), "x2": array([nan, 3.75])},
            ],
        }


class ZeroAndNaNMockSensitivityAnalysis(MockSensitivityAnalysis):
    """A sensitivity analysis mixing a `nan` index and a zero one.

    The input `x1` has no index at all,
    while the indices of the input `x2` are zero,
    as when an output does not depend on an input.
    """

    @property
    def main_indices(self):
        return {"y1": [{"x1": array([nan]), "x2": array([0.0, 0.0])}]}


class MockMorrisAnalysisIndices(MorrisAnalysis):
    """A mock of a Morris sensitivity analysis, from which a dataset can be exported."""

    def __init__(self) -> None:
        self.dataset = IODataset()
        self.dataset.sizes = {
            "x1": 1,
        }

    @property
    def input_names(self) -> list[str]:
        """The names of the inputs."""
        return ["x1"]

    @property
    def indices(self):
        return self.SensitivityIndices(
            mu={
                "y": [
                    {
                        "x1": array([-0.36000398]),
                    }
                ]
            },
            mu_star={
                "y": [
                    {
                        "x1": array([0.67947346]),
                    }
                ]
            },
            sigma={
                "y": [
                    {
                        "x1": array([0.98724949]),
                    }
                ]
            },
            relative_sigma={
                "y": [
                    {
                        "x1": array([1.45296254]),
                    }
                ]
            },
            min={
                "y": [
                    {
                        "x1": array([0.0338188]),
                    }
                ]
            },
            max={
                "y": [
                    {
                        "x1": array([2.2360336]),
                    }
                ]
            },
        )


@pytest.fixture
def mock_sensitivity_analysis() -> MockSensitivityAnalysis:
    """Return an instance of MockSensitivityAnalysis."""
    with concretize_classes(MockSensitivityAnalysis):
        return MockSensitivityAnalysis()


class ConstantMockSensitivityAnalysis(MockSensitivityAnalysis):
    """This sensitivity analysis has constant output components.

    The output y1 and the second component of the output y2 are constant.
    """

    @property
    def indices(self):
        return self.SensitivityIndices(
            m1={
                "y1": [None],
                "y2": [{"x1": array([2.25]), "x2": array([2.5, 2.75])}, None],
            },
            m2={
                "y1": [None],
                "y2": [{"x1": array([2]), "x2": array([2, 2])}, None],
            },
        )


@pytest.fixture
def constant_mock_sensitivity_analysis() -> ConstantMockSensitivityAnalysis:
    """Return an instance of ConstantMockSensitivityAnalysis."""
    with concretize_classes(ConstantMockSensitivityAnalysis):
        return ConstantMockSensitivityAnalysis()


class ScalarMockSensitivityAnalysis(MockSensitivityAnalysis):
    """This sensitivity analysis considers every output as scalar."""

    def _get_output_n_components(self, name: str) -> int:
        return 1


@pytest.fixture
def scalar_mock_sensitivity_analysis() -> ScalarMockSensitivityAnalysis:
    """Return an instance of ScalarMockSensitivityAnalysis."""
    with concretize_classes(ScalarMockSensitivityAnalysis):
        return ScalarMockSensitivityAnalysis()


def _create_mock_sensitivity_analysis(
    class_name: str, main_indices: FirstOrderIndicesType
) -> MockSensitivityAnalysis:
    """Create a sensitivity analysis with given main indices.

    Args:
        class_name: The name of the class of the sensitivity analysis,
            which appears in the error messages.
        main_indices: The main indices.

    Returns:
        The sensitivity analysis.
    """
    cls = type(
        class_name,
        (MockSensitivityAnalysis,),
        {"main_indices": property(lambda self: main_indices)},
    )
    with concretize_classes(cls):
        return cls()


def _create_one_component_mock_sensitivity_analysis() -> MockSensitivityAnalysis:
    """Create a sensitivity analysis with a single component of the output y2.

    Returns:
        The sensitivity analysis.
    """
    return _create_mock_sensitivity_analysis(
        "OneComponentMockSensitivityAnalysis",
        {
            "y1": [{"x1": array([1.25]), "x2": array([1.5, 1.75])}],
            "y2": [{"x1": array([2.25]), "x2": array([2.5, 2.75])}],
        },
    )


@pytest.fixture
def first_component_constant_mock_sensitivity_analysis() -> MockSensitivityAnalysis:
    """A sensitivity analysis with a constant first component of the output y2."""
    return _create_mock_sensitivity_analysis(
        "FirstComponentConstantMockSensitivityAnalysis",
        {
            "y1": [{"x1": array([1.25]), "x2": array([1.5, 1.75])}],
            "y2": [None, {"x1": array([3.25]), "x2": array([3.5, 3.75])}],
        },
    )


@pytest.fixture
def second_mock_sensitivity_analysis() -> SecondMockSensitivityAnalysis:
    """Return an instance of SecondMockSensitivityAnalysis."""
    with concretize_classes(SecondMockSensitivityAnalysis):
        return SecondMockSensitivityAnalysis()


@pytest.fixture
def nan_mock_sensitivity_analysis() -> NaNMockSensitivityAnalysis:
    """Return an instance of NaNMockSensitivityAnalysis."""
    with concretize_classes(NaNMockSensitivityAnalysis):
        return NaNMockSensitivityAnalysis()


@pytest.fixture
def zero_and_nan_mock_sensitivity_analysis() -> ZeroAndNaNMockSensitivityAnalysis:
    """Return an instance of ZeroAndNaNMockSensitivityAnalysis."""
    with concretize_classes(ZeroAndNaNMockSensitivityAnalysis):
        return ZeroAndNaNMockSensitivityAnalysis()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"outputs": "y2", "standardize": True},
        {"outputs": "y2", "input_names": ["x1"]},
        {"standardize": True, "input_names": ["x1"], "outputs": "y2"},
        {"outputs": [("y2", 0)]},
        {"outputs": "y2", "bar_plot_settings": BarPlot_Settings(n_digits=1)},
        {"outputs": ["y1", "y2"], "sort": False},
        {"outputs": ["y1", "y2"], "sorting_output": "y2"},
        {"outputs": ["y1", "y2"], "sorting_output": ("y2", 1)},
    ],
)
def test_plot_bar(kwargs, mock_sensitivity_analysis, snapshot_matplotlib) -> None:
    """Check that a Barplot is created with plot_bar."""
    mock_sensitivity_analysis.plot_bar(save=False, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"outputs": "y2", "standardize": True},
        {"outputs": "y2", "input_names": ["x1"]},
        {"standardize": True, "input_names": ["x1"], "outputs": "y2"},
        {"outputs": [("y2", 0)]},
    ],
)
def test_plot_radar(kwargs, mock_sensitivity_analysis, snapshot_matplotlib) -> None:
    """Check that a RadarChart is created with plot_radar."""
    mock_sensitivity_analysis.plot_radar(save=False, show=False, **kwargs)


@pytest.mark.parametrize(
    "use_bar_plot",
    [True, False],
)
def test_plot_comparison(
    use_bar_plot, discipline, parameter_space, snapshot_matplotlib
) -> None:
    """Check if the comparison of sensitivity indices works."""
    spearman = CorrelationAnalysis()
    spearman.compute_samples([discipline], parameter_space, 10)
    spearman.compute_indices()
    pearson = CorrelationAnalysis()
    pearson.compute_samples([discipline], parameter_space, 10)
    pearson.main_method = CorrelationAnalysisMethod.PEARSON
    pearson.compute_indices()
    plot = pearson.plot_comparison(
        spearman, "out", save=False, title="foo", use_bar_plot=use_bar_plot
    )
    assert plot.settings.title == "foo"


def test_input_names(mock_sensitivity_analysis) -> None:
    """Check the value of the attribute input_names.

    Args:
        mock_sensitivity_analysis: The sensitivity analysis.
    """
    assert mock_sensitivity_analysis.input_names == ["x1", "x2"]


@pytest.mark.parametrize(
    ("analysis_fixture", "output"),
    [
        ("mock_sensitivity_analysis", "y1"),
        ("mock_sensitivity_analysis", ("y1", 0)),
        ("mock_sensitivity_analysis", ("y2", 0)),
        # The second component of y2 is constant.
        ("constant_mock_sensitivity_analysis", ("y2", 0)),
    ],
)
def test_sort_parameters(request, analysis_fixture, output) -> None:
    """Check if the parameters are well sorted.

    Args:
        request: The pytest request.
        analysis_fixture: The name of the fixture of the sensitivity analysis.
        output: The value used to sort the parameters.
    """
    analysis = request.getfixturevalue(analysis_fixture)
    assert analysis.sort_input_variables(output) == ["x2", "x1"]


@pytest.mark.parametrize("output", ["y1", ("y1", 0), ("y2", 0)])
def test_sort_parameters_with_nan(nan_mock_sensitivity_analysis, output) -> None:
    """Check that the inputs whose indices are `nan` are ranked last.

    The cumulative index of an input sums the components that have an index,
    and so `x1`, which has none, comes after `x2`,
    whose second component has one.
    """
    assert nan_mock_sensitivity_analysis.sort_input_variables(output) == ["x2", "x1"]


def test_sort_parameters_with_nan_and_zero(
    zero_and_nan_mock_sensitivity_analysis,
) -> None:
    """Check that an input without index is ranked after an input with a zero index.

    The cumulative index of `x1`, which has no index at all,
    and that of `x2`, whose indices are zero, are both zero,
    and so only the presence of an index can tell the two apart.
    """
    analysis = zero_and_nan_mock_sensitivity_analysis
    assert analysis.sort_input_variables("y1") == ["x2", "x1"]


@pytest.mark.parametrize(
    ("standardize", "expected"),
    [(False, [[1.75], [2.75], [3.75]]), (True, [[1.0]] * 3)],
)
def test_plot_bar_without_the_components_without_indices(
    nan_mock_sensitivity_analysis, standardize, expected
) -> None:
    """Check that the plotted dataset leaves out the components without indices.

    The input `x1` has no index at all
    and the first component of the input `x2` has no index either,
    and so the second component of `x2` is the only one left to plot.
    """
    dataset = nan_mock_sensitivity_analysis.plot_bar(
        save=False, standardize=standardize
    ).dataset
    assert dataset.columns.tolist() == [(dataset.parameter_group, "x2", 1)]
    assert_array_equal(dataset.to_numpy(), array(expected))


def test_convert_to_dataset(mock_sensitivity_analysis) -> None:
    """Check if the sensitivity indices are well converted to Dataset.

    Args:
        mock_sensitivity_analysis: The sensitivity analysis.
    """
    dataset = mock_sensitivity_analysis.to_dataset()
    assert isinstance(dataset, Dataset)
    assert (dataset.index == ["x1", "x2[0]", "x2[1]"]).all()
    assert dataset.variable_names == ["y1", "y2"]
    assert dataset.group_names == ["m1", "m2"]
    assert_array_equal(
        dataset.get_view(group_names="m2", variable_names="y1").to_numpy(),
        array([[1], [1], [1]]),
    )
    assert_array_equal(
        dataset.get_view(group_names="m2", variable_names="y2").to_numpy(),
        array([[2, 3], [2, 3], [2, 3]]),
    )


@pytest.fixture(scope="module")
def ishigami() -> SobolAnalysis:
    """Return the Sobol' analysis for the Ishigami function."""
    space = RandomSpace()
    for variable in ["x1", "x2", "x3"]:
        space.add_variable(
            variable, OTUniformDistribution_Settings(minimum=-pi, maximum=pi)
        )

    sobol_analysis = SobolAnalysis()
    sobol_analysis.compute_samples(
        [Ishigami1D()], space, 100, compute_second_order=False
    )
    sobol_analysis.main_method = "total"
    sobol_analysis.compute_indices()
    return sobol_analysis


@pytest.mark.parametrize(
    ("kwargs", "output"),
    [
        ({}, "out"),
        ({}, ("out", 0)),
        ({"standardize": True}, "out"),
        ({"input_names": ["x1", "x3"]}, "out"),
        ({"standardize": True, "input_names": ["x1", "x3"]}, "out"),
        ({"properties": {"xlabel": "foo"}}, "out"),
    ],
)
def test_plot_1d_field(kwargs, output, ishigami, snapshot_matplotlib) -> None:
    """Check if a 1D field is well plotted."""
    ishigami.plot_field(output, save=False, show=False, **kwargs)


two_d_field_test_parameters_wo_mesh = {
    "without_option": ({}, ["2d_field_wo_mesh"]),
}


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"standardize": True},
        {"input_names": ["x1", "x3"]},
        {"standardize": True, "input_names": ["x1", "x3"]},
    ],
)
def test_plot_2d_field(kwargs, ishigami, snapshot_matplotlib) -> None:
    """Check if a 2D field is well plotted with mesh."""
    times = linspace(0, 1, 10)
    mesh = array([[time1, time2] for time1 in times for time2 in times])
    ishigami.plot_field("out", save=False, show=False, mesh=mesh, **kwargs)


def test_standardize_indices() -> None:
    """Check that the method standardize_indices() works."""
    indices = {
        "y1": [{"x1": array([-2.0]), "x2": array([0.5]), "x3": array([1.0])}],
        "y2": [
            {"x1": array([2.0]), "x2": array([0.5]), "x3": array([-1.0])},
            {"x1": array([0.0]), "x2": array([0.5]), "x3": array([2.0])},
        ],
    }
    standardized_indices = BaseSensitivityAnalysis.standardize_indices(indices)
    expected_standardized_indices = {
        "y1": [{"x1": array([1.0]), "x2": array([0.25]), "x3": array([0.5])}],
        "y2": [
            {"x1": array([1.0]), "x2": array([0.25]), "x3": array([0.5])},
            {"x1": array([0.0]), "x2": array([0.25]), "x3": array([1.0])},
        ],
    }
    assert standardized_indices == expected_standardized_indices


def test_standardize_indices_with_undefined_indices() -> None:
    """Check that standardize_indices() ignores the undefined indices.

    A `nan` index does not contribute to the largest index,
    which the other indices are divided by,
    and is left as it is.
    The `None` indices of a constant output component are left as they are too.
    """
    indices = {
        "y1": [{"x1": array([nan]), "x2": array([0.5]), "x3": array([1.0])}],
        "y2": [{"x1": array([nan]), "x2": array([nan]), "x3": array([nan])}],
        "y3": [None],
        "y4": [{"x1": array([-2.0]), "x2": array([1.0]), "x3": array([0.0])}, None],
    }
    standardized_indices = BaseSensitivityAnalysis.standardize_indices(indices)
    assert isnan(standardized_indices["y1"][0]["x1"]).all()
    assert_array_equal(standardized_indices["y1"][0]["x2"], array([0.5]))
    assert_array_equal(standardized_indices["y1"][0]["x3"], array([1.0]))
    # No index of the output y2 is a number, and so all of them are left as they are.
    assert all(isnan(value).all() for value in standardized_indices["y2"][0].values())
    assert standardized_indices["y3"] == [None]
    assert standardized_indices["y4"][1] is None
    assert_array_equal(standardized_indices["y4"][0]["x1"], array([1.0]))
    assert_array_equal(standardized_indices["y4"][0]["x2"], array([0.5]))


def test_multiple_disciplines(parameter_space) -> None:
    """Test a sensitivity analysis with multiple disciplines.

    Args:
        parameter_space: A parameter space for the analysis.
    """
    expressions = [{"y1": "x1+x3+y2"}, {"y2": "x2+x3+2*y1"}, {"f": "x3+y1+y2"}]
    d1 = create_discipline("AnalyticDiscipline", expressions[0])
    d2 = create_discipline("AnalyticDiscipline", expressions[1])
    d3 = create_discipline("AnalyticDiscipline", expressions[2])

    sensitivity_analysis = CorrelationAnalysis()
    sensitivity_analysis.compute_samples(
        [d1, d2, d3], parameter_space, 5, algo_settings=OT_MONTE_CARLO_Settings()
    )

    assert sensitivity_analysis.dataset.get_variable_names("inputs") == [
        "x1",
        "x2",
        "x3",
    ]
    assert sensitivity_analysis.dataset.get_variable_names("outputs") == [
        "f",
        "y1",
        "y2",
    ]
    assert sensitivity_analysis.dataset.n_samples == 5


def test_backup(tmp_wd):
    """Check the creation of a backup file."""
    sensitivity_analysis = SobolAnalysis()
    sensitivity_analysis.compute_samples(
        [IshigamiDiscipline()],
        IshigamiSpace(),
        n_samples=50,
        backup_settings=BackupSettings(file_path=".backup"),
    )
    assert Path(".backup").exists()


@pytest.mark.parametrize("output", ["y1", ("y2", 1)], ids=["name", "component"])
def test_sort_input_variables_with_constant_output(
    constant_mock_sensitivity_analysis, output, snapshot
) -> None:
    """Check that sorting the inputs of a constant output component raises."""
    with assert_exception(ValueError, snapshot):
        constant_mock_sensitivity_analysis.sort_input_variables(output)


@pytest.mark.parametrize("output", ["y1", ("y1", 0)], ids=["name", "component"])
def test_plot_comparison_with_constant_output(
    constant_mock_sensitivity_analysis, output, snapshot
) -> None:
    """Check that plot_comparison() raises for a constant output."""
    analysis = constant_mock_sensitivity_analysis
    with assert_exception(ValueError, snapshot):
        analysis.plot_comparison(analysis, output, save=False)


@pytest.mark.parametrize(
    "kwargs",
    [{"sorting_output": "y1"}, {"outputs": "y2", "sorting_output": ("y2", 1)}],
    ids=["name", "component"],
)
def test_plot_bar_with_constant_sorting_output(
    constant_mock_sensitivity_analysis, kwargs, snapshot
) -> None:
    """Check that plot_bar() raises when the sorting output is constant.

    The sorting output is either a name whose components are all constant
    or a constant component.
    """
    with assert_exception(ValueError, snapshot):
        constant_mock_sensitivity_analysis.plot_bar(save=False, **kwargs)


def _compute_analysis(
    analysis_class: type[BaseSensitivityAnalysis],
    discipline: Discipline,
    random_space: RandomSpace,
    n_samples: int,
) -> BaseSensitivityAnalysis:
    """Create a sensitivity analysis and compute its indices.

    Args:
        analysis_class: The class of the sensitivity analysis.
        discipline: The discipline to sample.
        random_space: The random space on which to sample the discipline.
        n_samples: The number of samples.

    Returns:
        The sensitivity analysis with its indices computed.
    """
    analysis = analysis_class()
    analysis.compute_samples([discipline], random_space, n_samples)
    analysis.compute_indices()
    return analysis


class AnalysisWithConstantOutputs(NamedTuple):
    """An analysis with constant outputs and a single varying output component."""

    analysis: BaseSensitivityAnalysis
    """The analysis."""

    constant_outputs: list[VariableOrComponent]
    """The constant outputs."""

    varying_output: tuple[str, int]
    """The only varying output component."""

    varying_label: str
    """The label of the only varying output component."""

    n_input_components: int
    """The number of input components."""


@pytest.fixture(params=["mock", CorrelationAnalysis, SobolAnalysis])
def analysis_with_constant_outputs(
    request, discipline_with_constant_output_and_space
) -> AnalysisWithConstantOutputs:
    """An analysis with constant outputs."""
    if request.param == "mock":
        with concretize_classes(ConstantMockSensitivityAnalysis):
            analysis = ConstantMockSensitivityAnalysis()

        return AnalysisWithConstantOutputs(
            analysis, ["y1", ("y2", 1)], ("y2", 0), "y2[0]", 3
        )

    analysis = _compute_analysis(
        request.param, *discipline_with_constant_output_and_space, 100
    )
    return AnalysisWithConstantOutputs(
        analysis, ["constant"], ("varying", 0), "varying", 2
    )


@pytest.mark.parametrize("plot_name", ["plot_bar", "plot_radar"])
def test_plot_only_constant_outputs(
    analysis_with_constant_outputs, plot_name, snapshot
) -> None:
    """Check that plotting only constant output components raises."""
    case = analysis_with_constant_outputs
    with assert_exception(ValueError, snapshot):
        getattr(case.analysis, plot_name)(outputs=case.constant_outputs, save=False)


@pytest.mark.parametrize("plot_name", ["plot_bar", "plot_radar"])
@pytest.mark.parametrize("standardize", [False, True], ids=["raw", "standardized"])
def test_plot_leaves_out_the_constant_output_components(
    analysis_with_constant_outputs, plot_name, standardize
) -> None:
    """Check that the plots leave out the constant output components.

    Regression test: standardize_indices() used to fail on the `None` indices
    of a constant output, even though it is not plotted.
    """
    case = analysis_with_constant_outputs
    dataset = getattr(case.analysis, plot_name)(
        save=False, standardize=standardize
    ).dataset
    assert dataset.index.tolist() == [case.varying_label]
    assert len(dataset.columns) == case.n_input_components
    # The plotted indices are those of the varying output component,
    # sorted in descending order and, if required, divided by the largest one.
    name, component = case.varying_output
    values = hstack(list(case.analysis.main_indices[name][component].values()))
    if standardize:
        values = abs(values) / abs(values).max()

    assert_allclose(dataset.to_numpy()[0], sorted(values, reverse=True))


@pytest.fixture(
    params=[
        ("mock", 0),
        (CorrelationAnalysis, 100),
        (MorrisAnalysis, 0),
        (SobolAnalysis, 100),
    ],
    ids=["mock", "CorrelationAnalysis", "MorrisAnalysis", "SobolAnalysis"],
)
def analysis_with_constant_output(
    request, discipline_with_constant_output_and_space
) -> tuple[BaseSensitivityAnalysis, str]:
    """An analysis with a constant output and the name of this output."""
    analysis_class, n_samples = request.param
    if analysis_class == "mock":
        with concretize_classes(ConstantMockSensitivityAnalysis):
            return ConstantMockSensitivityAnalysis(), "y1"

    analysis = _compute_analysis(
        analysis_class, *discipline_with_constant_output_and_space, n_samples
    )
    return analysis, "constant"


def test_plot_constant_output(analysis_with_constant_output, snapshot) -> None:
    """Check that plot() raises for a constant output."""
    analysis, output_name = analysis_with_constant_output
    with assert_exception(ValueError, snapshot):
        analysis.plot(output_name, save=False)


class AnalysisWithConstantOutputComponents(NamedTuple):
    """An analysis with constant output components and varying ones."""

    analysis: BaseSensitivityAnalysis
    """The analysis."""

    constant_components: list[tuple[str, int]]
    """The constant output components."""

    varying_components: list[tuple[str, int]]
    """The varying output components."""

    input_labels: list[str]
    """The labels of the input components."""


@pytest.fixture(
    params=[
        (None, 0, [("y1", 0), ("y2", 1)], [("y2", 0)], ["x1", "x2[0]", "x2[1]"]),
        (CorrelationAnalysis, 100, [("constant", 0)], [("varying", 0)], ["x1", "x2"]),
        (MorrisAnalysis, 0, [("constant", 0)], [("varying", 0)], ["x1", "x2"]),
        (SobolAnalysis, 100, [("constant", 0)], [("varying", 0)], ["x1", "x2"]),
    ],
    ids=["mock", "CorrelationAnalysis", "MorrisAnalysis", "SobolAnalysis"],
)
def analysis_with_constant_output_components(
    request, discipline_with_constant_output_and_space
) -> AnalysisWithConstantOutputComponents:
    """An analysis with constant output components."""
    analysis_class, n_samples, constant, varying, input_labels = request.param
    if analysis_class is None:
        with concretize_classes(ConstantMockSensitivityAnalysis):
            analysis = ConstantMockSensitivityAnalysis()
    else:
        analysis = _compute_analysis(
            analysis_class, *discipline_with_constant_output_and_space, n_samples
        )

    return AnalysisWithConstantOutputComponents(
        analysis, constant, varying, input_labels
    )


def test_to_dataset_with_constant_output(
    analysis_with_constant_output_components,
) -> None:
    """Check that to_dataset() stores NaN for a constant output in every group."""
    case = analysis_with_constant_output_components
    dataset = case.analysis.to_dataset()
    assert dataset.index.tolist() == case.input_labels
    assert all(
        isnan(
            dataset.get_view(variable_names=name, components=component).to_numpy()
        ).all()
        for name, component in case.constant_components
    )
    assert not any(
        isnan(
            dataset.get_view(variable_names=name, components=component).to_numpy()
        ).any()
        for name, component in case.varying_components
    )


def _compute_u(x1=0.0, x2=0.0):
    """Compute an output whose first component is constant.

    Args:
        x1: The first input.
        x2: The second input.

    Returns:
        The output.
    """
    u = array([1.0, x1 + x2, x1 - 2 * x2])
    return u  # noqa: RET504


@pytest.fixture(
    scope="module",
    params=[(CorrelationAnalysis, 100), (MorrisAnalysis, 0), (SobolAnalysis, 100)],
    ids=["CorrelationAnalysis", "MorrisAnalysis", "SobolAnalysis"],
)
def analysis_with_constant_first_component(
    request, random_space
) -> BaseSensitivityAnalysis:
    """An analysis whose output `u` is constant only for its first component."""
    analysis_class, n_samples = request.param
    return _compute_analysis(
        analysis_class, AutoPyDiscipline(_compute_u), random_space, n_samples
    )


@pytest.mark.parametrize(
    ("analysis_with_constant_first_component", "get_figure"),
    [
        ((CorrelationAnalysis, 100), lambda plot: plot.figures[0]),
        ((MorrisAnalysis, 0), lambda figure: figure),
        ((SobolAnalysis, 100), lambda figure: figure),
    ],
    ids=["CorrelationAnalysis", "MorrisAnalysis", "SobolAnalysis"],
    indirect=["analysis_with_constant_first_component"],
)
def test_plot_with_constant_first_component(
    analysis_with_constant_first_component, get_figure
) -> None:
    """Check that the plot of an output name uses its first varying component.

    The plot is either a dataset plot or a figure, according to the analysis.
    """
    plot = analysis_with_constant_first_component.plot("u", save=False)
    assert "u[1]" in get_figure(plot).axes[0].get_title()


@pytest.mark.parametrize(
    ("outputs", "expected_index"),
    [("u", ["u[1]", "u[2]"]), ([("u", 2)], ["u[2]"])],
    ids=["all_components", "last_component"],
)
def test_plot_bar_with_sorting_output_with_constant_first_component(
    analysis_with_constant_first_component, outputs, expected_index
) -> None:
    """Check that a sorting output name is sorted on its first plotted component.

    This component is the first varying one among the plotted ones.
    """
    dataset = analysis_with_constant_first_component.plot_bar(
        outputs=outputs, sorting_output="u", save=False
    ).dataset
    assert dataset.index.tolist() == expected_index
    values = dataset.loc[expected_index[0]].to_numpy()
    assert_array_equal(values, sorted(values, reverse=True))


def test_plot_field_with_constant_first_component(
    analysis_with_constant_first_component,
) -> None:
    """Check that a bare output name is sorted on its first varying component."""
    data = analysis_with_constant_first_component.plot_field(
        "u", save=False
    ).dataset.to_numpy()
    assert data.shape == (2, 3)
    assert isnan(data[:, 0]).all()
    assert not isnan(data[:, 1:]).any()


@pytest.mark.parametrize(
    ("kwargs", "expected_index", "expected_data"),
    [
        (
            {},
            ["x2[0]", "x2[1]", "x1"],
            [[2.5, 3.5], [2.75, 3.75], [2.25, 3.25]],
        ),
        ({"input_names": ["x2"]}, ["x2[0]", "x2[1]"], [[2.5, 3.5], [2.75, 3.75]]),
    ],
    ids=["all_inputs", "input_names"],
)
def test_plot_field_with_multi_component_input(
    mock_sensitivity_analysis, kwargs, expected_index, expected_data
) -> None:
    """Check that plot_field() plots every component of an input."""
    dataset = mock_sensitivity_analysis.plot_field("y2", save=False, **kwargs).dataset
    assert dataset.index.tolist() == expected_index
    assert_array_equal(dataset.to_numpy(), array(expected_data))


def test_plot_field_with_input_names_generator(mock_sensitivity_analysis) -> None:
    """Check that plot_field() accepts input names given as a generator."""
    dataset = mock_sensitivity_analysis.plot_field(
        "y2", input_names=(name for name in ["x1"]), save=False
    ).dataset
    assert dataset.index.tolist() == ["x1"]


@pytest.mark.parametrize("output", [("y2", 0), ["y2", 0]], ids=["tuple", "list"])
def test_plot_field_with_constant_output_component(
    constant_mock_sensitivity_analysis, output
) -> None:
    """Check that a constant output component is a gap in the plot of a field."""
    plot = constant_mock_sensitivity_analysis.plot_field(output, save=False)
    dataset = plot.dataset
    assert dataset.index.tolist() == ["x2[0]", "x2[1]", "x1"]
    data = dataset.to_numpy()
    assert_array_equal(data[:, 0], array([2.5, 2.75, 2.25]))
    assert isnan(data[:, 1]).all()


@pytest.mark.parametrize(
    "output", ["y1", ("y2", 1)], ids=["all_components", "sorting_component"]
)
def test_plot_field_with_constant_output(
    constant_mock_sensitivity_analysis, output, snapshot
) -> None:
    """Check that plot_field raises for a constant output component.

    The output is either a name whose components are all constant
    or a constant component used to sort the inputs.
    """
    with assert_exception(ValueError, snapshot):
        constant_mock_sensitivity_analysis.plot_field(output, save=False)


def test_plot_field_without_input(mock_sensitivity_analysis, snapshot) -> None:
    """Check that plot_field() raises when none of the input names are inputs."""
    with assert_exception(ValueError, snapshot):
        mock_sensitivity_analysis.plot_field("y2", input_names=["nope"], save=False)


def test_sort_input_variables_with_constant_first_component(
    analysis_with_constant_first_component,
) -> None:
    """Check that sort_input_variables() uses the first varying component."""
    analysis = analysis_with_constant_first_component
    assert analysis.sort_input_variables("u") == analysis.sort_input_variables(("u", 1))


@pytest.mark.parametrize(
    "analysis_with_constant_first_component",
    [(SobolAnalysis, 100)],
    indirect=True,
    ids=["SobolAnalysis"],
)
def test_plot_second_order_with_constant_first_component(
    analysis_with_constant_first_component,
) -> None:
    """Check that plot_second_order() uses the first non-constant component."""
    analysis = analysis_with_constant_first_component
    dataset = analysis.plot_second_order("u").dataset
    assert_frame_equal(dataset, analysis.plot_second_order(("u", 1)).dataset)


@pytest.mark.parametrize("output", ["y2", ("y2", 0)], ids=["name", "component"])
def test_plot_comparison_without_common_defined_component(
    constant_mock_sensitivity_analysis,
    first_component_constant_mock_sensitivity_analysis,
    output,
    snapshot,
) -> None:
    """Check that plot_comparison() raises without a component defined for all."""
    with assert_exception(ValueError, snapshot):
        constant_mock_sensitivity_analysis.plot_comparison(
            first_component_constant_mock_sensitivity_analysis, output, save=False
        )


@pytest.mark.parametrize(
    "analysis_fixture",
    ["mock_sensitivity_analysis", "scalar_mock_sensitivity_analysis"],
    ids=["mock", "scalar_mock"],
)
@pytest.mark.parametrize(
    "sorting_output",
    ["y2", ("y2", 0), "typo", ("typo", 0)],
    ids=["name", "tuple", "unknown_name", "unknown_tuple"],
)
def test_plot_bar_with_sorting_output_not_plotted(
    request, analysis_fixture, sorting_output, snapshot
) -> None:
    """Check that plot_bar() raises when the sorting output is not plotted."""
    analysis = request.getfixturevalue(analysis_fixture)
    with assert_exception(ValueError, snapshot):
        analysis.plot_bar(outputs="y1", sorting_output=sorting_output, save=False)


def _compute_c_and_v(x1=0.0, x2=0.0):
    """Compute an output with eleven constant components and a varying output.

    Args:
        x1: The first input.
        x2: The second input.

    Returns:
        The constant output and the varying one.
    """
    c = array([1.0] * 11)
    v = x1 + x2
    return c, v


@pytest.fixture(scope="module")
def analysis_with_multi_component_constant_output(
    random_space,
) -> CorrelationAnalysis:
    """An analysis whose output `c` has eleven constant components."""
    return _compute_analysis(
        CorrelationAnalysis, AutoPyDiscipline(_compute_c_and_v), random_space, 20
    )


@pytest.mark.parametrize(
    "outputs",
    ["c", [("c", 2), ("c", 10)], [("c", 2), ("c", 10), ("c", 2)]],
    ids=["all_components", "components", "duplicate_components"],
)
def test_plot_bar_with_multi_component_constant_output(
    analysis_with_multi_component_constant_output, outputs, snapshot
) -> None:
    """Check the error message for constant components of a multi-component output.

    An output whose components are all constant is named once
    and the constant components are listed once in the given order.
    """
    with assert_exception(ValueError, snapshot):
        analysis_with_multi_component_constant_output.plot_bar(
            outputs=outputs, save=False
        )


@pytest.mark.parametrize("plot_name", ["plot_bar", "plot_radar"])
def test_plot_duplicate_output(
    analysis_with_multi_component_constant_output, plot_name
) -> None:
    """Check that an output component listed twice is plotted once."""
    dataset = getattr(analysis_with_multi_component_constant_output, plot_name)(
        outputs=["v", ("v", 0)], save=False
    ).dataset
    assert dataset.index.tolist() == ["v"]


def test_plot_field_standardizes_only_the_output(
    mock_sensitivity_analysis, monkeypatch
) -> None:
    """Check that plot_field() standardizes only the indices of the output."""
    standardize_indices = mock_sensitivity_analysis.standardize_indices
    output_names = []

    def spy(indices):
        output_names.append(list(indices))
        return standardize_indices(indices)

    monkeypatch.setattr(mock_sensitivity_analysis, "standardize_indices", spy)
    dataset = mock_sensitivity_analysis.plot_field(
        "y2", standardize=True, save=False
    ).dataset
    assert output_names == [["y2"]]
    assert_array_equal(dataset.to_numpy().max(axis=0), array([1.0, 1.0]))


def _compute_field(x1=0.0, x2=0.0):
    """Compute a 2D field whose first component is constant.

    Args:
        x1: The first input.
        x2: The second input.

    Returns:
        The field.
    """
    field = array([1.0, x1, x2, x1 + x2, x1 - x2])
    return field  # noqa: RET504


def test_plot_field_2d_with_constant_component(random_space) -> None:
    """Check that a 2D field with a constant component is plotted with a gap."""
    analysis = _compute_analysis(
        CorrelationAnalysis, AutoPyDiscipline(_compute_field), random_space, 20
    )
    mesh = array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    plot = analysis.plot_field("field", mesh=mesh, save=False)
    assert isnan(plot.dataset.to_numpy()[:, 0]).all()
    # Each figure has a surface and its colorbar.
    assert [len(figure.axes) for figure in plot.figures] == [2, 2]


class ComparisonCase(NamedTuple):
    """Two analyses to compare and the output component used for a bare name."""

    analysis: BaseSensitivityAnalysis
    """The current analysis."""

    other_analysis: BaseSensitivityAnalysis
    """The compared analysis."""

    name: str
    """The name of the output."""

    component: int
    """The component of the output that `plot_comparison()` must use for `name`."""


@pytest.fixture(
    params=[
        "fewer_components",
        "different_constant_components",
        CorrelationAnalysis,
        MorrisAnalysis,
        SobolAnalysis,
    ],
    ids=[
        "fewer_components",
        "different_constant_components",
        "CorrelationAnalysis",
        "MorrisAnalysis",
        "SobolAnalysis",
    ],
)
def comparison_case(request) -> ComparisonCase:
    """Two analyses whose first component of an output is not defined for both."""
    if request.param == "fewer_components":
        # The output y2 has two components for the current analysis
        # and one for the compared one.
        with concretize_classes(MockSensitivityAnalysis):
            analysis = MockSensitivityAnalysis()

        return ComparisonCase(
            analysis, _create_one_component_mock_sensitivity_analysis(), "y2", 0
        )

    if request.param == "different_constant_components":
        # The first component of y2 is defined for the current analysis only.
        with concretize_classes(MockSensitivityAnalysis):
            analysis = MockSensitivityAnalysis()

        other_analysis = request.getfixturevalue(
            "first_component_constant_mock_sensitivity_analysis"
        )
        return ComparisonCase(analysis, other_analysis, "y2", 1)

    # The first component of the output u is constant.
    n_samples = 0 if request.param is MorrisAnalysis else 100
    analysis = _compute_analysis(
        request.param,
        AutoPyDiscipline(_compute_u),
        request.getfixturevalue("random_space"),
        n_samples,
    )
    return ComparisonCase(analysis, analysis, "u", 1)


def test_plot_comparison_with_bare_output_name(comparison_case) -> None:
    """Check that plot_comparison() uses the first component defined for all."""
    analysis, other_analysis, name, component = comparison_case
    dataset = analysis.plot_comparison(other_analysis, name, save=False).dataset
    expected_dataset = analysis.plot_comparison(
        other_analysis, (name, component), save=False
    ).dataset
    assert_frame_equal(dataset, expected_dataset)


@pytest.mark.parametrize("current", [False, True], ids=["compared", "current"])
@pytest.mark.parametrize("output", ["y2", ("y2", 0)], ids=["name", "component"])
def test_plot_comparison_with_missing_output(
    mock_sensitivity_analysis, current, output, snapshot
) -> None:
    """Check that plot_comparison() raises when an analysis lacks the output."""
    analysis_without_y2 = _create_mock_sensitivity_analysis(
        "WithoutY2MockSensitivityAnalysis",
        {"y1": [{"x1": array([1.25]), "x2": array([1.5, 1.75])}]},
    )
    if current:
        analysis = analysis_without_y2
        other_analyses = mock_sensitivity_analysis
    else:
        analysis = mock_sensitivity_analysis
        other_analyses = [mock_sensitivity_analysis, analysis_without_y2]

    with assert_exception(ValueError, snapshot):
        analysis.plot_comparison(other_analyses, output, save=False)


@pytest.mark.parametrize(
    ("current", "component"),
    [(False, 1), (True, 1), (False, -1), (False, -2)],
    ids=["compared", "current", "negative", "negative-beyond-compared"],
)
def test_plot_comparison_with_component_out_of_range(
    mock_sensitivity_analysis, current, component, snapshot
) -> None:
    """Check that plot_comparison() raises for a component out of range.

    The output `y2` has two components for one of the analyses
    and a single one for the other.
    """
    one_component_analysis = _create_one_component_mock_sensitivity_analysis()
    analyses = [mock_sensitivity_analysis, one_component_analysis]
    if current:
        analyses.reverse()

    analysis, other_analysis = analyses
    with assert_exception(ValueError, snapshot):
        analysis.plot_comparison(other_analysis, ("y2", component), save=False)


class AlphabeticalSortingMockSensitivityAnalysis(MockSensitivityAnalysis):
    """This sensitivity analysis sorts the inputs in alphabetical order."""

    def sort_input_variables(self, output: VariableOrComponent) -> list[str]:
        return ["x1", "x2"]


@pytest.fixture
def alphabetical_sorting_analysis() -> AlphabeticalSortingMockSensitivityAnalysis:
    """A sensitivity analysis sorting the inputs in alphabetical order."""
    with concretize_classes(AlphabeticalSortingMockSensitivityAnalysis):
        return AlphabeticalSortingMockSensitivityAnalysis()


def test_plot_comparison_with_overridden_sort_input_variables(
    alphabetical_sorting_analysis,
) -> None:
    """Check that plot_comparison() sorts the inputs with sort_input_variables()."""
    analysis = alphabetical_sorting_analysis
    dataset = analysis.plot_comparison(analysis, "y1", save=False).dataset
    assert dataset.get_variable_names(dataset.parameter_group) == ["x1", "x2"]


def test_plot_field_with_overridden_sort_input_variables(
    alphabetical_sorting_analysis,
) -> None:
    """Check that plot_field() sorts the inputs with sort_input_variables()."""
    plot = alphabetical_sorting_analysis.plot_field("y2", save=False)
    assert plot.dataset.index.tolist() == ["x1", "x2[0]", "x2[1]"]


@pytest.mark.parametrize(
    "method_name",
    [
        "sort_input_variables",
        "plot",
        "plot_bar",
        "plot_radar",
        "plot_field",
        "plot_comparison",
    ],
)
@pytest.mark.parametrize(
    "output",
    [("y2", -1), ("y2", 2), "typo", ("typo", 0)],
    ids=["negative", "too-large", "unknown-name", "unknown-name-component"],
)
def test_methods_with_invalid_output(
    mock_sensitivity_analysis, method_name, output, snapshot
) -> None:
    """Check that every method taking an output component rejects an invalid one."""
    kwargs = {} if method_name == "sort_input_variables" else {"save": False}
    if method_name in {"plot_bar", "plot_radar"}:
        kwargs["outputs"] = output
        args = ()
    elif method_name == "plot_comparison":
        args = (mock_sensitivity_analysis, output)
    else:
        args = (output,)

    with assert_exception(ValueError, snapshot):
        getattr(mock_sensitivity_analysis, method_name)(*args, **kwargs)


@pytest.mark.parametrize(
    "analysis_class", [CorrelationAnalysis, SobolAnalysis, MorrisAnalysis]
)
@pytest.mark.parametrize(
    "output", [("out", -1), ("out", 1), ("typo", 0)], ids=["negative", "large", "typo"]
)
def test_plot_with_invalid_output(
    analysis_class, output, discipline, parameter_space, snapshot
) -> None:
    """Check that the plot() of the analyses rejects an invalid output component."""
    analysis = _compute_analysis(analysis_class, discipline, parameter_space, 20)
    with assert_exception(ValueError, snapshot):
        analysis.plot(output, save=False)
