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
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Callable
from typing import Union

import pytest
from matplotlib.figure import Figure
from numpy import array
from numpy import isnan
from numpy import ndarray
from numpy import pi
from numpy import sign
from numpy import sin
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.uncertainty.sensitivity.base_sensitivity_analysis import (
    FirstOrderIndicesType,
)
from gemseo.uncertainty.sensitivity.base_sensitivity_analysis import (
    SecondOrderIndicesType,
)
from gemseo.uncertainty.sensitivity.sobol_analysis import SobolAnalysis
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo.utils.testing.helpers import image_comparison

if TYPE_CHECKING:
    from gemseo.core.discipline import Discipline

StatisticsType = tuple[
    dict[str, NDArray[float]],
    dict[str, Union[FirstOrderIndicesType, SecondOrderIndicesType]],
]


@pytest.fixture(scope="module")
def py_func() -> Callable[[ndarray, ndarray], tuple[ndarray, ndarray]]:
    """The Ishigami function."""

    def ishigami(x1, x23):
        y = array([sin(x1[0]) + 7 * sin(x23[0]) ** 2 + 0.1 * x23[1] ** 4 * sin(x1[0])])
        z = array([y[0], y[0]])
        return y, z

    return ishigami


@pytest.fixture(scope="module")
def discipline(
    py_func: Callable[[ndarray, ndarray], tuple[ndarray, ndarray]],
) -> AutoPyDiscipline:
    """The discipline of interest."""
    return AutoPyDiscipline(py_func=py_func, use_arrays=True)


@pytest.fixture(scope="module")
def discipline_cv1() -> AutoPyDiscipline:
    """A first CV discipline."""

    def cv1(x1, x23):
        sin_x1_tp = x1[0] - x1[0] ** 3 / 6
        sin_x2_tp = x23[0] - x23[0] ** 3 / 6 + x23[0] ** 5 / 120
        y = array([sin_x1_tp + 7 * sin_x2_tp**2 + 0.1 * x23[1] ** 4 * sin_x1_tp])
        z = array([y[0], y[0]])
        return y, z

    return AutoPyDiscipline(py_func=cv1, use_arrays=True)


@pytest.fixture(scope="module")
def discipline_cv2() -> AutoPyDiscipline:
    """A second CV discipline."""

    def cv2(x1, x23):
        sin_x1_tp = x1[0] - x1[0] ** 3 / 6 + x1[0] ** 5 / 120
        sin_x2_tp = x23[0] - x23[0] ** 3 / 6
        y = array([sin_x1_tp + 7 * sin_x2_tp**2 + 0.1 * x23[1] ** 4 * sin_x1_tp])
        z = array([y[0], y[0]])
        return y, z

    return AutoPyDiscipline(py_func=cv2, use_arrays=True)


@pytest.fixture(scope="module")
def uncertain_space() -> ParameterSpace:
    """The uncertain space of interest."""
    parameter_space = ParameterSpace()
    for name, size in zip(["x1", "x23"], [1, 2]):
        parameter_space.add_random_variable(
            name, "OTUniformDistribution", minimum=-pi, maximum=pi, size=size
        )
    return parameter_space


@pytest.fixture
def sobol(discipline: Discipline, uncertain_space: ParameterSpace) -> SobolAnalysis:
    """A Sobol' analysis."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], uncertain_space, 100)
    analysis.compute_indices()
    return analysis


@pytest.fixture
def first_intervals(sobol: SobolAnalysis) -> FirstOrderIndicesType:
    """The intervals of the first-order indices."""
    return sobol.get_intervals()


@pytest.fixture
def total_intervals(sobol: SobolAnalysis) -> FirstOrderIndicesType:
    """The intervals of the total-order indices."""
    return sobol.get_intervals(False)


@pytest.fixture(scope="module")
def cv1_stat(
    discipline_cv1: Discipline,
    uncertain_space: ParameterSpace,
) -> StatisticsType:
    """The estimated output variance and Sobol' indices.

    Here for the first CV discipline.
    """
    sobol_analysis = SobolAnalysis()
    sobol_analysis.compute_samples([discipline_cv1], uncertain_space, 100)
    return sobol_analysis.output_variances, sobol_analysis.compute_indices()


@pytest.fixture(scope="module")
def cv2_stat(
    discipline_cv2: Discipline,
    uncertain_space: ParameterSpace,
) -> StatisticsType:
    """The estimated output variance and Sobol' indices.

    Here for the second CV discipline.
    """
    sobol_analysis = SobolAnalysis()
    sobol_analysis.compute_samples([discipline_cv2], uncertain_space, 100)
    return sobol_analysis.output_variances, sobol_analysis.compute_indices()


@pytest.fixture(scope="module")
def sobol_cv(
    discipline: Discipline,
    uncertain_space: ParameterSpace,
    discipline_cv1: Discipline,
    cv1_stat: StatisticsType,
) -> SobolAnalysis:
    """A Sobol' analysis when a control variate is used."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], uncertain_space, 100)
    cv1 = analysis.ControlVariate(
        discipline=discipline_cv1,
        indices=cv1_stat[1],
        variance=cv1_stat[0],
    )
    analysis.compute_indices(control_variates=[cv1])
    return analysis


@pytest.fixture(scope="module")
def first_intervals_cv(sobol_cv: SobolAnalysis) -> FirstOrderIndicesType:
    """The intervals of the first-order indices."""
    return sobol_cv.get_intervals()


@pytest.fixture(scope="module")
def total_intervals_cv(sobol_cv: SobolAnalysis) -> FirstOrderIndicesType:
    """The intervals of the total-order indices."""
    return sobol_cv.get_intervals(False)


def test_algo(discipline, uncertain_space) -> None:
    """Check that algorithm can be passed either as a str or an Algorithm."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], uncertain_space, 100)
    assert compare_dict_of_arrays(
        analysis.compute_indices(algo=analysis.Algorithm.JANSEN).first["y"][0],
        analysis.compute_indices(algo="Jansen").first["y"][0],
    )


@pytest.mark.parametrize("method", ["total", SobolAnalysis.Method.TOTAL])
def test_method(sobol, method) -> None:
    """Check the use of the main method."""
    assert sobol.main_method == "first"
    assert compare_dict_of_arrays(
        sobol.main_indices["y"][0], sobol.indices.first["y"][0], 0.1
    )

    sobol.main_method = method
    assert sobol.main_method == "total"
    assert compare_dict_of_arrays(
        sobol.main_indices["y"][0], sobol.indices.total["y"][0], 0.1
    )

    sobol.main_method = SobolAnalysis.Method.FIRST


@pytest.mark.parametrize(
    ("name", "bound", "expected"),
    [
        ("x1", 0, [-0.3]),
        ("x23", 0, [-0.3, -1.3]),
        ("x1", 1, [0.1]),
        ("x23", 1, [0.1, 0.2]),
    ],
)
def test_first_intervals(first_intervals, name, bound, expected) -> None:
    """Check the values of the intervals for the first-order indices."""
    assert_almost_equal(
        first_intervals["y"][0][name][bound], array(expected), decimal=1
    )


@pytest.mark.parametrize(
    ("name", "bound", "expected"),
    [
        ("x1", 0, [0.1]),
        ("x23", 0, [0.3, -0.2]),
        ("x1", 1, [1.2]),
        ("x23", 1, [0.7, 0.9]),
    ],
)
def test_total_intervals(total_intervals, name, bound, expected) -> None:
    """Check the values of the intervals for the total-order indices."""
    assert_almost_equal(
        total_intervals["y"][0][name][bound], array(expected), decimal=1
    )


@pytest.mark.parametrize(
    ("name", "sort", "sort_by_total", "kwargs", "baseline_images"),
    [
        ("y", False, False, {}, ["plot"]),
        ("y", False, False, {"title": "foo"}, ["plot_title"]),
        ("y", True, False, {}, ["plot_sort_by_first"]),
        ("y", True, True, {}, ["plot_sort_by_total"]),
        ("z", False, False, {}, ["plot_name"]),
        (("z", 1), False, False, {}, ["plot_name_component"]),
    ],
)
@image_comparison(None)
def test_plot(name, sobol, sort, sort_by_total, kwargs, baseline_images) -> None:
    """Check the main visualization method."""
    fig = sobol.plot(name, save=False, sort=sort, sort_by_total=sort_by_total, **kwargs)
    assert isinstance(fig, Figure)


@pytest.mark.parametrize(
    ("order", "reference"),
    [
        (
            "first",
            {"x1": array([-0.06]), "x23": array([-0.10, -0.53])},
        ),
        (
            "second",
            {
                "x1": {"x1": array([[0.0]]), "x23": array([[0.79, 1.45]])},
                "x23": {
                    "x1": array([[0.79], [1.45]]),
                    "x23": array([[0.0, 0.97], [0.97, 0.0]]),
                },
            },
        ),
        (
            "total",
            {"x1": array([0.63]), "x23": array([0.48, 0.38])},
        ),
    ],
)
def test_indices(sobol, order, reference) -> None:
    """Check the values of the indices."""
    assert compare_dict_of_arrays(getattr(sobol.indices, order)["y"][0], reference, 0.1)


@pytest.mark.parametrize("compute_second_order", [False, True])
def test_second_order(discipline, uncertain_space, compute_second_order) -> None:
    """Check the computation of second-order indices."""
    analysis = SobolAnalysis()
    analysis.compute_samples(
        [discipline], uncertain_space, 100, compute_second_order=compute_second_order
    )
    analysis.compute_indices()
    assert bool(analysis.indices.second) is compute_second_order
    assert len(analysis.dataset) == (96 if compute_second_order else 100)


def test_asymptotic_or_bootstrap_intervals(discipline, uncertain_space) -> None:
    """Check the method to compute the confidence intervals."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], uncertain_space, 100)
    analysis.compute_indices()
    asymptotic_interval = analysis.get_intervals()["y"][0]["x1"]

    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], uncertain_space, 100)
    analysis.compute_indices(use_asymptotic_distributions=False)
    bootstrap_interval = analysis.get_intervals()["y"][0]["x1"]

    assert asymptotic_interval[0][0] != bootstrap_interval[0][0]
    assert asymptotic_interval[1][0] != bootstrap_interval[1][0]


def test_confidence_level_default(discipline, uncertain_space) -> None:
    """Check the default confidence level used by the algorithm."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], uncertain_space, 100)
    analysis.compute_indices()
    algos = analysis.dataset.misc["output_names_to_sobol_algos"]
    assert algos["y"][0].getConfidenceLevel() == 0.95


def test_confidence_level_custom(discipline, uncertain_space) -> None:
    """Check setting a custom confidence level."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], uncertain_space, 100)
    analysis.compute_indices(confidence_level=0.90)
    algos = analysis.dataset.misc["output_names_to_sobol_algos"]
    assert algos["y"][0].getConfidenceLevel() == 0.90


def test_output_variances(sobol) -> None:
    """Check SobolAnalysis.output_variances."""
    dataset = sobol.dataset
    assert compare_dict_of_arrays(
        sobol.output_variances,
        {
            name: dataset.get_view(variable_names=name)
            .to_numpy()[: len(dataset) // 8 * 2]
            .var(0)
            for name in ["y", "z"]
        },
        tolerance=0.1,
    )


def test_output_standard_deviations(sobol) -> None:
    """Check SobolAnalysis.output_standard_deviations."""
    dataset = sobol.dataset
    assert compare_dict_of_arrays(
        sobol.output_standard_deviations,
        {
            name: dataset.get_view(variable_names=name)
            .to_numpy()[: len(dataset) // 8 * 2]
            .std(0)
            for name in ["y", "z"]
        },
        tolerance=0.1,
    )


@pytest.mark.parametrize("use_variance", [False, True])
@pytest.mark.parametrize("order", ["first", "second", "total"])
def test_unscale_indices(sobol, use_variance, order) -> None:
    """Check SobolAnalysis.unscaled_indices()."""
    orders_to_indices = {
        "first": sobol.indices.first,
        "second": sobol.indices.second,
        "total": sobol.indices.total,
    }
    is_second_order = order == "second"
    indices = orders_to_indices[order]

    def f(x):
        return (
            {
                k: v * sobol.output_variances[x[0]][x[1]]
                for k, v in indices[x[0]][x[1]][x[2]].items()
            }
            if is_second_order
            else indices[x[0]][x[1]][x[2]] * sobol.output_variances[x[0]][x[1]]
        )

    expected = {
        "y": [{"x1": f(("y", 0, "x1")), "x23": f(("y", 0, "x23"))}],
        "z": [
            {"x1": f(("z", 0, "x1")), "x23": f(("z", 0, "x23"))},
            {"x1": f(("z", 1, "x1")), "x23": f(("z", 1, "x23"))},
        ],
    }
    if not use_variance:
        expected = {
            output_name: [
                {
                    input_name: (
                        {
                            k: sign(v) * (sign(v) * v) ** 0.5
                            for k, v in sobol_index.items()
                        }
                        if is_second_order
                        else sign(sobol_index)
                        * (sign(sobol_index) * sobol_index) ** 0.5
                    )
                    for input_name, sobol_index in output_value.items()
                }
                for output_value in output_values
            ]
            for output_name, output_values in expected.items()
        }

    unscaled_indices = sobol.unscale_indices(indices, use_variance=use_variance)
    for output_name, output_values in expected.items():
        unscaled_indices_ = unscaled_indices[output_name]
        for output_index, output_value in enumerate(output_values):
            assert compare_dict_of_arrays(
                unscaled_indices_[output_index], output_value, tolerance=0.1
            )


def test_unscale_index_with_negative_values(sobol):
    """Check that negative estimates do not include nan."""
    f = sobol._SobolAnalysis__unscale_index
    assert not any(isnan(f(array([0.23, -0.87]), "y", 0, False)))
    assert not any(isnan(f({"foo": array([0.23, -0.87])}, "y", 0, False)["foo"]))


def test_compute_indices_output_names(sobol) -> None:
    """Check compute_indices with different types for output_names."""
    assert sobol.compute_indices(["y"]).first
    assert sobol.compute_indices("y").first


def test_to_dataset(sobol) -> None:
    """Check that the second-order indices are stored in Dataset.misc."""
    dataset = sobol.to_dataset()
    assert "first" in dataset.group_names
    assert "total" in dataset.group_names
    assert "second" not in dataset.group_names
    assert "second" in dataset.misc


def test_output_variance_cv(
    sobol,
    discipline_cv1,
    cv1_stat,
) -> None:
    """Check the values of the variances computed with control variates."""
    decimal = 2
    cv1 = sobol.ControlVariate(
        discipline=discipline_cv1,
        variance=cv1_stat[0],
        indices=cv1_stat[1],
    )
    assert_almost_equal(sobol.output_variances["y"][0], 20.91, decimal=decimal)
    sobol.compute_indices(["y"], control_variates=[cv1])
    assert_almost_equal(sobol.output_variances["y"][0], 21.00, decimal=decimal)
    assert_almost_equal(sobol.output_variances["z"][0], 20.91, decimal=decimal)


def test_cv_wo_statistics(
    sobol,
    discipline_cv1,
    cv1_stat,
    uncertain_space,
) -> None:
    """Check the use of control variates without cv statistics."""
    cv1_variance, cv1_indices = cv1_stat
    cv = sobol.ControlVariate(
        discipline=discipline_cv1,
        indices=None,
        n_samples=100,
        variance=None,
    )
    cv = sobol._SobolAnalysis__compute_cv_stats(cv)
    assert cv.indices is not None
    assert cv.variance is not None

    cv.variance = None
    cv.indices = cv1_indices
    cv = sobol._SobolAnalysis__compute_cv_stats(cv)
    assert cv.indices is not None
    assert cv.indices != cv1_indices
    assert cv.variance is not None

    cv.variance = cv1_variance
    cv.indices = None
    cv.n_samples = 0
    cv = sobol._SobolAnalysis__compute_cv_stats(cv)
    assert cv.indices is not None
    assert cv.variance is not None
    assert cv.variance != cv1_variance


@pytest.mark.parametrize(
    ("order", "reference_cv1", "reference_cv11", "reference_cv12"),
    [
        (
            "first",
            {"x1": array([0.840]), "x23": array([0.051, -0.039])},
            {"x1": array([0.896]), "x23": array([0.051, -0.055])},
            {"x1": array([0.321]), "x23": array([0.037, 0.069])},
        ),
        (
            "total",
            {"x1": array([0.449]), "x23": array([0.700, 0.423])},
            {"x1": array([0.449]), "x23": array([0.249, 0.320])},
            {"x1": array([-0.305]), "x23": array([0.679, 0.063])},
        ),
    ],
)
def test_cv_algo(
    sobol,
    discipline_cv1,
    discipline_cv2,
    cv1_stat,
    cv2_stat,
    order,
    reference_cv1,
    reference_cv11,
    reference_cv12,
) -> None:
    """Check the values of the indices computed with control variates."""
    output_name = "z"
    cv1 = sobol.ControlVariate(
        discipline=discipline_cv1,
        variance=cv1_stat[0],
        indices=cv1_stat[1],
    )
    cv2 = sobol.ControlVariate(
        discipline=discipline_cv2,
        variance=cv2_stat[0],
        indices=cv2_stat[1],
    )
    indices_cv1 = sobol.compute_indices([output_name], control_variates=cv1)
    indices_cv11 = sobol.compute_indices([output_name], control_variates=[cv1, cv1])
    indices_cv12 = sobol.compute_indices([output_name], control_variates=[cv1, cv2])
    assert compare_dict_of_arrays(
        getattr(indices_cv1, order)["z"][0], reference_cv1, 0.001
    )
    assert compare_dict_of_arrays(
        getattr(indices_cv11, order)["z"][0], reference_cv11, 0.05
    )
    assert compare_dict_of_arrays(
        getattr(indices_cv12, order)["z"][0], reference_cv12, 0.001
    )


@pytest.mark.parametrize(
    "baseline_images",
    [["plot_cv"]],
)
@image_comparison(None)
def test_plot_cv(sobol_cv, baseline_images) -> None:
    """Check the main visualization method when a control variate is used."""
    fig = sobol_cv.plot("y", save=False, sort=False)
    assert isinstance(fig, Figure)


@pytest.mark.parametrize(
    ("name", "bound", "expected"),
    [
        ("x1", 0, [-0.54]),
        ("x23", 0, [-0.08, -0.37]),
        ("x1", 1, [1.87]),
        ("x23", 1, [0.19, 1.15]),
    ],
)
def test_first_intervals_cv(first_intervals_cv, name, bound, expected) -> None:
    """Check the values of the intervals for the first-order indices when a control
    variate is used."""
    assert_almost_equal(
        first_intervals_cv["y"][0][name][bound], array(expected), decimal=2
    )


@pytest.mark.parametrize(
    ("name", "bound", "expected"),
    [
        ("x1", 0, [0.16]),
        ("x23", 0, [0.53, 0.12]),
        ("x1", 1, [0.89]),
        ("x23", 1, [0.74, 0.74]),
    ],
)
def test_total_intervals_cv(total_intervals_cv, name, bound, expected) -> None:
    """Check the values of the intervals for the total-order indices when a control
    variate is used."""
    assert_almost_equal(
        total_intervals_cv["y"][0][name][bound], array(expected), decimal=2
    )


def test_from_samples(sobol, tmp_wd):
    """Check the instantiation from samples."""
    file_path = Path("samples.pkl")
    sobol.dataset.to_pickle(file_path)
    new_sobol = SobolAnalysis(samples=file_path)
    new_sobol.compute_indices()
    assert new_sobol.indices.first["y"][0]["x1"] == sobol.indices.first["y"][0]["x1"]


def test_from_samples_cv(sobol_cv, discipline_cv1, cv1_stat, tmp_wd):
    """Check the instantiation from samples with control variates.."""
    file_path = Path("samples.pkl")
    sobol_cv.dataset.to_pickle(file_path)
    new_sobol_cv = SobolAnalysis(samples=file_path)
    cv1 = new_sobol_cv.ControlVariate(
        discipline=discipline_cv1,
        indices=cv1_stat[1],
        variance=cv1_stat[0],
    )
    new_sobol_cv.compute_indices(control_variates=[cv1])
    assert not new_sobol_cv.indices.second
    assert (
        new_sobol_cv.indices.first["y"][0]["x1"] == sobol_cv.indices.first["y"][0]["x1"]
    )
