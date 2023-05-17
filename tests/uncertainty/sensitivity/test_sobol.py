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

from typing import Callable

import pytest
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.uncertainty.sensitivity.analysis import FirstOrderIndicesType
from gemseo.uncertainty.sensitivity.sobol.analysis import SobolAnalysis
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo.utils.testing.helpers import image_comparison
from numpy import array
from numpy import ndarray
from numpy import pi
from numpy import sin
from numpy.testing import assert_almost_equal


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
    py_func: Callable[[ndarray, ndarray], tuple[ndarray, ndarray]]
) -> AutoPyDiscipline:
    """The discipline of interest."""
    return AutoPyDiscipline(py_func=py_func, use_arrays=True)


@pytest.fixture(scope="module")
def uncertain_space() -> ParameterSpace:
    """The uncertain space of interest."""
    parameter_space = ParameterSpace()
    for name, size in zip(["x1", "x23"], [1, 2]):
        parameter_space.add_random_variable(
            name, "OTUniformDistribution", minimum=-pi, maximum=pi, size=size
        )
    return parameter_space


@pytest.fixture(scope="module")
def sobol(discipline: MDODiscipline, uncertain_space: ParameterSpace) -> SobolAnalysis:
    """A Sobol' analysis."""
    analysis = SobolAnalysis([discipline], uncertain_space, 100)
    analysis.compute_indices()
    return analysis


@pytest.fixture(scope="module")
def first_intervals(sobol: SobolAnalysis) -> FirstOrderIndicesType:
    """The intervals of the first-order indices."""
    return sobol.get_intervals()


@pytest.fixture(scope="module")
def total_intervals(sobol: SobolAnalysis) -> FirstOrderIndicesType:
    """The intervals of the total-order indices."""
    return sobol.get_intervals(False)


def test_algo(discipline, uncertain_space):
    """Check that algorithm can be passed either as a str or an Algorithm."""
    analysis = SobolAnalysis([discipline], uncertain_space, 100)
    indices = analysis.compute_indices(algo=analysis.Algorithm.JANSEN)["first"]["y"][0]
    assert compare_dict_of_arrays(
        indices, analysis.compute_indices(algo="Jansen")["first"]["y"][0]
    )


@pytest.mark.parametrize("method", ["total", SobolAnalysis.Method.TOTAL])
def test_method(sobol, method):
    """Check the use of the main method."""
    assert sobol.main_method == "first"
    assert compare_dict_of_arrays(
        sobol.main_indices["y"][0], sobol.indices["first"]["y"][0], 0.1
    )

    sobol.main_method = method
    assert sobol.main_method == "total"
    assert compare_dict_of_arrays(
        sobol.main_indices["y"][0], sobol.indices["total"]["y"][0], 0.1
    )

    sobol.main_method = SobolAnalysis.Method.FIRST


@pytest.mark.parametrize(
    "name,bound,expected",
    [
        ("x1", 0, [-0.3]),
        ("x23", 0, [-0.3, -1.3]),
        ("x1", 1, [0.1]),
        ("x23", 1, [0.1, 0.2]),
    ],
)
def test_first_intervals(first_intervals, name, bound, expected):
    """Check the values of the intervals for the first-order indices."""
    assert_almost_equal(
        first_intervals["y"][0][name][bound], array(expected), decimal=1
    )


@pytest.mark.parametrize(
    "name,bound,expected",
    [
        ("x1", 0, [0.1]),
        ("x23", 0, [0.3, -0.2]),
        ("x1", 1, [1.2]),
        ("x23", 1, [0.7, 0.9]),
    ],
)
def test_total_intervals(total_intervals, name, bound, expected):
    """Check the values of the intervals for the total-order indices."""
    assert_almost_equal(
        total_intervals["y"][0][name][bound], array(expected), decimal=1
    )


@pytest.mark.parametrize(
    "name,sort,sort_by_total,kwargs,baseline_images",
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
def test_plot(
    name, sobol, sort, sort_by_total, kwargs, baseline_images, pyplot_close_all
):
    """Check the main visualization method."""
    sobol.plot(name, save=False, sort=sort, sort_by_total=sort_by_total, **kwargs)


@pytest.mark.parametrize(
    "order,reference",
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
def test_indices(sobol, order, reference):
    """Check the values of the indices."""
    assert compare_dict_of_arrays(sobol.indices[order]["y"][0], reference, 0.1)
    assert compare_dict_of_arrays(
        getattr(sobol, f"{order.lower()}_order_indices")["y"][0], reference, 0.1
    )


def test_save_load(sobol, tmp_wd):
    """Check saving and loading a SobolAnalysis."""
    sobol.to_pickle("foo.pkl")
    new_sobol = SobolAnalysis.from_pickle("foo.pkl")
    assert new_sobol.dataset.equals(sobol.dataset)
    assert new_sobol.default_output == sobol.default_output


@pytest.mark.parametrize("compute_second_order", [False, True])
def test_second_order(discipline, uncertain_space, compute_second_order):
    """Check the computation of second-order indices."""
    analysis = SobolAnalysis(
        [discipline], uncertain_space, 100, compute_second_order=compute_second_order
    )
    analysis.compute_indices()
    assert bool(analysis.indices["second"]) is compute_second_order
    assert bool(analysis.second_order_indices) is compute_second_order
    assert len(analysis.dataset) == (96 if compute_second_order else 100)


def test_asymptotic_or_bootstrap_intervals(discipline, uncertain_space):
    """Check the method to compute the confidence intervals."""
    analysis = SobolAnalysis([discipline], uncertain_space, 100)
    analysis.compute_indices()
    asymptotic_interval = analysis.get_intervals()["y"][0]["x1"]

    analysis = SobolAnalysis(
        [discipline], uncertain_space, 100, use_asymptotic_distributions=False
    )
    analysis.compute_indices()
    bootstrap_interval = analysis.get_intervals()["y"][0]["x1"]

    assert asymptotic_interval[0][0] != bootstrap_interval[0][0]
    assert asymptotic_interval[1][0] != bootstrap_interval[1][0]


def test_confidence_level_default(discipline, uncertain_space):
    """Check the default confidence level used by the algorithm."""
    analysis = SobolAnalysis([discipline], uncertain_space, 100)
    analysis.compute_indices()
    algos = analysis._SobolAnalysis__output_names_to_sobol_algos
    assert algos["y"][0].getConfidenceLevel() == 0.95


def test_confidence_level_custom(discipline, uncertain_space):
    """Check setting a custom confidence level."""
    analysis = SobolAnalysis([discipline], uncertain_space, 100)
    analysis.compute_indices(confidence_level=0.90)
    algos = analysis._SobolAnalysis__output_names_to_sobol_algos
    assert algos["y"][0].getConfidenceLevel() == 0.90


def test_output_variances(sobol):
    """Check SobolAnalysis.output_variances."""
    dataset = sobol.dataset
    assert compare_dict_of_arrays(
        sobol.output_variances,
        {
            name: dataset.get_view(variable_names=name)
            .to_numpy()[: len(dataset) // 5]
            .var(0)
            for name in ["y", "z"]
        },
        tolerance=0.1,
    )


def test_output_standard_deviations(sobol):
    """Check SobolAnalysis.output_standard_deviations."""
    dataset = sobol.dataset
    assert compare_dict_of_arrays(
        sobol.output_standard_deviations,
        {
            name: dataset.get_view(variable_names=name)
            .to_numpy()[: len(dataset) // 5]
            .std(0)
            for name in ["y", "z"]
        },
        tolerance=0.1,
    )


@pytest.mark.parametrize("use_variance", [False, True])
@pytest.mark.parametrize("order", ["first", "second", "total"])
def test_unscale_indices(sobol, use_variance, order):
    """Check SobolAnalysis.unscaled_indices()."""
    orders_to_indices = {
        "first": sobol.first_order_indices,
        "second": sobol.second_order_indices,
        "total": sobol.total_order_indices,
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
                        {k: v**0.5 for k, v in sobol_index.items()}
                        if is_second_order
                        else sobol_index**0.5
                    )
                    for input_name, sobol_index in output_value.items()
                }
                for output_value in output_values
            ]
            for output_name, output_values in expected.items()
        }

    unscaled_indices = sobol.unscale_indices(indices, use_variance=use_variance)
    for output_name, output_values in expected.items():
        for output_index, output_value in enumerate(output_values):
            assert compare_dict_of_arrays(
                unscaled_indices[output_name][output_index], output_value, tolerance=0.1
            )


def test_compute_indices_output_names(sobol):
    """Check compute_indices with different types for output_names."""
    assert sobol.compute_indices(["y"]).keys() == sobol.compute_indices("y").keys()
