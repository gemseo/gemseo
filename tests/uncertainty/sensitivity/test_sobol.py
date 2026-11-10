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

import re
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import pytest
from matplotlib.figure import Figure
from numpy import array
from numpy import isnan
from numpy import nan
from numpy import pi
from numpy import sign
from numpy import sin
from numpy import zeros
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray

from gemseo import sample_disciplines
from gemseo.dataset import Dataset
from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.discipline.auto_py import AutoPyDiscipline
from gemseo.doe.openturns.settings.ot_sobol_indices import OT_SOBOL_INDICES_Settings
from gemseo.doe.scipy.settings.mc import MC_Settings
from gemseo.formulation.mdf_settings import MDF_Settings
from gemseo.post.dataset.heatmap_settings import Heatmap_Settings
from gemseo.post.dataset.treemap_settings import Treemap_Settings
from gemseo.problem.uncertainty.ishigami import statistics as ishigami_statistics
from gemseo.problem.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problem.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.space.random import RandomSpace
from gemseo.uncertainty.distribution.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.uncertainty.sensitivity.core.base import FirstOrderIndicesType
from gemseo.uncertainty.sensitivity.core.base import SecondOrderIndicesType
from gemseo.uncertainty.sensitivity.sobol import SobolAnalysis
from gemseo.uncertainty.sensitivity.sobol import SobolAnalysisMethod
from gemseo.util.comparison import compare_dict_of_arrays
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline
    from gemseo.post.dataset.treemap import Treemap

StatisticsType = tuple[
    dict[str, NDArray[float]],
    dict[str, FirstOrderIndicesType | SecondOrderIndicesType],
]


def ishigami(x1, x23):
    y = array([sin(x1[0]) + 7 * sin(x23[0]) ** 2 + 0.1 * x23[1] ** 4 * sin(x1[0])])
    z = array([y[0], y[0]])
    return y, z


@pytest.fixture(scope="module")
def discipline() -> AutoPyDiscipline:
    """The discipline of interest."""
    return AutoPyDiscipline(py_func=ishigami, use_arrays=True)


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
def random_space() -> RandomSpace:
    """The random space of interest."""
    space = RandomSpace()
    for name, size in zip(["x1", "x23"], [1, 2], strict=True):
        space.add_variable(
            name, *[OTUniformDistribution_Settings(minimum=-pi, maximum=pi)] * size
        )
    return space


@pytest.fixture(scope="module")
def sobol(discipline: Discipline, random_space: RandomSpace) -> SobolAnalysis:
    """A Sobol' analysis."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], random_space, 100)
    analysis.compute_indices()
    return analysis


@pytest.fixture(scope="module")
def sobol_without_second_order(
    discipline: Discipline, random_space: RandomSpace
) -> SobolAnalysis:
    """A Sobol' analysis without second-order indices."""
    analysis = SobolAnalysis()
    analysis.compute_samples(
        [discipline], random_space, 100, compute_second_order=False
    )
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


@pytest.fixture(scope="module")
def cv1_stat(
    discipline_cv1: Discipline,
    random_space: RandomSpace,
) -> StatisticsType:
    """The estimated output variance and Sobol' indices.

    Here for the first CV discipline.
    """
    sobol_analysis = SobolAnalysis()
    sobol_analysis.compute_samples([discipline_cv1], random_space, 100)
    return sobol_analysis.output_variances, sobol_analysis.compute_indices()


@pytest.fixture(scope="module")
def cv2_stat(
    discipline_cv2: Discipline,
    random_space: RandomSpace,
) -> StatisticsType:
    """The estimated output variance and Sobol' indices.

    Here for the second CV discipline.
    """
    sobol_analysis = SobolAnalysis()
    sobol_analysis.compute_samples([discipline_cv2], random_space, 100)
    return sobol_analysis.output_variances, sobol_analysis.compute_indices()


@pytest.fixture(scope="module")
def sobol_cv(
    discipline: Discipline,
    random_space: RandomSpace,
    discipline_cv1: Discipline,
    cv1_stat: StatisticsType,
) -> SobolAnalysis:
    """A Sobol' analysis when a control variate is used."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], random_space, 100)
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


def test_random_space_is_not_converted(discipline) -> None:
    """Check that a random space is stored and re-sampled unchanged."""
    random_space = RandomSpace()
    for name, size in zip(["x1", "x23"], [1, 2], strict=True):
        random_space.add_variable(
            name, *[OTUniformDistribution_Settings(minimum=-pi, maximum=pi)] * size
        )

    analysis = SobolAnalysis()
    dataset = analysis.compute_samples([discipline], random_space, 100)
    assert dataset.misc["random_space"] is random_space
    assert analysis._input_names == ["x1", "x23"]


def test_algo(discipline, random_space) -> None:
    """Check that algorithm can be passed either as a str or an Algorithm."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], random_space, 100)
    assert compare_dict_of_arrays(
        analysis.compute_indices(algo=analysis.Algorithm.JANSEN).first["y"][0],
        analysis.compute_indices(algo="Jansen").first["y"][0],
    )


@pytest.mark.parametrize("method", ["total", SobolAnalysisMethod.TOTAL])
def test_method(sobol, method) -> None:
    """Check the use of the main method."""
    main_method = sobol.main_method
    assert sobol.main_method == "first"
    assert compare_dict_of_arrays(
        sobol.main_indices["y"][0], sobol.indices.first["y"][0], 0.1
    )

    sobol.main_method = method
    assert sobol.main_method == "total"
    assert compare_dict_of_arrays(
        sobol.main_indices["y"][0], sobol.indices.total["y"][0], 0.1
    )

    sobol.main_method = main_method


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
    ("name", "sort", "sort_by_total", "kwargs"),
    [
        ("y", False, False, {}),
        ("y", False, False, {"title": "foo"}),
        ("y", True, False, {}),
        ("y", True, True, {}),
        ("z", False, False, {}),
        (("z", 1), False, False, {}),
    ],
)
def test_plot(name, sobol, sort, sort_by_total, kwargs, snapshot_matplotlib) -> None:
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
def test_second_order(discipline, random_space, compute_second_order) -> None:
    """Check the computation of second-order indices."""
    analysis = SobolAnalysis()
    analysis.compute_samples(
        [discipline], random_space, 100, compute_second_order=compute_second_order
    )
    analysis.compute_indices()
    assert bool(analysis.indices.second) is compute_second_order
    assert len(analysis.dataset) == (96 if compute_second_order else 100)


def test_algo_settings(discipline, random_space) -> None:
    """Check the effect of algo_settings."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], random_space, 100)
    reference = analysis.dataset
    analysis = SobolAnalysis()
    analysis.compute_samples(
        [discipline], random_space, 100, algo_settings=OT_SOBOL_INDICES_Settings()
    )
    analysis.dataset
    assert reference.equals(analysis.dataset)


def test_formulation_settings(discipline, random_space) -> None:
    """Check the effect of formulation_settings."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], random_space, 100)
    reference = analysis.dataset
    analysis.compute_samples(
        [discipline], random_space, 100, formulation_settings=MDF_Settings()
    )
    assert reference.equals(analysis.dataset)


def test_asymptotic_or_bootstrap_intervals(discipline, random_space) -> None:
    """Check the method to compute the confidence intervals."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], random_space, 100)
    analysis.compute_indices()
    asymptotic_interval = analysis.get_intervals()["y"][0]["x1"]

    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], random_space, 100)
    analysis.compute_indices(use_asymptotic_distributions=False)
    bootstrap_interval = analysis.get_intervals()["y"][0]["x1"]

    assert asymptotic_interval[0][0] != bootstrap_interval[0][0]
    assert asymptotic_interval[1][0] != bootstrap_interval[1][0]


def test_confidence_level_default(discipline, random_space) -> None:
    """Check the default confidence level used by the algorithm."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], random_space, 100)
    analysis.compute_indices()
    estimators = analysis._output_name_to_estimators
    assert estimators["y"][0].algo.getConfidenceLevel() == 0.95


def test_confidence_level_custom(discipline, random_space) -> None:
    """Check setting a custom confidence level."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], random_space, 100)
    analysis.compute_indices(confidence_level=0.90)
    estimators = analysis._output_name_to_estimators
    assert estimators["y"][0].algo.getConfidenceLevel() == 0.90


def test_output_variances(sobol) -> None:
    """Check SobolAnalysis.output_variances."""
    dataset = sobol.dataset
    assert compare_dict_of_arrays(
        sobol.output_variances,
        {
            name: dataset
            .get_view(variable_names=name)
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
            name: dataset
            .get_view(variable_names=name)
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
    f = sobol._SobolIndicesEstimatorMixin__unscale_index
    assert not any(isnan(f(array([0.23, -0.87]), "y", 0, False)))
    assert not any(isnan(f({"foo": array([0.23, -0.87])}, "y", 0, False)["foo"]))


def test_compute_indices_output_names(sobol) -> None:
    """Check compute_indices with different types for output_names."""
    assert sobol.compute_indices(["y"]).first
    assert sobol.compute_indices("y").first
    # sobol is a module-scoped fixture and so the original indexes must be restored.
    sobol.compute_indices()


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
    # sobol is a module-scoped fixture and so the original indexes must be restored.
    sobol.compute_indices()


def test_cv_wo_statistics(
    sobol,
    discipline_cv1,
    cv1_stat,
    random_space,
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


def test_cv_wo_statistics_legacy_parameter_space_key(
    sobol,
    discipline_cv1,
) -> None:
    """Check that the legacy ``misc["parameter_space"]`` key is still readable.

    Before the renaming of this key, the random space was stored under
    ``"parameter_space"``; ``compute_indices`` with control variates must still be
    able to read it, so that user code assembling the miscellaneous data of a
    dataset against the previous name still works.
    """
    misc = sobol.dataset.misc
    random_space = misc.pop("random_space")
    misc["parameter_space"] = random_space
    try:
        cv = sobol.ControlVariate(
            discipline=discipline_cv1, indices=None, n_samples=100, variance=None
        )
        cv = sobol._SobolAnalysis__compute_cv_stats(cv)
        assert cv.indices is not None
        assert cv.variance is not None
    finally:
        del misc["parameter_space"]
        misc["random_space"] = random_space


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
    # sobol is a module-scoped fixture and so the original indexes must be restored.
    sobol.compute_indices()


def test_plot_cv(sobol_cv, snapshot_matplotlib) -> None:
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
    """Check the instantiation from samples with control variates."""
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


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "control_variates": SobolAnalysis.ControlVariate(
                AnalyticDiscipline({"varying": "x1+x2", "constant": "1"})
            )
        },
    ],
)
def test_constant_output(discipline_with_constant_output_and_space, kwargs):
    """Check that SobolAnalysis supports constant outputs."""
    discipline, random_space = discipline_with_constant_output_and_space
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], random_space, 100)
    indices = analysis.compute_indices(**kwargs)
    assert indices.first["constant"][0] is None
    assert indices.total["constant"][0] is None
    if not kwargs:
        assert indices.second["constant"][0] is None

    assert indices.first["varying"][0]["x1"] is not None
    assert indices.total["varying"][0]["x2"] is not None
    if not kwargs:
        assert indices.second["varying"][0] is not None

    # Regression test: get_intervals() used to raise an AttributeError for a
    # zero-variance output, as it called methods on the `None` algorithm that
    # compute_indices() appends for such an output.
    for first_order in [True, False]:
        intervals = analysis.get_intervals(first_order=first_order)
        for input_name in ("x1", "x2"):
            assert_almost_equal(
                intervals["constant"][0][input_name], array([[0.0], [0.0]])
            )
            assert intervals["varying"][0][input_name].shape == (2, 1)


def test_rank_based_sobol_warning(discipline, random_space, caplog):
    """Check that a warning is logged if the user tries to compute second-order indices
    with a rank-based Sobol' analysis."""
    analysis = SobolAnalysis()
    analysis.compute_samples(
        [discipline],
        random_space,
        10,
        algo_settings=MC_Settings(),
    )
    assert (
        "gemseo.uncertainty.sensitivity._sobol_indices_estimator",
        30,
        (
            "The second-order indices can only be computed "
            "with the OT_SOBOL_INDICES algorithm."
        ),
    ) in caplog.record_tuples


@pytest.mark.parametrize(
    "algo",
    [
        algo
        for algo in SobolAnalysis.Algorithm
        if algo != SobolAnalysis.Algorithm.SALTELLI
    ],
)
def test_algo_control_variate_error(discipline, random_space, algo):
    """Check that an error is raised if the user tries to use control variates
    with a wrong algorithm."""
    analysis = SobolAnalysis()
    analysis.compute_samples(
        [discipline],
        random_space,
        10,
        algo_settings=MC_Settings() if algo == SobolAnalysis.Algorithm.RANK else None,
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The Saltelli algorithm is required for the use of control variates."
        ),
    ):
        analysis.compute_indices(algo=algo, control_variates=["mock"])


def test_pf_algo_compatibility_error(discipline, random_space):
    """Check that an error is raised if the user tries to use a Sobol' estimation
    algorithm expecting pick-and-freeze (PF) samples with non-PF samples."""
    analysis = SobolAnalysis()
    analysis.compute_samples(
        [discipline],
        random_space,
        10,
        algo_settings=MC_Settings(),
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Sobol' estimation algorithms (except rank-based) expect "
            "pick-and-freeze samples."
        ),
    ):
        analysis.compute_indices(algo=analysis.Algorithm.SALTELLI)


def test_rank_based_algo_compatibility_error(discipline, random_space):
    """Check that an error is raised if the user tries to use the rank-based Sobol'
    estimation algorithm samples with pick-and-freeze samples."""
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], random_space, 100)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The rank-based Sobol' estimation algorithm expects Monte Carlo samples."
        ),
    ):
        analysis.compute_indices(algo=analysis.Algorithm.RANK)


@pytest.fixture
def rank_based_sobol(discipline, random_space):
    """A rank-based Sobol' analysis."""
    analysis = SobolAnalysis()
    analysis.compute_samples(
        [discipline],
        random_space,
        100,
        algo_settings=MC_Settings(),
    )
    return analysis


def test_rank_based_sobol(rank_based_sobol, snapshot_matplotlib):
    """Check that the rank-based Sobol' analysis works."""
    rank_based_sobol.compute_indices()

    # Check that rank-based Sobol' analysis cannot compute second-order
    # and total indices.
    assert not rank_based_sobol.indices.total
    assert not rank_based_sobol.indices.second

    # Check the first-order indices.
    for output_name, component in [("y", 0), ("z", 0), ("z", 1)]:
        assert_almost_equal(
            rank_based_sobol.indices.first[output_name][component]["x1"],
            array([0.33418792]),
        )
        assert_almost_equal(
            rank_based_sobol.indices.first[output_name][component]["x23"],
            array([0.4059578, -0.05841109]),
        )

    # Check that plotting works, even if total indices are missing.
    rank_based_sobol.plot("y", save=False)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, [[0.1903898], [0.4456537]]),
        # Check the confidence intervals with 10 replicates instead of 100.
        ({"n_replicates": 10}, [[0.1602552], [0.4817939]]),
        # Check the confidence intervals with a 90% confidence level instead of 95%
        ({"confidence_level": 0.90}, [[0.2103695], [0.4231993]]),
    ],
)
def test_rank_based_sobol_bootstrap(rank_based_sobol, kwargs, expected):
    """Check that the bootstrap options of rank-based Sobol' analysis work."""
    rank_based_sobol.compute_indices(**kwargs)
    assert_almost_equal(rank_based_sobol.get_intervals()["y"][0]["x1"], array(expected))


def test_ranked_based_sobol_from_samples(discipline, random_space):
    """Rank-based Sobol' analysis supports any IODataset as samples.

    The only requirement is that
    the samples are independent and identically distributed.
    """
    samples = sample_disciplines(
        [discipline],
        random_space,
        ["y"],
        algo_settings_model=MC_Settings(n_samples=100),
    )
    analysis = SobolAnalysis(samples)
    analysis.compute_indices()
    assert_almost_equal(
        analysis.get_intervals()["y"][0]["x1"], array([[0.1903898], [0.4456537]])
    )


def test_plot_second_order(sobol, snapshot_matplotlib):
    """The plot_second_order method plots the second-order indices using a heat map."""
    heatmap = sobol.plot_second_order("y")
    heatmap.execute(save=False)
    heatmap = sobol.plot_second_order(
        "y", settings=Heatmap_Settings(colormap="viridis")
    )
    heatmap.execute(save=False)


@pytest.mark.parametrize(
    ("settings", "n_warnings"),
    [
        (OT_SOBOL_INDICES_Settings(eval_second_order=True), 1),
        (OT_SOBOL_INDICES_Settings(eval_second_order=False), 0),
        (OT_SOBOL_INDICES_Settings(), 0),
    ],
    ids=["conflicting", "consistent", "default"],
)
def test_eval_second_order_replaced_warning(
    discipline, random_space, caplog, settings, n_warnings
) -> None:
    """Check the warning about an explicit eval_second_order replaced."""
    SobolAnalysis().compute_samples(
        [discipline],
        random_space,
        20,
        algo_settings=settings,
        compute_second_order=False,
    )
    messages = [
        r.message
        for r in caplog.records
        if "The setting 'eval_second_order'" in r.message
    ]
    assert len(messages) == n_warnings


def test_settings_unchanged(discipline, random_space) -> None:
    """Check that compute_samples leaves the settings of the user unchanged."""
    settings = OT_SOBOL_INDICES_Settings()
    SobolAnalysis().compute_samples(
        [discipline],
        random_space,
        20,
        algo_settings=settings,
        compute_second_order=False,
    )
    assert settings == OT_SOBOL_INDICES_Settings()


@pytest.mark.parametrize(
    ("settings", "is_logged"),
    [
        (OT_SOBOL_INDICES_Settings(n_samples=10), True),
        (OT_SOBOL_INDICES_Settings(n_samples=20), False),
        (OT_SOBOL_INDICES_Settings(), False),
    ],
    ids=["explicit_replaced", "explicit_same", "default"],
)
def test_n_samples_setting_replaced_warning(
    discipline, random_space, caplog, settings, is_logged
) -> None:
    """Check the warning about an explicit n_samples replaced by the argument."""
    SobolAnalysis().compute_samples(
        [discipline],
        random_space,
        20,
        algo_settings=settings,
        compute_second_order=False,
    )
    message = (
        "The setting 'n_samples' of OT_SOBOL_INDICES_Settings (10) is replaced by "
        "the argument 'n_samples' (20)."
    )
    assert (message in caplog.text) is is_logged


@pytest.fixture(scope="module")
def ishigami_sobol() -> SobolAnalysis:
    """A Sobol' analysis of the Ishigami function with second-order indices."""
    analysis = SobolAnalysis()
    analysis.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 100)
    analysis.compute_indices()
    return analysis


def test_plot_variance_decomposition_from_estimates(ishigami_sobol) -> None:
    """Check that the variance decomposition can be drawn from estimated indices."""
    treemap = ishigami_sobol.plot_variance_decomposition("y")
    assert treemap.dataset.get_columns()[-1] == "orders ≥ 3"
    treemap.execute(save=False)


def _make_indices(
    first: Mapping[str, Sequence[float]],
    second: Mapping[tuple[str, str], Sequence[Sequence[float]]],
) -> SobolAnalysis.SensitivityIndices:
    """Return the Sobol' indices of an output "y" from hand-made values.

    Args:
        first: The first-order indices of each input.
        second: The second-order indices of some pairs of inputs,
            as matrices whose rows correspond to the components of the first input
            and columns to the ones of the second input.
            The missing pairs have zero indices.

    Returns:
        The Sobol' indices.
    """
    second_order = {
        name_i: {
            name_j: zeros((len(first[name_i]), len(first[name_j]))) for name_j in first
        }
        for name_i in first
    }
    for (name_i, name_j), values in second.items():
        second_order[name_i][name_j] = array(values)
        second_order[name_j][name_i] = array(values).T

    return SobolAnalysis.SensitivityIndices(
        first={"y": [{name: array(values) for name, values in first.items()}]},
        second={"y": [second_order]},
    )


def _patch_with_ishigami_indices(
    analysis: SobolAnalysis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Give a Sobol' analysis the exact indices of the Ishigami function.

    Args:
        analysis: The Sobol' analysis of the Ishigami function.
        monkeypatch: The pytest fixture to patch the analysis.
    """
    stats = ishigami_statistics
    indices = _make_indices(
        {"x1": [stats.sobol_1], "x2": [stats.sobol_2], "x3": [stats.sobol_3]},
        {
            ("x1", "x2"): [[stats.sobol_12]],
            ("x1", "x3"): [[stats.sobol_13]],
            ("x2", "x3"): [[stats.sobol_23]],
        },
    )
    monkeypatch.setattr(analysis, "_indices", indices)


def test_plot_variance_decomposition(
    ishigami_sobol, monkeypatch, snapshot_matplotlib
) -> None:
    """Check the tiles of the variance decomposition of the Ishigami function."""
    _patch_with_ishigami_indices(ishigami_sobol, monkeypatch)
    treemap = ishigami_sobol.plot_variance_decomposition("y")
    dataset = treemap.dataset
    assert dataset.get_columns() == ["x1", "x2", "x1 × x3", "orders ≥ 3"]  # noqa: RUF001
    stats = ishigami_statistics
    assert_almost_equal(
        dataset.get_view().to_numpy()[0],
        [stats.sobol_1, stats.sobol_2, stats.sobol_13, 0.0],
    )
    treemap.execute(save=False)


@pytest.mark.parametrize(
    ("setting_name", "expected"),
    [
        ("title", "Variance decomposition of y"),
        ("residual_name", "orders ≥ 3"),
        ("residual_label", "small terms"),
        ("colormap", "tab10"),
        # The exact indices sum to 1 up to a rounding error: no rescaling to report.
        ("footnote", ""),
    ],
)
def test_plot_variance_decomposition_default_settings(
    ishigami_sobol, monkeypatch, setting_name, expected
) -> None:
    """Check the settings computed when the user gives none."""
    _patch_with_ishigami_indices(ishigami_sobol, monkeypatch)
    treemap = ishigami_sobol.plot_variance_decomposition("y")
    assert getattr(treemap.settings, setting_name) == expected


@pytest.mark.parametrize(
    ("tile_name", "expected"),
    [
        pytest.param("x1", ("#1f77b4",), id="first_input"),
        pytest.param("x2", ("#ff7f0e",), id="second_input"),
        pytest.param("x1 × x3", ("#8fbbda", "#96d096"), id="interaction"),  # noqa: RUF001
    ],
)
def test_plot_variance_decomposition_tile_colors(
    ishigami_sobol, monkeypatch, tile_name, expected
) -> None:
    """Check that an input has a color of the colormap and an interaction faded ones."""
    _patch_with_ishigami_indices(ishigami_sobol, monkeypatch)
    treemap = ishigami_sobol.plot_variance_decomposition("y")
    assert treemap.settings.tile_colors[tile_name] == expected


def _plot_variance_decomposition_with_user_settings(
    sobol: SobolAnalysis, monkeypatch: pytest.MonkeyPatch, **settings: Any
) -> Treemap:
    """Plot the variance decomposition of hand-made indices with user settings.

    Args:
        sobol: The Sobol' analysis.
        monkeypatch: The pytest fixture to patch the analysis.
        **settings: The settings of the treemap given by the user.

    Returns:
        The treemap.
    """
    # The computed footnote is not empty.
    indices = _make_indices(
        {"x1": [0.8], "x23": [0.6, -0.1]}, {("x1", "x23"): [[0.4, 0.2]]}
    )
    monkeypatch.setattr(sobol, "_indices", indices)
    return sobol.plot_variance_decomposition("y", Treemap_Settings(**settings))


@pytest.mark.parametrize(
    ("setting_name", "value"),
    [
        ("title", "My title"),
        ("colormap", "viridis"),
        ("residual_label", "tiny"),
        ("footnote", "My footnote"),
    ],
)
def test_plot_variance_decomposition_user_settings(
    sobol, monkeypatch, setting_name, value
) -> None:
    """Check that a setting given by the user overrides the computed one."""
    treemap = _plot_variance_decomposition_with_user_settings(
        sobol, monkeypatch, **{setting_name: value}
    )
    assert getattr(treemap.settings, setting_name) == value


@pytest.mark.parametrize(
    ("setting_name", "value", "expected"),
    [
        pytest.param("tile_colors", {"x1": "red"}, {"x1": ("red",)}, id="tile_colors"),
        pytest.param("labels", {"x2": "Second"}, {"x2": "Second"}, id="labels"),
    ],
)
def test_plot_variance_decomposition_user_mapping_settings(
    sobol, monkeypatch, setting_name, value, expected
) -> None:
    """Check that the entries of a mapping given by the user override the computed."""
    treemap = _plot_variance_decomposition_with_user_settings(
        sobol, monkeypatch, **{setting_name: value}
    )
    actual = getattr(treemap.settings, setting_name)
    assert {key: actual[key] for key in value} == expected


@pytest.mark.parametrize(
    ("tile_colors", "expected"),
    [
        pytest.param({"x1": "red"}, ("#ff8080", "#ffbf86"), id="color_name"),
        pytest.param({"x1": (1.0, 0.0, 0.0)}, ("#ff8080", "#ffbf86"), id="rgb_tuple"),
        pytest.param(
            {"x23[0]": (0.0, 0.0, 1.0, 0.5)},
            ("#8fbbda", "#8080ff"),
            id="rgba_tuple",
        ),
        pytest.param(
            {"x1": ("red", "blue")},
            ("#8fbbda", "#ffbf86"),
            id="bands_ignored",
        ),
    ],
)
def test_plot_variance_decomposition_user_colors_fade_in_interactions(
    sobol, monkeypatch, tile_colors, expected
) -> None:
    """Check that the interaction of two inputs fades the colors set by the user."""
    indices = _make_indices(
        {"x1": [0.4], "x23": [0.3, 0.1]}, {("x1", "x23"): [[0.1, 0.05]]}
    )
    monkeypatch.setattr(sobol, "_indices", indices)
    settings = sobol.plot_variance_decomposition(
        "y", Treemap_Settings(tile_colors=tile_colors)
    ).settings
    assert settings.tile_colors["x1 × x23[0]"] == expected  # noqa: RUF001


def test_plot_variance_decomposition_two_components(sobol, monkeypatch) -> None:
    """Check that the residual tile is not said to have interactions of order 3."""
    indices = _make_indices({"x1": [0.5], "x2": [0.3]}, {("x1", "x2"): [[0.1]]})
    monkeypatch.setattr(sobol, "_indices", indices)
    monkeypatch.setattr(
        sobol,
        "dataset",
        Dataset.from_array(zeros((1, 3)), variable_names=["x1", "x2", "y"]),
    )
    treemap = sobol.plot_variance_decomposition("y")
    assert list(treemap.dataset.get_columns()) == [
        "x1",
        "x2",
        "x1 × x2",  # noqa: RUF001
        "estimation error",
    ]
    assert treemap.settings.residual_name == "estimation error"


def test_plot_variance_decomposition_input_named_residual(sobol, monkeypatch) -> None:
    """Check that an input named "residual" does not clash with the residual tile."""
    indices = _make_indices(
        {"residual": [0.5], "x": [0.3]}, {("residual", "x"): [[0.1]]}
    )
    monkeypatch.setattr(sobol, "_indices", indices)
    monkeypatch.setattr(
        sobol,
        "dataset",
        Dataset.from_array(zeros((1, 3)), variable_names=["residual", "x", "y"]),
    )
    treemap = sobol.plot_variance_decomposition(
        "y", Treemap_Settings(labels={"residual": "R"})
    )
    names = list(treemap.dataset.get_columns())
    assert len(set(names)) == len(names)
    assert names[-1] == "estimation error"
    assert treemap.settings.labels["estimation error"] == "estimation error"
    treemap.execute(save=False)


def test_plot_variance_decomposition_without_second_order(
    sobol_without_second_order,
) -> None:
    """Check that the first-order indices and a residual are drawn by default."""
    treemap = sobol_without_second_order.plot_variance_decomposition(("y", 0))
    names = list(treemap.dataset.get_columns())
    # The negative estimates have no tile.
    assert names[-1] == "orders ≥ 2"
    assert treemap.settings.residual_name == "orders ≥ 2"
    assert set(names[:-1]) <= {"x1", "x23[0]", "x23[1]"}
    assert "x1 × x23[0]" not in names  # noqa: RUF001
    assert_almost_equal(sum(treemap._create_specific_data_from_dataset()[1]), 1.0)
    treemap.execute(save=False)


def test_plot_variance_decomposition_without_second_order_logs(
    sobol_without_second_order, caplog
) -> None:
    """Check that the absence of second-order indices is logged."""
    sobol_without_second_order.plot_variance_decomposition(("y", 0))
    assert (
        "The second-order Sobol' indices of 'y' are not computed; "
        "only the first-order ones are shown, "
        "the residual tile gathering all the interactions." in caplog.text
    )


def test_plot_variance_decomposition_negative_terms(
    sobol, monkeypatch, snapshot_matplotlib
) -> None:
    """Check that negative estimates have no tile and are given in the footnote."""
    indices = _make_indices(
        {"x1": [0.4], "x23": [0.3, -0.05]},
        {("x1", "x23"): [[0.1, 0.05]], ("x23", "x23"): [[0.0, -0.02], [-0.02, 0.0]]},
    )
    monkeypatch.setattr(sobol, "_indices", indices)
    treemap = sobol.plot_variance_decomposition("y")
    dataset = treemap.dataset
    names = list(dataset.get_columns())
    areas = dataset.get_view().to_numpy()[0]
    assert names == ["x1", "x23[0]", "x1 × x23[0]", "x1 × x23[1]", "orders ≥ 3"]  # noqa: RUF001
    assert_almost_equal(areas, [0.4, 0.3, 0.1, 0.05, 0.15])
    assert treemap.settings.footnote == (
        "Negative estimates, without tile: x23[1] = -0.05, x23[0] × x23[1] = -0.02."  # noqa: RUF001
    )
    treemap.execute(save=False)


def _patch_with_scalar_inputs(
    analysis: SobolAnalysis, monkeypatch: pytest.MonkeyPatch, n_inputs: int
) -> list[str]:
    """Give a Sobol' analysis scalar inputs with equal first-order indices.

    Args:
        analysis: The Sobol' analysis.
        monkeypatch: The pytest fixture to patch the analysis.
        n_inputs: The number of scalar inputs.

    Returns:
        The names of the inputs.
    """
    names = [f"x{i}" for i in range(n_inputs)]
    indices = _make_indices({name: [0.5 / n_inputs] for name in names}, {})
    monkeypatch.setattr(analysis, "_indices", indices)
    monkeypatch.setattr(
        analysis,
        "dataset",
        Dataset.from_array(zeros((1, n_inputs + 1)), variable_names=[*names, "y"]),
    )
    return names


@pytest.mark.parametrize(
    ("n_inputs", "colormap"), [(10, "tab10"), (12, "tab20"), (21, "viridis")]
)
def test_plot_variance_decomposition_many_inputs(
    sobol, monkeypatch, n_inputs, colormap
) -> None:
    """Check that each input has its own color when the default colormap is short."""
    names = _patch_with_scalar_inputs(sobol, monkeypatch, n_inputs)
    settings = sobol.plot_variance_decomposition("y").settings
    assert settings.colormap == colormap
    assert len({settings.tile_colors[name] for name in names}) == n_inputs


def test_plot_variance_decomposition_many_inputs_tab20_order(
    sobol, monkeypatch
) -> None:
    """Check that the first inputs have distinct hues when tab20 is used."""
    names = _patch_with_scalar_inputs(sobol, monkeypatch, 12)
    tile_colors = sobol.plot_variance_decomposition("y").settings.tile_colors
    # The dark tones of tab20 come first, then the light ones.
    assert [tile_colors[name][0] for name in names[:4]] == [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
    ]
    assert tile_colors[names[10]] == ("#aec7e8",)


def test_plot_variance_decomposition_many_inputs_user_colormap(
    sobol, monkeypatch
) -> None:
    """Check that the colormap set by the user is kept, even if it is short."""
    _patch_with_scalar_inputs(sobol, monkeypatch, 12)
    settings = sobol.plot_variance_decomposition(
        "y", Treemap_Settings(colormap="tab10")
    ).settings
    assert settings.colormap == "tab10"
    assert settings.tile_colors["x0"] == settings.tile_colors["x10"]


def test_plot_variance_decomposition_of_constant_output(
    discipline_with_constant_output_and_space, snapshot
) -> None:
    """Check that the variance decomposition of a constant output is undefined."""
    discipline, random_space = discipline_with_constant_output_and_space
    analysis = SobolAnalysis()
    analysis.compute_samples([discipline], random_space, 100)
    analysis.compute_indices()
    with assert_exception(ValueError, snapshot):
        analysis.plot_variance_decomposition("constant")


@pytest.mark.parametrize(
    ("first", "second", "threshold", "expected_columns", "expected_values"),
    [
        pytest.param(
            {"x1": [0.4], "x23": [0.3, 0.005]},
            {
                ("x1", "x23"): [[0.1, -0.004]],
                ("x23", "x23"): [[0.0, 0.003], [0.003, 0.0]],
            },
            0.01,
            ["x1", "x23[0]", "x1 × x23[0]", "orders ≥ 3"],  # noqa: RUF001
            [0.4, 0.3, 0.1, 0.2],
            id="default_threshold",
        ),
        pytest.param(
            {"x1": [0.4], "x23": [0.3, 0.005]},
            {
                ("x1", "x23"): [[0.1, -0.004]],
                ("x23", "x23"): [[0.0, 0.003], [0.003, 0.0]],
            },
            0.0,
            [
                "x1",
                "x23[0]",
                "x23[1]",
                "x1 × x23[0]",  # noqa: RUF001
                "x23[0] × x23[1]",  # noqa: RUF001
                "orders ≥ 3",
            ],
            [0.4, 0.3, 0.005, 0.1, 0.003, 0.192],
            id="zero_threshold",
        ),
        pytest.param(
            {"x1": [0.4], "x23": [0.3, nan]},
            {("x1", "x23"): [[0.1, 0.05]]},
            0.01,
            [
                "x1",
                "x23[0]",
                "x1 × x23[0]",  # noqa: RUF001
                "x1 × x23[1]",  # noqa: RUF001
                "orders ≥ 3",
            ],
            [0.4, 0.3, 0.1, 0.05, 0.15],
            id="undefined_term",
        ),
        pytest.param(
            {"x1": [0.8], "x23": [0.6, 0.0]},
            {("x1", "x23"): [[0.4, 0.2]]},
            0.01,
            [
                "x1",
                "x23[0]",
                "x1 × x23[0]",  # noqa: RUF001
                "x1 × x23[1]",  # noqa: RUF001
                "orders ≥ 3",
            ],
            [0.8, 0.6, 0.4, 0.2, 0.0],
            id="sum_above_one",
        ),
    ],
)
def test_plot_variance_decomposition_dataset(
    sobol,
    monkeypatch,
    first,
    second,
    threshold,
    expected_columns,
    expected_values,
) -> None:
    """Check the columns and values of the dataset of the treemap."""
    monkeypatch.setattr(sobol, "_indices", _make_indices(first, second))
    treemap = sobol.plot_variance_decomposition("y", threshold=threshold)
    dataset = treemap.dataset
    assert list(dataset.get_columns()) == expected_columns
    assert_almost_equal(dataset.get_view().to_numpy()[0], expected_values)


@pytest.mark.parametrize(
    ("first", "second", "threshold", "settings", "expected"),
    [
        # The zero pair x23[0] x x23[1] is not a small term: it contributes nothing.
        pytest.param(
            {"x1": [0.4], "x23": [0.3, 0.1]},
            {("x1", "x23"): [[0.1, 0.1]]},
            0.01,
            {},
            "orders ≥ 3",
            id="no_small_term",
        ),
        pytest.param(
            {"x1": [0.4], "x23": [0.3, 0.1]},
            {
                ("x1", "x23"): [[0.1, 0.1]],
                ("x23", "x23"): [[0.0, 0.005], [0.005, 0.0]],
            },
            0.01,
            {},
            "orders ≥ 3 + small terms",
            id="small_term",
        ),
        pytest.param(
            {"x1": [0.4], "x23": [0.3, 0.005]},
            {
                ("x1", "x23"): [[0.1, -0.004]],
                ("x23", "x23"): [[0.0, 0.003], [0.003, 0.0]],
            },
            0.0,
            {},
            "orders ≥ 3",
            id="zero_threshold",
        ),
        pytest.param(
            {"x1": [0.5], "x23": [0.3, 0.001]},
            None,
            0.01,
            {},
            "orders ≥ 2 + small terms",
            id="without_second_order",
        ),
        pytest.param(
            {"x1": [0.5], "x23": [0.3, 0.001]},
            None,
            0.0,
            {},
            "orders ≥ 2",
            id="without_second_order_zero_threshold",
        ),
        # The user's residual label replaces "small terms" in the residual label.
        pytest.param(
            {"x1": [0.6], "x23": [0.3, 0.05]},
            {("x1", "x23"): [[0.0, 0.003]]},
            0.01,
            {"min_area": 0.1, "residual_label": "noise"},
            "orders ≥ 3 + noise",
            id="user_label_with_min_area",
        ),
        pytest.param(
            {"x1": [0.6], "x23": [0.3, 0.05]},
            {("x1", "x23"): [[0.0, 0.003]]},
            0.01,
            {"residual_label": "noise"},
            "orders ≥ 3 + noise",
            id="user_label_without_min_area",
        ),
        # "small terms" is not repeated when tiles are merged by min_area.
        pytest.param(
            {"x1": [0.6], "x23": [0.3, 0.05]},
            {("x1", "x23"): [[0.0, 0.0]]},
            0.01,
            {"min_area": 0.1},
            "orders ≥ 3 + small terms",
            id="default_label_with_min_area",
        ),
        # The user's label of the residual tile keeps the suffix.
        pytest.param(
            {"x1": [0.6], "x23": [0.3, 0.05]},
            {("x1", "x23"): [[0.0, 0.003]]},
            0.01,
            {"labels": {"orders ≥ 3": "rest"}},
            "rest + small terms",
            id="user_residual_tile_label_with_small_terms",
        ),
        pytest.param(
            {"x1": [0.6], "x23": [0.3, 0.05]},
            {("x1", "x23"): [[0.0, 0.0]]},
            0.01,
            {"labels": {"orders ≥ 3": "rest"}},
            "rest",
            id="user_residual_tile_label_without_small_terms",
        ),
    ],
)
def test_plot_variance_decomposition_residual_label(
    sobol, monkeypatch, first, second, threshold, settings, expected
) -> None:
    """Check the label of the residual tile.

    It mentions the small terms only if any, and only once.
    """
    if second is None:
        indices = SobolAnalysis.SensitivityIndices(
            first={"y": [{name: array(value) for name, value in first.items()}]}
        )
    else:
        indices = _make_indices(first, second)

    monkeypatch.setattr(sobol, "_indices", indices)
    treemap = sobol.plot_variance_decomposition(
        "y", Treemap_Settings(**settings), threshold=threshold
    )
    assert treemap._create_specific_data_from_dataset()[0][-1] == expected


@pytest.mark.parametrize(
    ("first", "second", "threshold", "expected"),
    [
        pytest.param(
            {"x1": [0.8], "x23": [0.6, 0.0]},
            {("x1", "x23"): [[0.4, 0.2]]},
            0.01,
            "The estimates sum to 2.00 > 1: the areas are rescaled.",
            id="sum_above_one",
        ),
        pytest.param(
            {"x1": [0.8], "x23": [0.6, 0.005]},
            {("x1", "x23"): [[0.4, 0.2]]},
            0.01,
            "The estimates sum to 2.00 > 1: the areas are rescaled.\n"
            "The indices below the threshold are not shown.",
            id="sum_above_one_and_small_terms",
        ),
        pytest.param(
            {"x1": [0.8], "x23": [0.6, -0.1]},
            {("x1", "x23"): [[0.4, 0.2]]},
            0.01,
            "Negative estimates, without tile: x23[1] = -0.10.\n"
            "The estimates sum to 2.00 > 1: the areas are rescaled.",
            id="negative_terms_and_sum_above_one",
        ),
        pytest.param(
            {"x1": [0.4], "x23": [0.3, -0.02]},
            {
                ("x1", "x23"): [[-0.07, -0.05]],
                ("x23", "x23"): [[0.0, -0.04], [-0.04, 0.0]],
            },
            0.01,
            "4 negative estimates, without tile; "
            "the most negative: "
            "x1 × x23[0] = -0.07, x1 × x23[1] = -0.05, "  # noqa: RUF001
            "x23[0] × x23[1] = -0.04.",  # noqa: RUF001
            id="four_negative_terms_summed_up",
        ),
        # Three negative terms are listed one by one.
        pytest.param(
            {"x1": [0.4], "x23": [0.3, 0.02]},
            {
                ("x1", "x23"): [[-0.07, -0.05]],
                ("x23", "x23"): [[0.0, -0.04], [-0.04, 0.0]],
            },
            0.01,
            "Negative estimates, without tile: "
            "x1 × x23[0] = -0.07, x1 × x23[1] = -0.05, "  # noqa: RUF001
            "x23[0] × x23[1] = -0.04.",  # noqa: RUF001
            id="three_negative_terms_listed",
        ),
        pytest.param(
            {"x1": [0.4], "x23": [0.3, nan]},
            {("x1", "x23"): [[0.1, 0.05]]},
            0.01,
            "Undefined estimates, without tile: x23[1].",
            id="undefined_term",
        ),
        # A negative estimate at the opposite of the threshold is negative.
        pytest.param(
            {"x1": [0.4], "x23": [0.3, -0.02]},
            {},
            0.02,
            "Negative estimates, without tile: x23[1] = -0.02.",
            id="negative_term_equal_to_threshold",
        ),
        pytest.param(
            {"x1": [0.4], "x23": [0.3, -0.02]},
            {},
            0.03,
            "",
            id="negative_term_above_minus_threshold",
        ),
        pytest.param(
            {"x1": [0.4], "x23": [0.3, -0.02]},
            {},
            0.01,
            "Negative estimates, without tile: x23[1] = -0.02.",
            id="negative_term_below_minus_threshold",
        ),
        # A negative estimate is listed even if it rounds to zero.
        pytest.param(
            {"x1": [0.4], "x23": [0.3, 0.005]},
            {
                ("x1", "x23"): [[0.1, -0.004]],
                ("x23", "x23"): [[0.0, 0.003], [0.003, 0.0]],
            },
            0.0,
            "Negative estimates, without tile: x1 × x23[1] = -0.00.",  # noqa: RUF001
            id="negative_term_rounded_to_zero",
        ),
    ],
)
def test_plot_variance_decomposition_footnote(
    sobol, monkeypatch, first, second, threshold, expected
) -> None:
    """Check the footnote listing the estimates without tile and the rescaling."""
    monkeypatch.setattr(sobol, "_indices", _make_indices(first, second))
    treemap = sobol.plot_variance_decomposition("y", threshold=threshold)
    assert treemap.settings.footnote == expected


def test_plot_variance_decomposition_noise_does_not_rescale(sobol, monkeypatch) -> None:
    """Check that the second-order estimation noise neither adds tiles nor rescales."""
    n_inputs = 20
    names = _patch_with_scalar_inputs(sobol, monkeypatch, n_inputs)
    second = {}
    k = 0
    for i in range(n_inputs):
        for j in range(i + 1, n_inputs):
            second[names[i], names[j]] = [[0.005 if k % 2 else -0.005]]
            k += 1

    monkeypatch.setattr(
        sobol,
        "_indices",
        _make_indices({name: [0.4 / n_inputs] for name in names}, second),
    )
    treemap = sobol.plot_variance_decomposition("y")
    assert list(treemap.dataset.get_columns()) == [*names, "orders ≥ 3"]
    assert not treemap.settings.footnote
    assert_almost_equal(treemap.dataset.get_view().to_numpy()[0][-1], 0.6)
    assert treemap.settings.labels["orders ≥ 3"] == "orders ≥ 3 + small terms"


@pytest.mark.parametrize(
    "threshold",
    [pytest.param(-0.1, id="negative"), pytest.param(nan, id="nan")],
)
def test_plot_variance_decomposition_invalid_threshold_error(
    sobol, snapshot, threshold
) -> None:
    """Check that a negative or NaN threshold is rejected."""
    with assert_exception(ValueError, snapshot):
        sobol.plot_variance_decomposition("y", threshold=threshold)


def test_plot_variance_decomposition_input_labels(sobol, monkeypatch) -> None:
    """Check that the labels of the inputs also label their interactions."""
    indices = _make_indices(
        {"x1": [0.3], "x23": [0.2, 0.1]},
        {
            ("x1", "x23"): [[0.1, 0.05]],
            ("x23", "x23"): [[0.0, -0.05], [-0.05, 0.0]],
        },
    )
    monkeypatch.setattr(sobol, "_indices", indices)
    labels = {"x1": "a", "x23": "b", "x23[1]": "c", "x1 × x23[0]": "special"}  # noqa: RUF001
    treemap = sobol.plot_variance_decomposition("y", Treemap_Settings(labels=labels))
    settings = treemap.settings
    assert {name: settings.labels[name] for name in treemap.dataset.get_columns()} == {
        "x1": "a",
        "x23[0]": "b[0]",
        "x23[1]": "c",
        "x1 × x23[0]": "special",  # noqa: RUF001
        "x1 × x23[1]": "a × c",  # noqa: RUF001
        "orders ≥ 3": "orders ≥ 3",
    }
    assert settings.footnote == "Negative estimates, without tile: b[0] × c = -0.05."  # noqa: RUF001
