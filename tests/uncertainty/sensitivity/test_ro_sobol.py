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

from typing import TYPE_CHECKING

import pytest
from matplotlib.figure import Figure
from numpy import array
from numpy import concatenate
from numpy import exp
from numpy import isfinite
from numpy import isnan
from numpy import newaxis
from numpy import roll
from numpy import sqrt
from numpy import tile
from numpy import triu_indices
from numpy import zeros
from numpy.random import default_rng
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from openturns import JansenSensitivityAlgorithm
from openturns import MartinezSensitivityAlgorithm
from openturns import MauntzKucherenkoSensitivityAlgorithm
from openturns import RankSobolSensitivityAlgorithm
from openturns import SaltelliSensitivityAlgorithm
from openturns import Sample
from scipy.stats import multivariate_normal
from scipy.stats import norm

from gemseo.dataset.io_dataset import IODataset
from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.discipline.auto_py import AutoPyDiscipline
from gemseo.doe.openturns._algorithm.ot_sobol_doe import OTSobolDOE
from gemseo.doe.openturns.settings.ot_sobol_indices import OT_SOBOL_INDICES_Settings
from gemseo.space.random import RandomSpace
from gemseo.uncertainty import create_sensitivity_analysis
from gemseo.uncertainty.distribution.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.reliability.openturns.form_settings import OT_FORM_Settings
from gemseo.uncertainty.reliability.openturns.mc_settings import OT_MC_Settings
from gemseo.uncertainty.reliability.scenario import ReliabilityScenario
from gemseo.uncertainty.sensitivity._is_sobol_indices_estimator import (
    ISSobolIndicesEstimator,
)
from gemseo.uncertainty.sensitivity._seeding import seed_ot_random_generator
from gemseo.uncertainty.sensitivity.ro_sobol import ROSobolAnalysis
from gemseo.uncertainty.sensitivity.sobol import SobolAnalysis
from gemseo.uncertainty.sensitivity.sobol import SobolAnalysisMethod
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.util.typing import RealArray

threshold = 3.0


@pytest.fixture(scope="module")
def discipline() -> AnalyticDiscipline:
    """A differentiable discipline y = x1 + 2*x2."""
    return AnalyticDiscipline({"y": "x1 + 2*x2"}, name="my_function")


@pytest.fixture(scope="module")
def parameter_space() -> RandomSpace:
    """The random space of two standard normal variables."""
    space = RandomSpace()
    space.add_variable("x1", OTNormalDistribution_Settings())
    space.add_variable("x2", OTNormalDistribution_Settings())
    return space


@pytest.fixture(scope="module")
def analysis(discipline, parameter_space) -> ROSobolAnalysis:
    """A reliability-oriented Sobol' analysis with a pick-and-freeze design."""
    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    analysis.compute_samples(
        [discipline], parameter_space, {"y_high": y > threshold}, n_samples=500
    )
    analysis.compute_indices()
    return analysis


def test_output_names(analysis) -> None:
    """Check the output names, which are the event names."""
    assert analysis.default_output_names == ["y_high"]


def test_main_method(analysis) -> None:
    """Check the default main method is the first-order Sobol' index."""
    assert analysis.main_method == SobolAnalysisMethod.FIRST


def test_main_indices(analysis) -> None:
    """Check the structure of the main sensitivity indices."""
    main_indices = analysis.main_indices
    assert set(main_indices) == {"y_high"}
    assert len(main_indices["y_high"]) == 1
    assert set(main_indices["y_high"][0]) == {"x1", "x2"}


def test_dataset(analysis) -> None:
    """Check that the standard samples and the indicator are stored in the dataset."""
    dataset = analysis.dataset
    assert isinstance(dataset, IODataset)
    assert set(dataset.input_names) == {"x1", "x2"}
    assert dataset.output_names == ["y_high"]
    assert dataset.misc["use_pick_and_freeze"]
    assert dataset.misc["eval_second_order"]
    # The output is the raw indicator; the IS weights are recomputed from the
    # standard inputs and the design point when the indices are estimated.
    indicator = dataset.get_view(group_names=dataset.output_group).to_numpy()
    assert set(indicator.ravel()) <= {0.0, 1.0}
    assert 0.0 < indicator.mean() < 1.0


def test_budget_includes_form(analysis) -> None:
    """Check that n_samples is the total budget, FORM evaluations included.

    With a single event, the whole sampling budget left after FORM is spent on it.
    The pick-and-freeze design uses the largest sampling size N
    such that N(2+d) does not exceed that budget (here d=2, so 2+d=4).
    """
    dataset = analysis.dataset
    sample_size = dataset.misc["sample_size"]["y_high"]
    assert sample_size >= 1
    # The design has N(2+d) rows with d=2, i.e. 4*N rows.
    assert len(dataset) == 4 * sample_size
    # The whole design fits within the n_samples=500 budget.
    assert len(dataset) <= 500


def test_compute_samples_with_random_space(discipline) -> None:
    """Check that compute_samples() accepts a RandomSpace, not only a ParameterSpace.

    Regression test: compute_samples() used to read `random_space.distribution`,
    an attribute only available on ParameterSpace, after the FORM pass had already
    run, raising AttributeError when a RandomSpace was passed.
    """
    space = RandomSpace()
    space.add_variable("x1", OTNormalDistribution_Settings())
    space.add_variable("x2", OTNormalDistribution_Settings())

    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    dataset = analysis.compute_samples(
        [discipline], space, {"y_high": y > threshold}, n_samples=500
    )
    assert set(dataset.input_names) == {"x1", "x2"}
    assert dataset.output_names == ["y_high"]


def test_compute_second_order_false(discipline, parameter_space) -> None:
    """Check that a pick-and-freeze design can skip second-order indices."""
    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    dataset = analysis.compute_samples(
        [discipline],
        parameter_space,
        {"y_high": y > threshold},
        n_samples=500,
        compute_second_order=False,
    )
    assert dataset.misc["eval_second_order"] is False
    indices = analysis.compute_indices()
    assert set(indices.first) == {"y_high"}
    assert indices.second == {}


def test_high_dimension_sampling_factor() -> None:
    """Check the pick-and-freeze design size when the dimension exceeds 2.

    For d>2, computing second-order indices doubles the sampling factor:
    the design has N(2+2d) rows instead of N(2+d).
    """
    discipline = AnalyticDiscipline({"y": "x1 + 2*x2 + 3*x3"}, name="my_function")
    space = RandomSpace()
    for name in ("x1", "x2", "x3"):
        space.add_variable(name, OTNormalDistribution_Settings())

    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    dataset = analysis.compute_samples(
        [discipline], space, {"y_high": y > threshold}, n_samples=2000
    )
    sample_size = dataset.misc["sample_size"]["y_high"]
    assert len(dataset) == sample_size * (2 + 2 * 3)

    analysis_no_second_order = ROSobolAnalysis()
    y = analysis_no_second_order.get_event_variables("y")
    dataset_no_second_order = analysis_no_second_order.compute_samples(
        [discipline],
        space,
        {"y_high": y > threshold},
        n_samples=2000,
        compute_second_order=False,
    )
    sample_size_no_second_order = dataset_no_second_order.misc["sample_size"]["y_high"]
    assert len(dataset_no_second_order) == sample_size_no_second_order * (2 + 3)


def test_too_small_n_samples(discipline, parameter_space, snapshot) -> None:
    """Check that a budget too small to sample after FORM raises an error."""
    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    with assert_exception(ValueError, snapshot):
        analysis.compute_samples(
            [discipline], parameter_space, {"y_high": y > threshold}, n_samples=1
        )


def test_seed_reproducibility_pick_and_freeze(discipline, parameter_space) -> None:
    """Check that seed reproducibly controls the pick-and-freeze design.

    Regression test: the pick-and-freeze branch used to call `OTSobolDOE` outside
    of a `seed_ot_random_generator` context, making the `seed` argument inert.
    """

    def get_input_data(seed):
        analysis = ROSobolAnalysis()
        y = analysis.get_event_variables("y")
        dataset = analysis.compute_samples(
            [discipline],
            parameter_space,
            {"y_high": y > threshold},
            n_samples=100,
            seed=seed,
        )
        return dataset.get_view(group_names=dataset.input_group).to_numpy()

    assert_array_equal(get_input_data(1), get_input_data(1))
    assert not (get_input_data(1) == get_input_data(2)).all()


def test_small_budget_accepted_by_rank_not_pick_and_freeze(
    discipline, parameter_space, snapshot
) -> None:
    """Check that the minimum-budget guard only applies to the pick-and-freeze path.

    Regression test: the guard used the pick-and-freeze sampling factor even when
    the Rank/i.i.d. design was selected, which only needs one sample per event.
    """
    scenario = ReliabilityScenario([discipline], parameter_space)
    y = scenario.get_event_variables("y")
    scenario.add_event(y > threshold, "y_high")
    scenario.execute(OT_FORM_Settings())
    result = scenario.event_name_to_reliability_result["y_high"]
    n_form_evaluations = result.raw_result.getOptimizationResult().getCallsNumber()
    # A budget of exactly 1 sample after FORM: accepted by Rank, rejected by
    # pick-and-freeze (which needs N(2+d)=4 samples here, with d=2).
    n_samples = n_form_evaluations + 1

    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    dataset = analysis.compute_samples(
        [discipline],
        parameter_space,
        {"y_high": y > threshold},
        n_samples=n_samples,
        algo_settings=OT_MC_Settings(),
    )
    assert len(dataset) >= 1

    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    with assert_exception(ValueError, snapshot):
        analysis.compute_samples(
            [discipline],
            parameter_space,
            {"y_high": y > threshold},
            n_samples=n_samples,
        )


def test_vector_valued_event_output_raises(
    parameter_space, monkeypatch, snapshot
) -> None:
    """Check that a vector-valued event output raises a clear error.

    FORM only supports scalar limit-state functions,
    so a vector-valued event output is already rejected by FORM;
    the FORM step is bypassed here (via monkeypatching the design-point search)
    to exercise the dedicated check in the model-evaluation phase.
    """

    def vector_func(x1, x2):
        y = array([x1[0] + x2[0], x1[0] - x2[0]])
        return y  # noqa: RET504

    vector_discipline = AutoPyDiscipline(py_func=vector_func, use_arrays=True)

    def fake_compute_standard_design_point(*args, **kwargs):
        return zeros(parameter_space.dimension), 0

    monkeypatch.setattr(
        ROSobolAnalysis,
        "_ROSobolAnalysis__compute_standard_design_point",
        staticmethod(fake_compute_standard_design_point),
    )

    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    with assert_exception(ValueError, snapshot):
        analysis.compute_samples(
            [vector_discipline],
            parameter_space,
            {"y_high": y > threshold},
            n_samples=100,
        )


def test_indices_orders(analysis) -> None:
    """Check that first-, second- and total-order indices are populated."""
    indices = analysis.indices
    for input_name in ("x1", "x2"):
        assert indices.first["y_high"][0][input_name].size == 1
        assert indices.total["y_high"][0][input_name].size == 1

    assert set(indices.second["y_high"][0]) == {"x1", "x2"}


def test_first_lower_than_total(analysis) -> None:
    """Check that the first-order index does not exceed the total-order one."""
    first = analysis.indices.first["y_high"][0]
    total = analysis.indices.total["y_high"][0]
    for input_name in ("x1", "x2"):
        assert first[input_name][0] <= total[input_name][0] + 1e-9


def test_x2_more_influential(analysis) -> None:
    """Check that x2 (coefficient 2) is more influential than x1 on the event."""
    total = analysis.indices.total["y_high"][0]
    assert total["x2"][0] > total["x1"][0]


def test_probability(analysis) -> None:
    """Check that the estimated event probability matches the analytic value."""
    analytic = norm.sf(threshold / sqrt(5.0))
    probability = analysis.dataset.misc["probability"]["y_high"]
    assert probability == pytest.approx(analytic, rel=0.2)


def test_rank_based(discipline, parameter_space) -> None:
    """Check the rank-based estimation from independent samples."""
    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    analysis.compute_samples(
        [discipline],
        parameter_space,
        {"y_high": y > threshold},
        n_samples=2000,
        algo_settings=OT_MC_Settings(),
    )
    # The i.i.d. budget is n_samples minus the FORM evaluations.
    dataset = analysis.dataset
    assert dataset.misc["sample_size"]["y_high"] == len(dataset)
    assert len(dataset) < 2000
    indices = analysis.compute_indices()
    assert set(indices.first) == {"y_high"}
    # The rank-based algorithm does not provide second- or total-order indices.
    assert indices.second == {}


def test_inconsistent_algorithm_with_iid_samples(
    discipline, parameter_space, snapshot
) -> None:
    """Check that a pick-and-freeze algorithm rejects independent samples."""
    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    analysis.compute_samples(
        [discipline],
        parameter_space,
        {"y_high": y > threshold},
        n_samples=2000,
        algo_settings=OT_MC_Settings(),
    )
    with assert_exception(ValueError, snapshot):
        analysis.compute_indices(algo=SobolAnalysis.Algorithm.SALTELLI)


def test_form_design_point_consistency(discipline, parameter_space, analysis) -> None:
    """Cross-check the embedded FORM design point against a direct FORM study."""
    scenario = ReliabilityScenario([discipline], parameter_space)
    y = scenario.get_event_variables("y")
    scenario.add_event(y > threshold, "y_high")
    scenario.execute(OT_FORM_Settings())
    result = scenario.event_name_to_reliability_result["y_high"]
    assert_almost_equal(
        analysis.dataset.misc["design_point"]["y_high"],
        result.design_point.standard,
    )


def test_several_events(discipline, parameter_space, analysis) -> None:
    """Check that several events share the n_samples budget."""
    multi = ROSobolAnalysis()
    y = multi.get_event_variables("y")
    multi.compute_samples(
        [discipline],
        parameter_space,
        {"y_high": y > threshold, "y_higher": y > 2 * threshold},
        n_samples=1000,
    )
    indices = multi.compute_indices()
    assert multi.default_output_names == ["y_high", "y_higher"]
    assert set(indices.first) == {"y_high", "y_higher"}
    dataset = multi.dataset
    # Each event keeps its own design point, sample size and probability.
    assert set(dataset.misc["sample_size"]) == {"y_high", "y_higher"}
    assert set(dataset.misc["design_point"]) == {"y_high", "y_higher"}
    # n_samples is the total budget over all the events: the total number of
    # model evaluations (all the events' designs) stays within n_samples=1000.
    assert len(dataset) <= 1000
    # The budget is shared, so each event gets a smaller design than when a
    # single event uses the whole n_samples=500 budget.
    single_event_size = analysis.dataset.misc["sample_size"]["y_high"]
    for event_name in ("y_high", "y_higher"):
        assert dataset.misc["sample_size"][event_name] < single_event_size
    # The rarer event y > 2*threshold is less probable.
    assert (
        dataset.misc["probability"]["y_higher"] < dataset.misc["probability"]["y_high"]
    )
    # x2 (coefficient 2) is the more influential input on both events.
    for event_name in ("y_high", "y_higher"):
        total = indices.total[event_name][0]
        assert total["x2"][0] > total["x1"][0]


def test_inconsistent_algorithm(analysis, snapshot) -> None:
    """Check that a rank algorithm rejects a pick-and-freeze design."""
    with assert_exception(ValueError, snapshot):
        analysis.compute_indices(algo=SobolAnalysis.Algorithm.RANK)


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("sort_by_total", [False, True])
@pytest.mark.parametrize("kwargs", [{}, {"title": "foo"}])
def test_plot(analysis, sort, sort_by_total, kwargs) -> None:
    """Check the dedicated error-bar visualization method."""
    fig = analysis.plot(
        "y_high", save=False, sort=sort, sort_by_total=sort_by_total, **kwargs
    )
    assert isinstance(fig, Figure)
    title = kwargs.get("title", "Sobol' indices for the event 'y_high'")
    probability = analysis.dataset.misc["probability"]["y_high"]
    assert fig.axes[0].get_title() == f"{title}\nP={probability:.1e}"


def test_plot_rank_based(discipline, parameter_space) -> None:
    """Check that plot() works when only first-order indices are available."""
    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    analysis.compute_samples(
        [discipline],
        parameter_space,
        {"y_high": y > threshold},
        n_samples=2000,
        algo_settings=OT_MC_Settings(),
    )
    analysis.compute_indices()
    fig = analysis.plot("y_high", save=False)
    assert isinstance(fig, Figure)


def test_get_intervals(analysis) -> None:
    """Check the structure of the confidence intervals of the Sobol' indices."""
    for first_order in [True, False]:
        intervals = analysis.get_intervals(first_order=first_order)
        assert set(intervals) == {"y_high"}
        assert len(intervals["y_high"]) == 1
        event_intervals = intervals["y_high"][0]
        assert set(event_intervals) == {"x1", "x2"}
        for input_name in ("x1", "x2"):
            assert event_intervals[input_name].shape == (2, 1)


def test_sort_input_variables(analysis) -> None:
    """Check that the inputs are sorted by decreasing influence on the event."""
    assert analysis.sort_input_variables("y_high") == ["x2", "x1"]


def test_factory() -> None:
    """Check that the high-level API creates an ROSobolAnalysis."""
    assert isinstance(create_sensitivity_analysis("ROSobol"), ROSobolAnalysis)


# Ground truth: the Sobol' indices of the indicator of a linear limit state
# y = a . u > t with standard normal inputs are available in closed form.
# With beta = t / |a| and p = Phi(-beta),
# E[1_F(U) 1_F(U')] for two standard normal vectors sharing a set K of coordinates
# is the bivariate normal CDF Phi_2(-beta, -beta; rho)
# with rho = sum_{k in K} a_k^2 / |a|^2.
linear_coefficients = array([0.8, 0.5, 0.33])
linear_threshold = 2.0
linear_input_names = ("x1", "x2", "x3")

# The budget of the pick-and-freeze analysis of the linear limit state:
# at least 3000 rows for each of the 2 + 2d blocks of the design
# with the second-order indices, plus a margin for the FORM evaluations.
linear_n_samples = (2 + 2 * len(linear_input_names)) * 3_000 + 200

# The budget of the rank-based analysis of the linear limit state,
# spent on independent rows:
# the standard deviations of its first-order indices are at most 0.0055,
# measured over 20 designs of this size.
linear_rank_n_samples = 24_000


def compute_exact_linear_indices() -> tuple[float, RealArray, RealArray, RealArray]:
    """Compute the exact Sobol' indices of the indicator of the linear limit state.

    Returns:
        The failure probability,
        the first-order indices,
        the total-order indices
        and the second-order indices (shaped as `(3, 3)`).
    """
    squared_norm = linear_coefficients @ linear_coefficients
    beta = linear_threshold / sqrt(squared_norm)
    probability = norm.cdf(-beta)
    variance = probability * (1.0 - probability)

    def compute_closed_variance(*coordinates: int) -> float:
        rho = sum(linear_coefficients[k] ** 2 for k in coordinates) / squared_norm
        cdf = multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]).cdf
        return cdf([-beta, -beta]) - probability**2

    first = array([compute_closed_variance(i) / variance for i in range(3)])
    total = array([
        1.0 - compute_closed_variance(*(k for k in range(3) if k != i)) / variance
        for i in range(3)
    ])
    second = zeros((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            second[i, j] = second[j, i] = (
                compute_closed_variance(i, j) / variance - first[i] - first[j]
            )

    return probability, first, total, second


@pytest.fixture(scope="module")
def linear_discipline() -> AnalyticDiscipline:
    """The linear discipline y = 0.8*x1 + 0.5*x2 + 0.33*x3."""
    return AnalyticDiscipline({"y": "0.8*x1 + 0.5*x2 + 0.33*x3"}, name="linear")


@pytest.fixture(scope="module")
def linear_space() -> RandomSpace:
    """The uncertain space of three standard normal variables."""
    space = RandomSpace()
    for name in linear_input_names:
        space.add_variable(name, OTNormalDistribution_Settings())
    return space


@pytest.fixture(scope="module")
def linear_analysis(linear_discipline, linear_space) -> ROSobolAnalysis:
    """A reliability-oriented Sobol' analysis of the linear limit state.

    It uses a pick-and-freeze design.
    """
    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    analysis.compute_samples(
        [linear_discipline],
        linear_space,
        {"y_high": y > linear_threshold},
        n_samples=linear_n_samples,
    )
    return analysis


@pytest.fixture(scope="module")
def linear_rank_analysis(linear_discipline, linear_space) -> ROSobolAnalysis:
    """A reliability-oriented Sobol' analysis of the linear limit state.

    Its design has independent rows.
    """
    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    analysis.compute_samples(
        [linear_discipline],
        linear_space,
        {"y_high": y > linear_threshold},
        n_samples=linear_rank_n_samples,
        algo_settings=OT_MC_Settings(),
    )
    return analysis


def get_indices_array(indices: Mapping[str, RealArray]) -> RealArray:
    """Flatten the indices of the three inputs into an array.

    Args:
        indices: The indices indexed by input name.

    Returns:
        The indices of the three inputs.
    """
    return array([indices[name][0] for name in linear_input_names])


def test_linear_probability(linear_analysis) -> None:
    """Check the IS estimate of the failure probability of the linear limit state."""
    probability, _, _, _ = compute_exact_linear_indices()
    estimate = linear_analysis.dataset.misc["probability"]["y_high"]
    assert estimate == pytest.approx(probability, rel=0.05)


# The tolerances are about four standard deviations of the estimators,
# measured over 40 designs of the same size:
# 0.02 for the first-, total- and second-order indices
# of Saltelli, Mauntz-Kucherenko and Martinez,
# 0.03, 0.02 and 0.07 for those of Jansen.
@pytest.mark.parametrize(
    (
        "algo",
        "first_tolerance",
        "total_tolerance",
        "second_tolerance",
        "max_interval_width",
    ),
    [
        (ROSobolAnalysis.Algorithm.SALTELLI, 0.05, 0.05, 0.1, 0.3),
        (ROSobolAnalysis.Algorithm.MAUNTZ_KUCHERENKO, 0.05, 0.05, 0.1, 0.3),
        (ROSobolAnalysis.Algorithm.MARTINEZ, 0.05, 0.05, 0.1, 0.3),
        (ROSobolAnalysis.Algorithm.JANSEN, 0.12, 0.08, 0.3, 1.0),
    ],
)
def test_pick_and_freeze_indices_match_exact_values(
    linear_analysis,
    algo,
    first_tolerance,
    total_tolerance,
    second_tolerance,
    max_interval_width,
) -> None:
    """Check the weighted pick-and-freeze estimators against the exact indices.

    Regression test: the IS-reweighted indicator w * 1_F used to be passed
    to the OpenTURNS estimators as a model output,
    which estimates the Sobol' indices of w * 1_F under the auxiliary density
    rather than those of 1_F under the true density;
    on this design, the first-order index of x1 was then off
    by more than the first-order tolerance of Saltelli,
    with a bias that does not vanish with the sample size.
    """
    _, first, total, second = compute_exact_linear_indices()
    indices = linear_analysis.compute_indices(
        algo=algo, use_asymptotic_distributions=False, seed=1
    )
    assert_allclose(
        get_indices_array(indices.first["y_high"][0]),
        first,
        atol=first_tolerance,
        rtol=0,
    )
    assert_allclose(
        get_indices_array(indices.total["y_high"][0]),
        total,
        atol=total_tolerance,
        rtol=0,
    )
    estimated_second = indices.second["y_high"][0]
    for i, first_name in enumerate(linear_input_names):
        for j, second_name in enumerate(linear_input_names):
            assert estimated_second[first_name][second_name][0, 0] == pytest.approx(
                second[i, j], abs=second_tolerance
            )

    # The bootstrap intervals contain the estimates.
    for first_order in (True, False):
        intervals = linear_analysis.get_intervals(first_order=first_order)["y_high"][0]
        estimates = indices.first if first_order else indices.total
        for name in linear_input_names:
            lower, upper = intervals[name][:, 0]
            assert lower <= estimates["y_high"][0][name][0] <= upper
            assert upper - lower < max_interval_width


def test_rank_indices_match_exact_values(linear_rank_analysis) -> None:
    """Check the weighted rank-based estimator against the exact first-order indices.

    The importance-sampling variant of the rank-based estimator
    pairs each sample with its successor along a coordinate
    and weights the pair with the shared coordinate counted once.
    """
    _, first, _, _ = compute_exact_linear_indices()
    indices = linear_rank_analysis.compute_indices(n_replicates=20)
    assert_allclose(
        get_indices_array(indices.first["y_high"][0]), first, atol=0.025, rtol=0
    )
    assert indices.total == {}
    assert indices.second == {}
    intervals = linear_rank_analysis.get_intervals()["y_high"][0]
    for name in linear_input_names:
        lower, upper = intervals[name][:, 0]
        assert lower <= indices.first["y_high"][0][name][0] <= upper


@pytest.mark.parametrize(
    "algo",
    [
        ROSobolAnalysis.Algorithm.SALTELLI,
        ROSobolAnalysis.Algorithm.MAUNTZ_KUCHERENKO,
        ROSobolAnalysis.Algorithm.MARTINEZ,
        ROSobolAnalysis.Algorithm.JANSEN,
    ],
)
def test_asymptotic_intervals_match_bootstrap(linear_analysis, algo) -> None:
    """Check the delta-method intervals against the bootstrap intervals.

    The asymptotic half-widths are within a factor two of the bootstrap ones
    and the estimates do not depend on the interval method.
    """
    indices = linear_analysis.compute_indices(algo=algo)
    asymptotic = {
        first_order: linear_analysis.get_intervals(first_order=first_order)["y_high"][0]
        for first_order in (True, False)
    }
    bootstrap_indices = linear_analysis.compute_indices(
        algo=algo, use_asymptotic_distributions=False, n_replicates=200, seed=1
    )
    for first_order in (True, False):
        bootstrap = linear_analysis.get_intervals(first_order=first_order)["y_high"][0]
        estimates = (indices.first if first_order else indices.total)["y_high"][0]
        bootstrap_estimates = (
            bootstrap_indices.first if first_order else bootstrap_indices.total
        )["y_high"][0]
        for name in linear_input_names:
            estimate = estimates[name][0]
            assert estimate == bootstrap_estimates[name][0]
            lower, upper = asymptotic[first_order][name][:, 0]
            assert lower < estimate < upper
            # The asymptotic interval is symmetric around the estimate.
            assert upper - estimate == pytest.approx(estimate - lower)
            bootstrap_half_width = (bootstrap[name][1, 0] - bootstrap[name][0, 0]) / 2
            assert 0.5 < (upper - lower) / 2 / bootstrap_half_width < 2.0


@pytest.fixture
def restored_analysis(analysis) -> Iterator[ROSobolAnalysis]:
    """The module-scoped analysis, whose indices are recomputed after the test."""
    yield analysis
    analysis.compute_indices()


def test_asymptotic_intervals_are_deterministic(restored_analysis) -> None:
    """Check that the asymptotic intervals do not depend on the bootstrap settings."""
    analysis = restored_analysis
    analysis.compute_indices(seed=1, n_replicates=10)
    intervals_1 = analysis.get_intervals()["y_high"][0]["x1"]
    analysis.compute_indices(seed=2, n_replicates=0)
    intervals_2 = analysis.get_intervals()["y_high"][0]["x1"]
    assert_array_equal(intervals_1, intervals_2)


def test_intervals_without_replicates(restored_analysis, snapshot) -> None:
    """Check that the bootstrap requires at least one replicate.

    OpenTURNS refuses a bootstrap size of zero too,
    so `SobolAnalysis` already raises in this case.
    """
    with assert_exception(ValueError, snapshot):
        restored_analysis.compute_indices(
            use_asymptotic_distributions=False, n_replicates=0
        )


def test_intervals_seed(restored_analysis) -> None:
    """Check that the seed controls the bootstrap intervals."""
    analysis = restored_analysis
    analysis.compute_indices(use_asymptotic_distributions=False, seed=1)
    intervals_1 = analysis.get_intervals()["y_high"][0]["x1"]
    analysis.compute_indices(use_asymptotic_distributions=False, seed=1)
    intervals_1_again = analysis.get_intervals()["y_high"][0]["x1"]
    analysis.compute_indices(use_asymptotic_distributions=False, seed=2)
    intervals_2 = analysis.get_intervals()["y_high"][0]["x1"]
    assert_array_equal(intervals_1, intervals_1_again)
    assert not (intervals_1 == intervals_2).all()


def test_rank_intervals_contain_the_estimates(linear_rank_analysis) -> None:
    """Check that the rank bootstrap intervals contain the estimates and are narrow.

    This holds because the bootstrap draws its rows without replacement.
    With replacement,
    a row would be ranked next to its own duplicate and paired with itself,
    which would pull every replicate towards one,
    push the whole interval above the estimate
    and widen it by an order of magnitude.
    """
    indices = linear_rank_analysis.compute_indices(n_replicates=50, seed=1)
    intervals = linear_rank_analysis.get_intervals()["y_high"][0]
    for name in linear_input_names:
        lower, upper = intervals[name][:, 0]
        estimate = indices.first["y_high"][0][name][0]
        assert lower < estimate < upper
        assert upper - lower < 0.05


def test_output_variances(analysis) -> None:
    """Check that the variance of an event is that of its indicator over block A.

    This is the variance the indices are divided by,
    i.e. N/(N-1) p_A(1-p_A) with p_A the IS estimate of the probability
    over the rows of the block A.
    """
    dataset = analysis.dataset
    sample_size = dataset.misc["sample_size"]["y_high"]
    design_point = dataset.misc["design_point"]["y_high"]
    inputs = dataset.get_view(group_names=dataset.input_group).to_numpy()
    indicator = dataset.get_view(group_names=dataset.output_group).to_numpy()[:, 0]
    rows_a = slice(0, sample_size)
    weights = exp(-inputs[rows_a] @ design_point + design_point @ design_point / 2)
    probability = (weights * indicator[rows_a]).mean()
    variance = analysis.output_variances["y_high"][0]
    assert variance == pytest.approx(
        sample_size / (sample_size - 1) * probability * (1.0 - probability)
    )
    assert analysis.output_standard_deviations["y_high"][0] == pytest.approx(
        sqrt(variance)
    )


def test_unscale_indices(analysis) -> None:
    """Check that the indices are unscaled by the variance of the indicator."""
    indices = analysis.indices.first
    variance = analysis.output_variances["y_high"][0]
    unscaled = analysis.unscale_indices(indices)
    assert unscaled["y_high"][0]["x1"][0] == pytest.approx(
        indices["y_high"][0]["x1"][0] * variance
    )
    unscaled = analysis.unscale_indices(indices, use_variance=False)
    assert unscaled["y_high"][0]["x1"][0] == pytest.approx(
        sqrt(indices["y_high"][0]["x1"][0] * variance)
    )


def test_plot_second_order(analysis) -> None:
    """Check the heat map of the second-order indices of an event."""
    heatmap = analysis.plot_second_order("y_high")
    assert heatmap.dataset.variable_names == ["x1", "x2"]


@pytest.mark.parametrize("constant", [0.0, 1.0])
def test_constant_event_has_zero_variance(
    discipline, parameter_space, constant
) -> None:
    """Check that an event with a constant indicator has a zero variance and no indices.

    The event occurs at no sample or at every sample,
    so no estimator is built for it.
    """
    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    dataset = analysis.compute_samples(
        [discipline], parameter_space, {"y_high": y > threshold}, n_samples=500
    )
    output_columns = dataset.get_view(group_names=dataset.output_group).columns
    dataset[output_columns] = constant
    indices = analysis.compute_indices()
    assert analysis.output_variances["y_high"][0] == 0.0
    assert indices.first["y_high"] == [None]
    assert indices.total["y_high"] == [None]
    assert indices.second["y_high"] == [None]
    assert analysis.unscale_indices(indices.first)["y_high"] == [None]
    assert analysis.unscale_indices(indices.total)["y_high"] == [None]
    assert analysis.unscale_indices(indices.second)["y_high"] == [None]


def test_plot_second_order_of_constant_event(
    discipline, parameter_space, snapshot
) -> None:
    """Check that the second-order indices of a constant event cannot be plotted."""
    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    dataset = analysis.compute_samples(
        [discipline], parameter_space, {"y_high": y > threshold}, n_samples=500
    )
    output_columns = dataset.get_view(group_names=dataset.output_group).columns
    dataset[output_columns] = 0.0
    analysis.compute_indices()
    with assert_exception(ValueError, snapshot):
        analysis.plot_second_order("y_high")


def test_plot_second_order_not_computed(discipline, parameter_space, snapshot) -> None:
    """Check that missing second-order indices cannot be plotted."""
    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    analysis.compute_samples(
        [discipline],
        parameter_space,
        {"y_high": y > threshold},
        n_samples=500,
        compute_second_order=False,
    )
    analysis.compute_indices()
    with assert_exception(ValueError, snapshot):
        analysis.plot_second_order("y_high")


pick_and_freeze_estimators = {
    ROSobolAnalysis.Algorithm.SALTELLI: SaltelliSensitivityAlgorithm,
    ROSobolAnalysis.Algorithm.JANSEN: JansenSensitivityAlgorithm,
    ROSobolAnalysis.Algorithm.MAUNTZ_KUCHERENKO: (MauntzKucherenkoSensitivityAlgorithm),
    ROSobolAnalysis.Algorithm.MARTINEZ: MartinezSensitivityAlgorithm,
}
"""The OpenTURNS estimators associated with the pick-and-freeze algorithms."""


def compute_nonlinear_indicator(standard_samples: RealArray) -> RealArray:
    """Compute the indicator of an event with a nonlinear limit state.

    Args:
        standard_samples: The input samples in the standard space.

    Returns:
        The indicator of every sample.
    """
    coefficients = array([1.0, 0.7, 0.3])[: standard_samples.shape[1]]
    return (
        standard_samples @ coefficients
        + 0.5 * standard_samples[:, 0] * standard_samples[:, 1]
        > 1.0
    ).astype(float)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("eval_second_order", [False, True])
@pytest.mark.parametrize("algorithm", pick_and_freeze_estimators)
def test_pick_and_freeze_at_origin_matches_openturns(
    algorithm, eval_second_order, dimension
) -> None:
    """Check that the estimators are the OpenTURNS ones for a design point at 0.

    All the importance-sampling weights are then equal to one.
    For d=2, OpenTURNS uses the blocks E^2 and E^1 as the blocks C^1 and C^2.
    """
    sample_size = 500
    second = eval_second_order and dimension > 2
    # OTSobolDOE draws from the global OpenTURNS random generator.
    with seed_ot_random_generator(1):
        unit_samples = OTSobolDOE().generate_samples(
            dimension,
            OT_SOBOL_INDICES_Settings(
                n_samples=sample_size * (2 + dimension * (1 + second)),
                eval_second_order=eval_second_order,
            ),
        )
    standard_samples = norm.ppf(unit_samples)
    indicator = compute_nonlinear_indicator(standard_samples)
    estimator = ISSobolIndicesEstimator(
        algorithm,
        standard_samples,
        indicator,
        zeros(dimension),
        sample_size,
        eval_second_order,
        0.95,
        True,
        100,
        1,
    )
    openturns_estimator = pick_and_freeze_estimators[algorithm](
        Sample(standard_samples), Sample(indicator[:, newaxis]), sample_size
    )
    assert_almost_equal(
        estimator.first_order_indices,
        array(openturns_estimator.getFirstOrderIndices()),
        decimal=10,
    )
    assert_almost_equal(
        estimator.total_order_indices,
        array(openturns_estimator.getTotalOrderIndices()),
        decimal=10,
    )
    if eval_second_order:
        indices = triu_indices(dimension, 1)
        assert_almost_equal(
            estimator.second_order_indices[indices],
            array(openturns_estimator.getSecondOrderIndices())[indices],
            decimal=10,
        )


def test_rank_at_origin_matches_openturns() -> None:
    """Check that the rank-based estimator is the OpenTURNS one at the origin."""
    standard_samples = default_rng(1).standard_normal((2500, 3))
    indicator = compute_nonlinear_indicator(standard_samples)
    estimator = ISSobolIndicesEstimator(
        ROSobolAnalysis.Algorithm.RANK,
        standard_samples,
        indicator,
        zeros(3),
        len(standard_samples),
        False,
        0.95,
        False,
        10,
        1,
    )
    openturns_estimator = RankSobolSensitivityAlgorithm(
        Sample(standard_samples), Sample(indicator[:, newaxis])
    )
    assert_almost_equal(
        estimator.first_order_indices,
        array(openturns_estimator.getFirstOrderIndices()),
        decimal=10,
    )


def build_estimator(
    algo: ROSobolAnalysis.Algorithm,
    indicator_blocks: Sequence[RealArray],
    use_asymptotic_distributions: bool,
    n_replicates: int = 100,
) -> ISSobolIndicesEstimator:
    """Build an estimator of a design with a design point at the origin.

    Args:
        algo: The Sobol' estimation algorithm.
        indicator_blocks: The indicator of each block of the design.
        use_asymptotic_distributions: Whether to use the asymptotic intervals.
        n_replicates: The number of bootstrap replicates.

    Returns:
        The estimator.
    """
    sample_size = len(indicator_blocks[0])
    standard_samples = default_rng(0).normal(
        size=(sample_size * len(indicator_blocks), 2)
    )
    return ISSobolIndicesEstimator(
        algo,
        standard_samples,
        concatenate(indicator_blocks),
        zeros(2),
        sample_size,
        False,
        0.95,
        use_asymptotic_distributions,
        n_replicates,
        1,
    )


def create_martinez_blocks() -> list[RealArray]:
    """Create the indicator blocks of a design whose block B is constant.

    Returns:
        The blocks A, B, E^1 and E^2.
    """
    alternating = tile([0.0, 1.0], 50)
    return [alternating, zeros(100), roll(alternating, 1), alternating]


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("use_asymptotic_distributions", [True, False])
def test_martinez_with_constant_block(use_asymptotic_distributions) -> None:
    """Check that a constant block makes the Martinez first-order indices NaN."""
    estimator = build_estimator(
        ROSobolAnalysis.Algorithm.MARTINEZ,
        create_martinez_blocks(),
        use_asymptotic_distributions,
        10,
    )
    assert isnan(estimator.first_order_indices).all()
    assert isnan(estimator.first_order_interval).all()
    assert isfinite(estimator.total_order_indices).all()
    assert isfinite(estimator.total_order_interval).all()
    assert estimator.n_degenerate_replicates == (
        0 if use_asymptotic_distributions else 10
    )


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_saltelli_bootstrap_discards_degenerate_replicates() -> None:
    """Check that the replicates whose block A is constant are discarded."""
    rng = default_rng(2)
    block_a = zeros(20)
    block_a[0] = 1.0
    blocks = [block_a] + [rng.integers(0, 2, 20).astype(float) for _ in range(3)]
    estimator = build_estimator(ROSobolAnalysis.Algorithm.SALTELLI, blocks, False)
    assert 0 < estimator.n_degenerate_replicates < 100
    assert isfinite(estimator.first_order_interval).all()
    assert isfinite(estimator.total_order_interval).all()


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_rank_bootstrap_discards_degenerate_replicates() -> None:
    """Check that the replicates without any failing row are discarded."""
    indicator = zeros(20)
    indicator[0] = 1.0
    estimator = build_estimator(ROSobolAnalysis.Algorithm.RANK, [indicator], False)
    assert 0 < estimator.n_degenerate_replicates < 100
    assert isfinite(estimator.first_order_interval).all()


def test_undefined_indices_are_logged(discipline, parameter_space, caplog) -> None:
    """Check the warnings about the undefined indices and discarded replicates."""
    analysis = ROSobolAnalysis()
    y = analysis.get_event_variables("y")
    dataset = analysis.compute_samples(
        [discipline], parameter_space, {"y_high": y > threshold}, n_samples=500
    )
    sample_size = dataset.misc["sample_size"]["y_high"]
    column = dataset.columns.get_loc(
        dataset.get_view(group_names=dataset.output_group).columns[0]
    )
    dataset.iloc[sample_size : 2 * sample_size, column] = 0.0
    indices = analysis.compute_indices(
        algo="Martinez", use_asymptotic_distributions=False, n_replicates=10
    )
    assert isnan(indices.first["y_high"][0]["x1"]).all()
    assert "Some Sobol' indices of the event 'y_high' are undefined" in caplog.text
    assert "10 of the 10 bootstrap replicates of the event 'y_high'" in caplog.text
