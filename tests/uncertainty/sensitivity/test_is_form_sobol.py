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

import pytest
from matplotlib.figure import Figure
from numpy import array
from numpy import sqrt
from numpy import zeros
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from scipy.stats import norm

from gemseo.dataset.io_dataset import IODataset
from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.discipline.auto_py import AutoPyDiscipline
from gemseo.space.parameter import ParameterSpace
from gemseo.uncertainty import create_sensitivity_analysis
from gemseo.uncertainty.distribution.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.reliability.openturns.form_settings import OT_FORM_Settings
from gemseo.uncertainty.reliability.openturns.mc_settings import OT_MC_Settings
from gemseo.uncertainty.reliability.scenario import ReliabilityScenario
from gemseo.uncertainty.sensitivity.is_form_sobol import ISFORMSobolAnalysis
from gemseo.uncertainty.sensitivity.sobol import SobolAnalysis
from gemseo.uncertainty.sensitivity.sobol import SobolAnalysisMethod
from gemseo.util.testing.helper import assert_exception

THRESHOLD = 3.0


@pytest.fixture(scope="module")
def discipline() -> AnalyticDiscipline:
    """A differentiable discipline y = x1 + 2*x2."""
    return AnalyticDiscipline({"y": "x1 + 2*x2"}, name="my_function")


@pytest.fixture(scope="module")
def parameter_space() -> ParameterSpace:
    """The uncertain space of two standard normal variables."""
    space = ParameterSpace()
    space.add_random_variable("x1", OTNormalDistribution_Settings())
    space.add_random_variable("x2", OTNormalDistribution_Settings())
    return space


@pytest.fixture(scope="module")
def analysis(discipline, parameter_space) -> ISFORMSobolAnalysis:
    """An IS-FORM-Sobol' analysis with a pick-and-freeze design."""
    analysis = ISFORMSobolAnalysis()
    y = analysis.get_event_variables("y")
    analysis.compute_samples(
        [discipline], parameter_space, {"y_high": y > THRESHOLD}, n_samples=500
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
    """Check that the IS-reweighted samples are stored in the dataset."""
    assert isinstance(analysis.dataset, IODataset)
    assert set(analysis.dataset.input_names) == {"x1", "x2"}
    assert analysis.dataset.output_names == ["y_high"]
    assert analysis.dataset.misc["use_pick_and_freeze"]
    assert analysis.dataset.misc["eval_second_order"]


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


def test_compute_second_order_false(discipline, parameter_space) -> None:
    """Check that a pick-and-freeze design can skip second-order indices."""
    analysis = ISFORMSobolAnalysis()
    y = analysis.get_event_variables("y")
    dataset = analysis.compute_samples(
        [discipline],
        parameter_space,
        {"y_high": y > THRESHOLD},
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
    space = ParameterSpace()
    for name in ("x1", "x2", "x3"):
        space.add_random_variable(name, OTNormalDistribution_Settings())

    analysis = ISFORMSobolAnalysis()
    y = analysis.get_event_variables("y")
    dataset = analysis.compute_samples(
        [discipline], space, {"y_high": y > THRESHOLD}, n_samples=2000
    )
    sample_size = dataset.misc["sample_size"]["y_high"]
    assert len(dataset) == sample_size * (2 + 2 * 3)

    analysis_no_second_order = ISFORMSobolAnalysis()
    y = analysis_no_second_order.get_event_variables("y")
    dataset_no_second_order = analysis_no_second_order.compute_samples(
        [discipline],
        space,
        {"y_high": y > THRESHOLD},
        n_samples=2000,
        compute_second_order=False,
    )
    sample_size_no_second_order = dataset_no_second_order.misc["sample_size"]["y_high"]
    assert len(dataset_no_second_order) == sample_size_no_second_order * (2 + 3)


def test_too_small_n_samples(discipline, parameter_space, snapshot) -> None:
    """Check that a budget too small to sample after FORM raises an error."""
    analysis = ISFORMSobolAnalysis()
    y = analysis.get_event_variables("y")
    with assert_exception(ValueError, snapshot):
        analysis.compute_samples(
            [discipline], parameter_space, {"y_high": y > THRESHOLD}, n_samples=1
        )


def test_seed_reproducibility_pick_and_freeze(discipline, parameter_space) -> None:
    """Check that seed reproducibly controls the pick-and-freeze design.

    Regression test: the pick-and-freeze branch used to call `OTSobolDOE` outside
    of a `seed_ot_random_generator` context, making the `seed` argument inert.
    """

    def get_input_data(seed):
        analysis = ISFORMSobolAnalysis()
        y = analysis.get_event_variables("y")
        dataset = analysis.compute_samples(
            [discipline],
            parameter_space,
            {"y_high": y > THRESHOLD},
            n_samples=100,
            seed=seed,
        )
        return dataset.get_view(group_names=dataset.INPUT_GROUP).to_numpy()

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
    scenario.add_event(y > THRESHOLD, "y_high")
    scenario.execute(OT_FORM_Settings())
    result = scenario.event_name_to_reliability_result["y_high"]
    n_form_evaluations = result.raw_result.getOptimizationResult().getCallsNumber()
    # A budget of exactly 1 sample after FORM: accepted by Rank, rejected by
    # pick-and-freeze (which needs N(2+d)=4 samples here, with d=2).
    n_samples = n_form_evaluations + 1

    analysis = ISFORMSobolAnalysis()
    y = analysis.get_event_variables("y")
    dataset = analysis.compute_samples(
        [discipline],
        parameter_space,
        {"y_high": y > THRESHOLD},
        n_samples=n_samples,
        algo_settings=OT_MC_Settings(),
    )
    assert len(dataset) >= 1

    analysis = ISFORMSobolAnalysis()
    y = analysis.get_event_variables("y")
    with assert_exception(ValueError, snapshot):
        analysis.compute_samples(
            [discipline],
            parameter_space,
            {"y_high": y > THRESHOLD},
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
        ISFORMSobolAnalysis,
        "_ISFORMSobolAnalysis__compute_standard_design_point",
        staticmethod(fake_compute_standard_design_point),
    )

    analysis = ISFORMSobolAnalysis()
    y = analysis.get_event_variables("y")
    with assert_exception(ValueError, snapshot):
        analysis.compute_samples(
            [vector_discipline],
            parameter_space,
            {"y_high": y > THRESHOLD},
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
    analytic = norm.sf(THRESHOLD / sqrt(5.0))
    probability = analysis.dataset.misc["probability"]["y_high"]
    assert probability == pytest.approx(analytic, rel=0.2)


def test_rank_based(discipline, parameter_space) -> None:
    """Check the rank-based estimation from independent samples."""
    analysis = ISFORMSobolAnalysis()
    y = analysis.get_event_variables("y")
    analysis.compute_samples(
        [discipline],
        parameter_space,
        {"y_high": y > THRESHOLD},
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
    analysis = ISFORMSobolAnalysis()
    y = analysis.get_event_variables("y")
    analysis.compute_samples(
        [discipline],
        parameter_space,
        {"y_high": y > THRESHOLD},
        n_samples=2000,
        algo_settings=OT_MC_Settings(),
    )
    with assert_exception(ValueError, snapshot):
        analysis.compute_indices(algo=SobolAnalysis.Algorithm.SALTELLI)


def test_form_design_point_consistency(discipline, parameter_space, analysis) -> None:
    """Cross-check the embedded FORM design point against a direct FORM study."""
    scenario = ReliabilityScenario([discipline], parameter_space)
    y = scenario.get_event_variables("y")
    scenario.add_event(y > THRESHOLD, "y_high")
    scenario.execute(OT_FORM_Settings())
    result = scenario.event_name_to_reliability_result["y_high"]
    assert_almost_equal(
        analysis.dataset.misc["design_point"]["y_high"],
        result.design_point.standard,
    )


def test_several_events(discipline, parameter_space, analysis) -> None:
    """Check that several events share the n_samples budget."""
    multi = ISFORMSobolAnalysis()
    y = multi.get_event_variables("y")
    multi.compute_samples(
        [discipline],
        parameter_space,
        {"y_high": y > THRESHOLD, "y_higher": y > 2 * THRESHOLD},
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
    # The rarer event y > 2*THRESHOLD is less probable.
    assert (
        dataset.misc["probability"]["y_higher"] < dataset.misc["probability"]["y_high"]
    )
    # x2 (coefficient 2) is the more influential input on both events.
    for event_name in ("y_high", "y_higher"):
        total = indices.total[event_name][0]
        assert total["x2"][0] > total["x1"][0]


def test_zero_variance_event_output_yields_none_indices(
    discipline, parameter_space
) -> None:
    """Check that an event with a constant output yields None Sobol' indices.

    When every sample of an event gives the same reweighted indicator value
    (e.g. the event is never triggered over its samples), the event output has
    zero variance and no Sobol' algorithm can be built for it; its indices are None.
    """
    analysis = ISFORMSobolAnalysis()
    y = analysis.get_event_variables("y")
    dataset = analysis.compute_samples(
        [discipline], parameter_space, {"y_high": y > THRESHOLD}, n_samples=500
    )
    # Force the event output to a constant so that its variance is zero.
    output_columns = dataset.get_view(group_names=dataset.OUTPUT_GROUP).columns
    dataset[output_columns] = 0.0
    indices = analysis.compute_indices()
    assert indices.first["y_high"] == [None]
    assert indices.total["y_high"] == [None]
    assert indices.second["y_high"] == [None]


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
    analysis = ISFORMSobolAnalysis()
    y = analysis.get_event_variables("y")
    analysis.compute_samples(
        [discipline],
        parameter_space,
        {"y_high": y > THRESHOLD},
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
    """Check that the high-level API creates an ISFORMSobolAnalysis."""
    assert isinstance(create_sensitivity_analysis("ISFORMSobol"), ISFORMSobolAnalysis)
