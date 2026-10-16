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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Damien Guenot
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import nan
from numpy.testing import assert_equal

from gemseo import execute_algo
from gemseo import execute_post
from gemseo.core.function.array_function import ArrayFunction
from gemseo.optimization.problem import OptimizationProblem
from gemseo.post.opt_history_view import OptHistoryView
from gemseo.post.opt_history_view_settings import OptHistoryView_Settings
from gemseo.space.design import DesignSpace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

dir_path = Path(__file__).parent
power2_path = dir_path / "power2_opt_pb.h5"
power2_nan_path = dir_path / "power2_opt_pb_nan.h5"


def test_get_constraints() -> None:
    """Test that the constraints of the problem are retrieved correctly."""
    problem = OptimizationProblem.from_hdf(power2_path)
    view = OptHistoryView(problem)

    assert len(view._OptHistoryView__get_constraint_history(["toto", "ineq1"])) == 1


@pytest.mark.parametrize(
    "obj_relative",
    [False, True],
)
def test_opt_hist_const(obj_relative, snapshot_matplotlib) -> None:
    """Test that a problem with constraints is properly rendered."""
    problem = OptimizationProblem.from_hdf(power2_path)
    execute_post(
        problem,
        post_name="OptHistoryView",
        show=False,
        save=False,
        variable_names=["x"],
        file_path="power2_2",
        obj_min=0.0,
        obj_max=5.0,
        obj_relative=obj_relative,
    )


@pytest.mark.parametrize(
    "problem_path",
    [power2_nan_path, power2_path],
)
def test_opt_hist_from_database(
    problem_path,
    snapshot_matplotlib,
) -> None:
    """Test the generation of the plots from databases.

    Args:
        problem_path: The path to the hdf5 database of the problem to test.
    """
    problem = OptimizationProblem.from_hdf(problem_path)
    # The use of the default value is deliberate;
    # to check that the JSON grammar works properly.
    execute_post(
        problem, post_name="OptHistoryView", variable_names=(), show=False, save=False
    )


@pytest.mark.parametrize(
    "use_standardized_objective",
    [True, False],
)
def test_common_scenario(
    use_standardized_objective,
    three_length_common_problem,
    snapshot_matplotlib,
) -> None:
    """Check OptHistoryView with objective, standardized or not."""
    opt = OptHistoryView(three_length_common_problem)
    opt.execute(
        OptHistoryView_Settings(
            save=False,
            use_standardized_objective=use_standardized_objective,
        )
    )


@pytest.mark.parametrize(
    "case",
    [1, 2],
)
def test_461(case, snapshot_matplotlib) -> None:
    """Check that OptHistoryView works with the cases mentioned in issue 461.

    1. Design space of dimension 1 and scalar output.
    2. Design space of dimension > 1 and vector output.
    """
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=-2.0, upper_bound=2.0, value=-2.0)
    if case == 2:
        design_space.add_real_variable(
            "y", lower_bound=-2.0, upper_bound=2.0, value=-2.0
        )

    problem = OptimizationProblem(design_space)
    if case == 1:
        problem.objective = ArrayFunction(lambda x: x[0] ** 2, name="func")
    elif case == 2:
        problem.objective = problem.objective = ArrayFunction(
            lambda x: array([x[0] ** 2 + x[1] ** 2]), name="func"
        )
    problem.differentiation_method = problem.ApproximationMode.FINITE_DIFFERENCES

    execute_algo(problem, algo_name="NLOPT_SLSQP", max_iter=5)
    execute_post(problem, post_name="OptHistoryView", save=False, show=False)


def test_variable_names(snapshot_matplotlib) -> None:
    execute_post(
        Path(__file__).parent / "mdf_backup.h5",
        post_name="OptHistoryView",
        variable_names=["x_2", "x_1"],
        save=False,
        show=False,
    )


def test_no_gradient_history(caplog) -> None:
    """Check that OptHistoryView works without gradient history."""
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=-1.0, upper_bound=1.0, value=0.5)

    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(lambda x: x**2, name="f")
    problem.database.store(array([-1]), {"f": array([1])})
    problem.database.store(array([0]), {"f": array([0])})
    problem.database.store(array([1]), {"f": array([1])})
    post_processor = execute_post(
        problem, post_name="OptHistoryView", save=False, show=False
    )
    assert set(post_processor.figures.keys()) == {"variables", "objective", "x_xstar"}


def test_no_optimum() -> None:
    """Check that OptHistoryView does not mark a best design that does not exist."""
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=-1.0, upper_bound=1.0, value=0.5)

    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(lambda x: x**2, name="f")
    problem.add_constraint(ArrayFunction(lambda x: x, name="g"), constraint_type="ineq")
    # The only feasible designs are the last two, whose objective is NaN,
    # so the problem has no optimum solution.
    for x, f, g in [
        (-1.0, 1.0, 1.0),
        (-0.5, 0.25, 0.5),
        (0.5, nan, -0.5),
        (1.0, nan, -1.0),
    ]:
        problem.database.store(array([x]), {"f": array([f]), "g": array([g])})

    post_processor = execute_post(
        problem, post_name="OptHistoryView", save=False, show=False
    )
    assert post_processor._optimization_metadata.optimum_iteration is None
    # The figure of the distance to the best design is not drawn.
    assert set(post_processor.figures) == {"variables", "objective", "ineq_constraints"}
    # Neither the mark of the best iteration nor its mention is drawn.
    assert not post_processor.figures["variables"].axes[0].lines
    assert not post_processor.figures["ineq_constraints"].axes[0].lines
    assert post_processor.figures["variables"].axes[0].get_xlabel() == "Iterations"
    assert (
        post_processor.figures["objective"].axes[0].get_xlabel()
        == r"Iterations (NaN: $\times$)"
    )
    assert (
        post_processor.figures["ineq_constraints"].axes[0].get_xlabel() == "Iterations"
    )


def test_get_history_design_variable_is_function():
    """Test the _get_history method when the objective is also a design variable.

    This kind of situation can happen with the IDF formulation when the objective is
    also a coupling variable.
    """
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=-1.0, upper_bound=1.5, value=0.5)

    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(lambda x: x**2, name="x")
    problem.database.store(array([-1.0]), {"x": array([1.21])})
    problem.database.store(array([1.21]), {"x": array([1.4641])})

    view = OptHistoryView(problem)
    f_history, _, _, _ = view._get_history(
        view._optimization_metadata.standardized_objective_name, variable_names="x"
    )
    # Check that the shape matches and that the values are those of the function and
    # not those of the design vector.
    assert_equal(f_history, array([1.21, 1.4641]))


def _create_problem(
    biobjective: bool = False,
    output_names: Sequence[str] = ("f1", "f2"),
    minimize: bool = True,
    nan_x: float | None = None,
) -> OptimizationProblem:
    """Create and solve a problem with a DOE.

    Args:
        biobjective: Whether the objective has two outputs.
        output_names: The names of the outputs of the biobjective.
        minimize: Whether to minimize the objective.
        nan_x: The value of `x` for which the objective is `NaN`, if any.

    Returns:
        The solved problem.
    """
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=-1.0, upper_bound=2.0, value=0.5)
    problem = OptimizationProblem(design_space)

    def compute(x):
        if nan_x is not None and x[0] == nan_x:
            return array([nan, nan]) if biobjective else array([nan])
        if biobjective:
            return array([x[0] ** 2, (x[0] - 1) ** 2])
        return x**2

    if biobjective:
        problem.objective = ArrayFunction(
            compute, "f1_f2", output_names=list(output_names), dim=2
        )
    else:
        problem.objective = ArrayFunction(compute, "f")

    problem.add_constraint(
        ArrayFunction(lambda x: x - 0.8, "g"), constraint_type="ineq"
    )

    problem.minimize_objective = minimize
    execute_algo(problem, "doe", algo_name="PYDOE_FULLFACT", n_samples=5)
    return problem


def _get_curve_labels(ax: Axes) -> list[str]:
    """Return the labels of the legend of axes.

    Args:
        ax: The axes.

    Returns:
        The labels of the legend.
    """
    return [text.get_text() for text in ax.get_legend().get_texts()]


@pytest.mark.parametrize("minimize", [True, False])
@pytest.mark.parametrize("use_standardized_objective", [True, False])
def test_objective_labels(minimize, use_standardized_objective) -> None:
    """Check that the y-label of the objective plot is the name of the objective."""
    problem = _create_problem(minimize=minimize)
    figures = OptHistoryView(problem).execute(
        OptHistoryView_Settings(
            save=False, use_standardized_objective=use_standardized_objective
        )
    )
    prefix = "" if minimize or not use_standardized_objective else "-"
    ax = figures["objective"].axes[0]
    assert len(ax.lines[0].get_xdata()) == 5
    assert ax.get_ylabel() == f"{prefix}f"
    assert ax.get_legend() is None


@pytest.mark.parametrize("minimize", [True, False])
@pytest.mark.parametrize("use_standardized_objective", [True, False])
def test_multiobjective_labels(minimize, use_standardized_objective) -> None:
    """Check that a multiobjective is plotted with one curve per output."""
    problem = _create_problem(biobjective=True, minimize=minimize)
    figures = OptHistoryView(problem).execute(
        OptHistoryView_Settings(
            save=False, use_standardized_objective=use_standardized_objective
        )
    )
    prefix = "" if minimize or not use_standardized_objective else "-"
    ax = figures["objective"].axes[0]
    assert len(ax.lines[0].get_xdata()) == 5
    assert ax.get_ylabel() == f"{prefix}f1_f2"
    assert _get_curve_labels(ax) == [f"{prefix}f1", f"{prefix}f2"]
    assert not ax.lines[0].get_label().startswith("_")
    assert not ax.lines[1].get_label().startswith("_")


def test_multiobjective_snapshot(snapshot_matplotlib) -> None:
    """Check the figures of a multiobjective problem."""
    problem = _create_problem(biobjective=True)
    execute_post(problem, post_name="OptHistoryView", save=False, show=False)


@pytest.mark.parametrize("minimize", [True, False])
def test_multiobjective_no_best_iteration(minimize) -> None:
    """Check that the figures of a multiobjective problem mark no best iteration."""
    problem = _create_problem(biobjective=True, minimize=minimize)
    figures = OptHistoryView(problem).execute(OptHistoryView_Settings(save=False))
    assert "x_xstar" not in figures
    for figure in figures.values():
        assert "best iteration" not in figure.axes[0].get_xlabel()

    # The 2 curves only.
    assert len(figures["objective"].axes[0].lines) == 2


def test_multiobjective_without_output_names() -> None:
    """Check the legend of a multiobjective whose output names are unknown."""
    problem = _create_problem(biobjective=True, output_names=())
    figures = OptHistoryView(problem).execute(OptHistoryView_Settings(save=False))
    assert _get_curve_labels(figures["objective"].axes[0]) == [
        "f1_f2[0]",
        "f1_f2[1]",
    ]


def test_multiobjective_nan() -> None:
    """Check that a NaN in the objective is marked for a multiobjective."""
    problem = _create_problem(biobjective=True, nan_x=0.5)
    figures = OptHistoryView(problem).execute(OptHistoryView_Settings(save=False))
    ax = figures["objective"].axes[0]
    assert "NaN" in ax.get_xlabel()
    assert len(ax.lines[0].get_xdata()) == 4
    # 2 curves and 2 lines for the NaN mark.
    assert len(ax.lines) == 4


@pytest.mark.parametrize("obj_relative", [False, True])
def test_relative_objective_label(obj_relative) -> None:
    """Check that the y-label of the objective plot tells when it is relative."""
    problem = _create_problem()
    figures = OptHistoryView(problem).execute(
        OptHistoryView_Settings(save=False, obj_relative=obj_relative)
    )
    expected = "f (relative to iteration 1)" if obj_relative else "f"
    assert figures["objective"].axes[0].get_ylabel() == expected


def test_objective_all_nan() -> None:
    """Check that an objective whose values are all NaN can be plotted."""
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=-1.0, upper_bound=2.0, value=0.5)
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(lambda x: array([nan]), "f")
    execute_algo(problem, "doe", algo_name="PYDOE_FULLFACT", n_samples=5)
    figures = OptHistoryView(problem).execute(OptHistoryView_Settings(save=False))
    ax = figures["objective"].axes[0]
    assert ax.get_ylim() == pytest.approx((-1.1, 1.1))
    assert "NaN" in ax.get_xlabel()


@pytest.mark.parametrize("biobjective", [False, True])
def test_relative_objective_nan_at_first_iteration(biobjective, caplog) -> None:
    """Check that the relative objective is relative to the first non-NaN iteration."""
    problem = _create_problem(biobjective=biobjective, nan_x=-1.0)
    figures = OptHistoryView(problem).execute(
        OptHistoryView_Settings(save=False, obj_relative=True)
    )
    ax = figures["objective"].axes[0]
    name = "f1_f2" if biobjective else "f"
    assert ax.get_ylabel() == f"{name} (relative to iteration 2)"
    for line in ax.lines[: 2 if biobjective else 1]:
        assert_equal(line.get_xdata(), [1, 2, 3, 4])
        assert line.get_ydata()[0] == 0.0
    # Only the first iteration is marked as a NaN one.
    assert ax.lines[-1].get_xdata()[0] == 0
    reference = "[0.0625 1.5625]" if biobjective else "0.0625"
    assert (
        "with relative variation compared to "
        f"the objective value at iteration 2 = {reference}"
    ) in caplog.text


def test_relative_objective_all_nan(caplog) -> None:
    """Check that an objective always NaN is plotted without relative variation."""
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=-1.0, upper_bound=2.0, value=0.5)
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(lambda x: array([nan]), "f")
    execute_algo(problem, "doe", algo_name="PYDOE_FULLFACT", n_samples=5)
    figures = OptHistoryView(problem).execute(
        OptHistoryView_Settings(save=False, obj_relative=True)
    )
    assert figures["objective"].axes[0].get_ylabel() == "f"
    assert "Every iteration has a NaN objective value" in caplog.text
