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
from __future__ import annotations

from pathlib import Path
from threading import Thread
from threading import current_thread
from threading import get_native_id
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from numpy import array
from pydantic import ValidationError

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import sample_disciplines
from gemseo.core.discipline import Discipline
from gemseo.core.grammar.factory import GrammarFactory
from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from gemseo.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.doe.scipy.settings.lhs import LHS_Settings
from gemseo.formulation.disciplinary_opt_settings import DisciplinaryOpt_Settings
from gemseo.formulation.idf_settings import IDF_Settings
from gemseo.formulation.mdf_settings import MDF_Settings
from gemseo.mda.chain_settings import MDAChain_Settings
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
from gemseo.mda.jacobi_settings import MDAJacobi_Settings
from gemseo.optimization.nlopt.settings.nlopt_cobyla_settings import (
    NLOPT_COBYLA_Settings,
)
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.util._directory_manager.manager import DirectoryManager
from gemseo.util._directory_manager.settings import CleanUpPolicy
from gemseo.util._directory_manager.settings import MDACleanUpPolicy
from gemseo.util._directory_manager.settings import Settings
from gemseo.util._filename_sanitizer import secure_filename
from gemseo.util._workflow_observer.scenario import ScenarioWorkflowObserver
from gemseo.util.discipline import DummyDiscipline
from gemseo.util.global_configuration import _configuration
from gemseo.util.platform import platform_is_windows
from gemseo.util.testing.helper import assert_exception

from ...formulation.bilevel_test_helper import create_sobieski_bilevel_bcd_scenario
from ...formulation.bilevel_test_helper import create_sobieski_bilevel_scenario
from .directory_manager_test_helper import _keep_all_references
from .directory_manager_test_helper import _run_fresh
from .directory_manager_test_helper import assert_policies_distinguishable
from .directory_manager_test_helper import build_monolevel_scenario
from .directory_manager_test_helper import build_reference_key
from .directory_manager_test_helper import check_cleanup_policy
from .directory_manager_test_helper import create_disc_from_exe
from .directory_manager_test_helper import derive_policy_tree
from .directory_manager_test_helper import get_directory_tree

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from gemseo.scenario.evaluation import EvaluationScenario
    from gemseo.util.typing import StrKeyMapping


@pytest.fixture
def dm_settings(tmp_wd: Path):
    """Enable and reset the directory manager."""
    # The manager cannot be disabled once enabled, so restore the previous
    # (disabled) settings instance on teardown instead of toggling enable off.
    previous_settings = _configuration.directory_manager
    dm_settings = _configuration.directory_manager = Settings()
    dm_settings.enable = True
    dm_settings.execution_root_path = tmp_wd / "root"
    yield dm_settings
    # TODO: move this to a module wise teardown fixture.
    _configuration.directory_manager = previous_settings


@pytest.fixture
def generate_sobieski_bilevel_scenario() -> Callable[..., EvaluationScenario]:
    """Generate a BiLevel scenario for the Sobieski's SSBJ problem."""
    return create_sobieski_bilevel_scenario()


@pytest.fixture
def generate_sobieski_bilevel_bcd_scenario() -> Callable[..., EvaluationScenario]:
    """Generate a BiLevelBCD scenario for the Sobieski's SSBJ problem."""
    return create_sobieski_bilevel_bcd_scenario()


parametrized_clean_up_policy = pytest.mark.parametrize(
    "clean_up_policy",
    [
        CleanUpPolicy.KEEP_ALL,
        CleanUpPolicy.KEEP_LAST_ONLY,
        CleanUpPolicy.KEEP_SOLUTION_ONLY,
        CleanUpPolicy.KEEP_BASELINE_AND_SOLUTION,
    ],
)

parametrized_policy = pytest.mark.parametrize(
    "policy", list(CleanUpPolicy), ids=lambda policy: policy.name
)

parametrized_mda_policy = pytest.mark.parametrize(
    "mda_policy",
    [MDACleanUpPolicy.KEEP_ALL, MDACleanUpPolicy.KEEP_LAST_ONLY],
    ids=lambda mda_policy: mda_policy.name,
)

# A small hand-written KEEP_ALL tree used to check derive_policy_tree against
# the cleanup policy definitions, independently of the manager:
#   - 3 top-level scenario iterations, with the optimum at iteration 2 (the
#     middle one), neither the first nor the last;
#   - an MDA nested under iteration 1, with 2 solver iterations;
#   - a nested sub-scenario under iteration 3, with 2 of its own iterations;
#   - a managed, non-iteration child directly under the top-level scenario
#     (e.g. a discipline executed once outside the solver loop), kept only
#     by KEEP_ALL: every other policy keeps only the last / solution /
#     baseline-and-solution iteration directories.
_oracle_keep_all_tree = sorted([
    "MDOScenario",
    "MDOScenario/Foo_execution",
    "MDOScenario/Optimizer_iteration_1",
    "MDOScenario/Optimizer_iteration_1/MDAJacobi",
    "MDOScenario/Optimizer_iteration_1/MDAJacobi/MDAJacobi_iteration_0",
    "MDOScenario/Optimizer_iteration_1/MDAJacobi/MDAJacobi_iteration_1",
    "MDOScenario/Optimizer_iteration_2",
    "MDOScenario/Optimizer_iteration_3",
    "MDOScenario/Optimizer_iteration_3/SubScenario",
    "MDOScenario/Optimizer_iteration_3/SubScenario/Optimizer_iteration_1",
    "MDOScenario/Optimizer_iteration_3/SubScenario/Optimizer_iteration_2",
])
_oracle_optimum_iteration = 2


@pytest.mark.parametrize(
    ("policy", "mda_policy", "expected_tree"),
    [
        (
            CleanUpPolicy.KEEP_ALL,
            MDACleanUpPolicy.KEEP_ALL,
            _oracle_keep_all_tree,
        ),
        (
            CleanUpPolicy.KEEP_ALL,
            MDACleanUpPolicy.KEEP_LAST_ONLY,
            [
                "MDOScenario",
                "MDOScenario/Foo_execution",
                "MDOScenario/Optimizer_iteration_1",
                "MDOScenario/Optimizer_iteration_1/MDAJacobi",
                "MDOScenario/Optimizer_iteration_1/MDAJacobi/MDAJacobi_iteration_1",
                "MDOScenario/Optimizer_iteration_2",
                "MDOScenario/Optimizer_iteration_3",
                "MDOScenario/Optimizer_iteration_3/SubScenario",
                "MDOScenario/Optimizer_iteration_3/SubScenario/Optimizer_iteration_1",
                "MDOScenario/Optimizer_iteration_3/SubScenario/Optimizer_iteration_2",
            ],
        ),
        (
            CleanUpPolicy.KEEP_LAST_ONLY,
            MDACleanUpPolicy.KEEP_ALL,
            [
                "MDOScenario",
                "MDOScenario/Optimizer_iteration_3",
                "MDOScenario/Optimizer_iteration_3/SubScenario",
                "MDOScenario/Optimizer_iteration_3/SubScenario/Optimizer_iteration_2",
            ],
        ),
        (
            CleanUpPolicy.KEEP_SOLUTION_ONLY,
            MDACleanUpPolicy.KEEP_ALL,
            [
                "MDOScenario",
                "MDOScenario/Optimizer_iteration_2",
            ],
        ),
        (
            CleanUpPolicy.KEEP_BASELINE_AND_SOLUTION,
            MDACleanUpPolicy.KEEP_ALL,
            [
                "MDOScenario",
                "MDOScenario/Optimizer_iteration_1",
                "MDOScenario/Optimizer_iteration_1/MDAJacobi",
                "MDOScenario/Optimizer_iteration_1/MDAJacobi/MDAJacobi_iteration_0",
                "MDOScenario/Optimizer_iteration_1/MDAJacobi/MDAJacobi_iteration_1",
                "MDOScenario/Optimizer_iteration_2",
            ],
        ),
    ],
)
def test_derive_policy_tree_oracle(policy, mda_policy, expected_tree):
    """Verify derive_policy_tree against a small, hand-written KEEP_ALL tree.

    The KEEP_LAST_ONLY and the solution-based cases are exact for the
    top-level scenario (the optimum is at iteration 2 of 3); for the nested
    sub-scenario (under iteration 3), only KEEP_LAST_ONLY is checked here, as
    the solution-based policies for a nested scenario need an `actual_tree`
    reflecting a real run, which the per-policy snapshots of the scenario
    tests below provide (see `check_cleanup_policy`).
    """
    expected_tree = sorted(expected_tree)
    assert (
        derive_policy_tree(
            _oracle_keep_all_tree,
            policy,
            _oracle_optimum_iteration,
            expected_tree,
            mda_policy=mda_policy,
        )
        == expected_tree
    )


def test_derive_policy_tree_oracle_with_homonym_mda_suffix():
    """Verify that an MDA directory is recognized despite a homonym `#<n>` suffix.

    The manager appends a `#<n>` suffix to a directory name (e.g.
    `MDAJacobi#0`) when the same MDA is instantiated more than once under the
    same parent; the suffix must be stripped both when classifying the
    directory as an MDA directory and when matching its `_iteration_<n>`
    children, or `MDACleanUpPolicy.KEEP_LAST_ONLY` would wrongly keep every
    iteration instead of pruning all but the last.
    """
    keep_all_tree = sorted([
        "MDOScenario",
        "MDOScenario/MDAJacobi#0",
        "MDOScenario/MDAJacobi#0/MDAJacobi_iteration_0",
        "MDOScenario/MDAJacobi#0/MDAJacobi_iteration_1",
    ])
    expected_tree = sorted([
        "MDOScenario",
        "MDOScenario/MDAJacobi#0",
        "MDOScenario/MDAJacobi#0/MDAJacobi_iteration_1",
    ])
    assert (
        derive_policy_tree(
            keep_all_tree,
            CleanUpPolicy.KEEP_ALL,
            1,
            expected_tree,
            mda_policy=MDACleanUpPolicy.KEEP_LAST_ONLY,
        )
        == expected_tree
    )


def test_get_directory_tree_structural_only_strips_homonym_suffix(tmp_wd):
    """Verify that structural_only drops a leaf directory with a homonym suffix.

    The manager appends a `#<n>` suffix to a directory name (e.g.
    `Foo_execution#0`) when the same discipline is executed more than once
    under the same parent; the suffix must be stripped before checking the
    `_execution` / `_linearization` leaf-discipline suffix, or such a
    directory would wrongly survive the `structural_only` filter.
    """
    (tmp_wd / "Foo_execution#0").mkdir()

    assert get_directory_tree(tmp_wd, structural_only=True) == []


@pytest.mark.parametrize(
    (
        "formulation_settings_model",
        "settings_model",
        "require_feasible",
        "start_at_lower_bounds",
    ),
    [
        pytest.param(
            MDF_Settings(
                main_mda_settings=MDAChain_Settings(
                    inner_mda_settings=MDAGaussSeidel_Settings(
                        max_mda_iter=3, tolerance=0.0
                    )
                ),
            ),
            # max_iter=8 (rather than a smaller value): with fewer sweep
            # points, this configuration finds at most one feasible point,
            # which leaves KEEP_SOLUTION_ONLY / KEEP_BASELINE_AND_SOLUTION
            # untested against a genuine runner-up (assert_policies_
            # distinguishable would fail).
            SLSQP_Settings(max_iter=8),
            True,
            False,
            id="mdf-gauss_seidel-slsqp",
        ),
        pytest.param(
            IDF_Settings(),
            # max_iter=6 (rather than 5): with only 5, the sweep point
            # closest to the optimum of the constraint-violation function is
            # always the last one evaluated, which fails the interior-
            # iteration check.
            SLSQP_Settings(max_iter=6),
            False,
            # x0 moved to the lower bounds: the default point is close to the
            # SSBJ equilibrium, less infeasible than any sweep point, so it
            # would always be the (non-interior) optimum at iteration 1.
            True,
            id="idf-slsqp",
        ),
        pytest.param(
            MDF_Settings(
                main_mda_settings=MDAChain_Settings(
                    inner_mda_settings=MDAJacobi_Settings(max_mda_iter=3, tolerance=0.0)
                )
            ),
            NLOPT_COBYLA_Settings(max_iter=8),
            True,
            False,
            id="mdf-jacobi-cobyla",
        ),
        pytest.param(
            IDF_Settings(),
            # Same structural issue as the SLSQP IDF configuration above
            # (COBYLA uses the same deterministic sweep).
            NLOPT_COBYLA_Settings(max_iter=6),
            False,
            True,
            id="idf-cobyla",
        ),
        pytest.param(
            MDF_Settings(
                main_mda_settings=MDAChain_Settings(
                    inner_mda_settings=MDAJacobi_Settings(max_mda_iter=3, tolerance=0.0)
                )
            ),
            # A much larger n_samples than the other configurations: the
            # feasible region defined by g_1, g_2 and g_3 is small, and a
            # handful of LHS samples almost never lands two points inside it.
            LHS_Settings(n_samples=30),
            True,
            False,
            id="mdf-jacobi-lhs",
        ),
        pytest.param(
            IDF_Settings(),
            # n_samples=12: no sample raises a math domain error in the SSBJ
            # disciplines (with 15, samples 10 and 14 do, and the DOE skips
            # them, which shifts the `DOE_sample_<n>` suffixes away from the
            # database iterations), and the optimum is interior (iteration 9).
            LHS_Settings(n_samples=12),
            # False (not None): unlike the two MDO configurations above, the
            # LHS samples are not on the deterministic sweep, so an interior,
            # well-separated optimum is reachable; only the feasibility
            # itself is unreachable (the consistency constraints are
            # essentially never satisfied by random sampling).
            False,
            False,
            id="idf-lhs",
        ),
    ],
)
@parametrized_policy
def test_monolevel_scenarios_all_policies(
    dm_settings,
    tmp_wd,
    snapshot,
    request,
    formulation_settings_model,
    settings_model,
    require_feasible,
    start_at_lower_bounds,
    policy,
):
    """Test one cleanup policy for a mono-level scenario.

    The `KEEP_ALL` tree of each configuration (MDF or IDF formulation, with a
    gradient-based or a derivative-free optimizer, or a DOE) is checked
    against a syrupy snapshot; the tree of every other policy is checked
    against its own syrupy snapshot and against the oracle
    (`derive_policy_tree`), fed with the `KEEP_ALL` tree and the optimum
    iteration of this configuration's reference run (see
    `check_cleanup_policy`).

    For IDF, the coupling variables are free design variables that the
    deterministic sweep does not solve for, so the optimum found by MDO is
    never feasible: `assert_policies_distinguishable` is called with
    `require_feasible=False` for every IDF case (the interior-iteration and
    the gap checks still run, over every database point rather than only the
    feasible ones). `start_at_lower_bounds` moves the design space's current
    value to its lower bounds before each fresh scenario is executed, for the
    IDF MDO configurations (see the parametrization comments).
    """

    def build_scenario():
        scenario = build_monolevel_scenario(
            formulation_settings_model.model_copy(deep=True)
        )
        if start_at_lower_bounds:
            design_space = scenario.design_space
            design_space.set_current_value(design_space.get_lower_bounds())
        return scenario

    check_cleanup_policy(
        build_scenario,
        settings_model,
        tmp_wd,
        snapshot,
        policy,
        build_reference_key(request),
        require_feasible=require_feasible,
    )


@parametrized_mda_policy
def test_mda_clean_up_policies_for_mono_level_scenarios(
    dm_settings, tmp_wd, snapshot, request, mda_policy
):
    """Test the KEEP_ALL and KEEP_LAST_ONLY policies for an inner MDA.

    The scenario cleanup policy is fixed to KEEP_BASELINE_AND_SOLUTION, so
    the top-level scenario directories are already pruned in both runs; only
    the MDA cleanup policy varies. The `MDACleanUpPolicy.KEEP_ALL` reference
    tree and optimum iteration of this configuration are fetched from
    `_keep_all_references`, building and caching them first if this is the
    first MDA policy of the configuration to run.

    For `mda_policy == MDACleanUpPolicy.KEEP_ALL`, the reference tree itself
    is compared to the syrupy snapshot and checked for policy distinguishability.
    For `MDACleanUpPolicy.KEEP_LAST_ONLY`, a fresh scenario is built and
    executed under its own execution root, checked for determinism against the
    reference's optimum iteration, and its tree is compared to the oracle's
    prediction.
    """

    def build_scenario():
        return build_monolevel_scenario(
            MDF_Settings(
                main_mda_settings=MDAChain_Settings(
                    inner_mda_settings=MDAJacobi_Settings(max_mda_iter=3, tolerance=0.0)
                )
            ),
        )

    algo_settings_model = SLSQP_Settings(max_iter=8)
    reference_key = build_reference_key(request)

    reference = _keep_all_references.get(reference_key)
    if reference is None:
        scenario, keep_all_tree, optimum_iteration = _run_fresh(
            build_scenario,
            algo_settings_model,
            tmp_wd / "ref",
            CleanUpPolicy.KEEP_BASELINE_AND_SOLUTION,
            MDACleanUpPolicy.KEEP_ALL,
        )
        assert_policies_distinguishable(scenario)
        reference = keep_all_tree, optimum_iteration
        _keep_all_references[reference_key] = reference

    keep_all_tree, optimum_iteration = reference

    if mda_policy == MDACleanUpPolicy.KEEP_ALL:
        assert keep_all_tree == snapshot
        return

    _, actual_tree, fresh_optimum_iteration = _run_fresh(
        build_scenario,
        algo_settings_model,
        tmp_wd / "run",
        CleanUpPolicy.KEEP_BASELINE_AND_SOLUTION,
        mda_policy,
    )
    assert fresh_optimum_iteration == optimum_iteration, (
        "The scenario execution is not deterministic: the optimum "
        f"iteration was {optimum_iteration} under MDACleanUpPolicy.KEEP_ALL and "
        f"{fresh_optimum_iteration} under {mda_policy}."
    )
    assert actual_tree == derive_policy_tree(
        keep_all_tree,
        CleanUpPolicy.KEEP_ALL,
        optimum_iteration,
        actual_tree,
        mda_policy=mda_policy,
    )


@pytest.mark.skip_under_windows
@parametrized_policy
def test_directory_manager_with_multiprocessing(
    dm_settings, tmp_wd, snapshot, request, generate_sobieski_bilevel_scenario, policy
):
    """Test one cleanup policy for a bilevel scenario run with multiprocessing."""
    settings_model = LHS_Settings(n_samples=4, n_processes=4)

    def build_scenario():
        return generate_sobieski_bilevel_scenario(
            main_mda_settings=MDAJacobi_Settings(max_mda_iter=2, tolerance=0.0),
        )

    check_cleanup_policy(
        build_scenario,
        settings_model,
        tmp_wd,
        snapshot,
        policy,
        build_reference_key(request),
    )


def test_directory_manager_with_spawn_multiprocessing(dm_settings, monkeypatch):
    """Verify the directories created by workers of the spawn start method."""
    monkeypatch.setattr(
        CallableParallelExecution, "multi_processing_start_method", "spawn"
    )
    discipline = create_discipline("AnalyticDiscipline", expressions={"y": "2*x"})
    design_space = create_design_space()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    scenario = create_scenario(
        discipline,
        "y",
        design_space,
        formulation_settings_model=DisciplinaryOpt_Settings(),
    )
    scenario.execute(LHS_Settings(n_samples=3, n_processes=2))

    scenario_path = dm_settings.execution_root_path / "MDOScenario"
    for sample in (1, 2, 3):
        assert (
            scenario_path / f"DOE_sample_{sample}" / "AnalyticDiscipline_execution"
        ).is_dir()


@pytest.mark.parametrize(
    ("scenario_type", "settings_model"),
    [
        pytest.param("MDO", NLOPT_COBYLA_Settings(max_iter=5), id="mdo-cobyla"),
        pytest.param(
            "DOE",
            # n_samples=15 (rather than a smaller value): with fewer LHS samples
            # the optimum lands on the first database iteration, which fails
            # assert_policies_distinguishable.
            LHS_Settings(n_samples=15),
            id="doe-lhs",
        ),
    ],
)
@parametrized_policy
@pytest.mark.xfail(
    platform_is_windows,
    reason="Windows can't handle directory paths that are too long.",
)
def test_all_policies_sobieski_bilevel(
    dm_settings,
    tmp_wd,
    snapshot,
    request,
    generate_sobieski_bilevel_scenario,
    scenario_type,
    settings_model,
    policy,
):
    """Test one cleanup policy for the bilevel formulation.

    The tree is compared structurally only (`structural_only=True`): the
    leaf discipline execution/linearization directories are dropped, as their
    number and nesting are a consequence of the differentiation method, not
    of the cleanup policy under test.
    """

    def build_scenario():
        return generate_sobieski_bilevel_scenario(
            main_mda_settings=MDAChain_Settings(
                max_mda_iter=2, inner_mda_settings=MDAJacobi_Settings(tolerance=0.0)
            ),
        )

    check_cleanup_policy(
        build_scenario,
        settings_model,
        tmp_wd,
        snapshot,
        policy,
        build_reference_key(request),
        structural_only=True,
    )


@pytest.mark.parametrize(
    ("scenario_type", "settings_model"),
    [
        pytest.param("MDO", NLOPT_COBYLA_Settings(max_iter=3), id="mdo-cobyla"),
        pytest.param("DOE", LHS_Settings(n_samples=3), id="doe-lhs"),
    ],
)
@parametrized_policy
@pytest.mark.xfail(
    platform_is_windows,
    reason="Windows can't handle directory paths that are too long.",
)
def test_all_policies_bilevel_bcd_sobieski(
    dm_settings,
    tmp_wd,
    snapshot,
    request,
    generate_sobieski_bilevel_bcd_scenario,
    scenario_type,
    settings_model,
    policy,
):
    """Test one cleanup policy for the bilevel BCD formulation.

    `short_names=True` is used on every platform, not only on Windows: the
    BCD formulation nests two levels of sub-scenarios inside the top-level
    one, and the resulting paths are long enough to be worth shortening
    everywhere, so that this test exercises the same directory names as it
    would on Windows. The tree is compared structurally only, like the plain
    bilevel formulation.
    """

    def build_scenario():
        scenario = generate_sobieski_bilevel_bcd_scenario(short_names=True)
        scenario.formulation._mda1.inner_mdas[0].settings = MDAJacobi_Settings(
            max_mda_iter=2, tolerance=0.0
        )
        scenario.formulation._mda2.inner_mdas[0].settings = MDAJacobi_Settings(
            max_mda_iter=2, tolerance=0.0
        )
        scenario.formulation._bcd_mda.settings = MDAGaussSeidel_Settings(
            max_mda_iter=2, tolerance=0.0
        )
        for scenario_adapter in scenario.formulation._scenario_adapters:
            scenario_adapter.scenario.set_algorithm(SLSQP_Settings(max_iter=3))
            scenario_adapter.scenario.formulation.mda.settings = (
                MDAGaussSeidel_Settings(max_mda_iter=2, tolerance=0.0)
            )
        scenario.formulation._mda1.inner_mdas[0].name = "MDA1"
        scenario.formulation._mda2.inner_mdas[0].name = "MDA2"
        return scenario

    check_cleanup_policy(
        build_scenario,
        settings_model,
        tmp_wd,
        snapshot,
        policy,
        build_reference_key(request),
        structural_only=True,
    )


class DisciplineWithFiles(Discipline):
    """A discipline that generates files at each execution."""

    def __init__(self):
        super().__init__()
        self.input_grammar.update_from_names(["x"])
        self.output_grammar.update_from_names(["y"])
        self.default_input_data = {"x": array([1.0])}

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        y = input_data["x"] + 1.0
        Path("out.txt").write_text(str(y))
        return {"y": y}


def test_discipline_files(dm_settings):
    """Test that disciplines that generate files store them in the right directory."""
    root_path = dm_settings.execution_root_path
    discipline = DisciplineWithFiles()
    discipline.execute()

    assert Path(root_path / "DisciplineWithFiles_execution" / "out.txt").exists()

    discipline.execute({"x": array([3.0])})

    assert Path(root_path / "DisciplineWithFiles_execution#0" / "out.txt").exists()
    assert Path(root_path / "DisciplineWithFiles_execution#1" / "out.txt").exists()


def test_discipline_files_with_untracked_subdirectory(dm_settings):
    """Verify renaming a homonym directory containing an untracked subdirectory."""
    root_path = dm_settings.execution_root_path
    discipline = DisciplineWithFiles()
    discipline.execute()
    # A subdirectory created behind the manager's back, e.g. by an executable.
    (root_path / "DisciplineWithFiles_execution" / "untracked").mkdir()

    discipline.execute({"x": array([3.0])})

    assert (root_path / "DisciplineWithFiles_execution#0" / "untracked").exists()
    assert (root_path / "DisciplineWithFiles_execution#1").exists()


def test_scenario_discipline_with_files(dm_settings):
    """Test the execution of a scenario with a discipline that writes files."""
    discipline = DisciplineWithFiles()
    design_space = create_design_space()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    scenario = create_scenario(
        discipline,
        "y",
        design_space,
        formulation_settings_model=DisciplinaryOpt_Settings(),
    )
    scenario.execute(LHS_Settings(n_samples=3))
    for iteration in range(1, 4):
        assert Path(
            dm_settings.execution_root_path
            / f"MDOScenario/DOE_sample_{iteration}/DisciplineWithFiles_execution"
            / "out.txt"
        ).exists()


def build_disciplinary_doe_scenario() -> EvaluationScenario:
    """Build a single-discipline scenario for DOE executions.

    Returns:
        The scenario.
    """
    design_space = create_design_space()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    return create_scenario(
        DisciplineWithFiles(),
        "y",
        design_space,
        formulation_settings_model=DisciplinaryOpt_Settings(),
    )


@pytest.mark.parametrize(
    "n_processes",
    [
        1,
        2,
    ],
)
def test_doe_with_duplicated_samples(dm_settings, n_processes):
    """Verify per-sample directories with duplicated DOE samples."""
    scenario = build_disciplinary_doe_scenario()
    scenario.execute(
        CustomDOE_Settings(
            samples=array([[0.2], [0.2], [0.5]]), n_processes=n_processes
        )
    )

    # The directory of the duplicated sample may be pruned (the evaluation is
    # a database hit, so no discipline executes inside): only check that the
    # samples are numbered by their index, without homonym '#' suffixes.
    scenario_path = dm_settings.execution_root_path / "MDOScenario"
    sample_dir_names = {path.name for path in scenario_path.iterdir() if path.is_dir()}
    assert {"DOE_sample_1", "DOE_sample_3"} <= sample_dir_names
    assert sample_dir_names <= {"DOE_sample_1", "DOE_sample_2", "DOE_sample_3"}


def test_worker_thread_nested_directories(dm_settings):
    """Verify that directories created in a worker thread are correctly nested."""
    root_path = dm_settings.execution_root_path
    discipline = DisciplineWithFiles()
    design_space = create_design_space()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    scenario = create_scenario(
        discipline,
        "y",
        design_space,
        formulation_settings_model=DisciplinaryOpt_Settings(),
    )

    def run() -> None:
        # Tag the thread like CallableParallelExecution tags its worker threads.
        thread = current_thread()
        thread.parent_id = get_native_id()
        thread.parent_path = root_path
        scenario.execute(LHS_Settings(n_samples=2))

    thread = Thread(target=run)
    thread.start()
    thread.join()

    for sample in (1, 2):
        assert (
            root_path
            / "MDOScenario"
            / f"DOE_sample_{sample}"
            / "DisciplineWithFiles_execution"
        ).is_dir()
        assert not (root_path / f"DOE_sample_{sample}").exists()


@parametrized_clean_up_policy
def test_clean_up_preserves_unmanaged_directories(dm_settings, clean_up_policy):
    """Verify that cleanup policies never remove directories created by users."""
    dm_settings.clean_up_policy = clean_up_policy
    discipline = DisciplineWithFiles()
    design_space = create_design_space()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    scenario = create_scenario(
        discipline,
        "y",
        design_space,
        formulation_settings_model=DisciplinaryOpt_Settings(),
    )
    user_dir = dm_settings.execution_root_path / "MDOScenario" / "user_data"

    def create_user_directory(index, data) -> None:
        user_dir.mkdir(exist_ok=True)

    scenario.execute(LHS_Settings(n_samples=2, callbacks=[create_user_directory]))

    assert user_dir.exists()


@pytest.mark.xfail(
    platform_is_windows,
    reason="Windows can't handle directory paths that are too long.",
)
def test_backup_h5(dm_settings, generate_sobieski_bilevel_scenario):
    """Test the backup h5 file write for each iteration."""
    dm_settings.save_history_backup = True
    dm_settings.backup_settings.plot = True
    dm_settings.backup_settings.at_each_iteration = True
    dm_settings.backup_settings.at_each_function_call = False

    scenario = generate_sobieski_bilevel_scenario(
        main_mda_settings=MDAChain_Settings(max_mda_iter=2),
    )
    scenario.execute(NLOPT_COBYLA_Settings(max_iter=5))

    root_path = dm_settings.execution_root_path

    assert Path(
        root_path / "MDOScenario/Optimizer_iteration_1/AerodynamicsScenario/backup.h5"
    ).exists()
    assert Path(root_path / "MDOScenario/backup.h5").exists()


def test_save_history_backup_with_evaluation_scenario(dm_settings):
    """Verify the history backup with a scenario lacking the plot option."""
    dm_settings.save_history_backup = True
    discipline = DisciplineWithFiles()
    design_space = create_design_space()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)

    sample_disciplines(
        [discipline],
        design_space,
        "y",
        algo_settings_model=LHS_Settings(n_samples=2),
    )

    assert (dm_settings.execution_root_path / "Sampling" / "backup.h5").exists()


def test_save_mda_residuals(dm_settings):
    """Test saving the mda residuals plot."""
    dm_settings.save_mda_residuals = True

    scenario = build_monolevel_scenario(MDF_Settings())
    scenario.execute(SLSQP_Settings(max_iter=2))

    assert Path(
        dm_settings.execution_root_path
        / "MDOScenario/Optimizer_iteration_1/MDAJacobi/MDAJacobi_residuals_history.pdf"
    ).exists()


def test_executable_discipline(dm_settings):
    """Test the Executable disciplines with the DirectoryManager enabled."""
    root_path = dm_settings.execution_root_path
    file_path = Path(__file__).parent.parent.parent / "discipline" / "wrapper"
    disc = create_disc_from_exe(file_path)
    design_space = create_design_space()
    design_space.add_real_variable("a", lower_bound=0.0, upper_bound=10.0, value=1.0)
    design_space.add_real_variable("b", lower_bound=0.0, upper_bound=10.0, value=1.0)
    design_space.add_real_variable("c", lower_bound=0.0, upper_bound=10.0, value=1.0)
    scenario = create_scenario(
        disc,
        "out",
        design_space,
        formulation_settings_model=DisciplinaryOpt_Settings(),
    )
    scenario.execute(LHS_Settings(n_samples=3))

    for i in [1, 2, 3]:
        assert Path(
            root_path
            / "MDOScenario"
            / f"DOE_sample_{i}"
            / "DiscFromExe_execution"
            / "input.json"
        ).exists()
        assert Path(
            root_path
            / "MDOScenario"
            / f"DOE_sample_{i}"
            / "DiscFromExe_execution"
            / "output.json"
        ).exists()


def test_discipline_with_space(dm_settings):
    """Test that the directory of discipline with a space on its name gets replaced
    by '_'."""

    class NoOpDiscipline(Discipline):
        def _run(self, input_data):
            return {}

    disc = NoOpDiscipline(name="my discipline")
    disc.execute()
    assert not (dm_settings.execution_root_path / "my discipline_execution").exists()
    assert (dm_settings.execution_root_path / "my_discipline_execution").exists()


def test_dummy_discipline_is_not_observed(dm_settings):
    """Verify that DummyDiscipline (e.g. built in bulk by XLSStudyParser) is excluded.

    See the exclusion in DisciplineWorkflowObserver._spec.
    """
    DummyDiscipline(name="dummy").execute()

    assert not (dm_settings.execution_root_path / "dummy_execution").exists()


def test_inheriting_disciplines(dm_settings):
    """Verify the observation of a subclass of an observed concrete discipline."""
    parent = DisciplineWithFiles()
    parent.execute()

    class ChildDiscipline(DisciplineWithFiles):
        pass

    child = ChildDiscipline()
    child.execute()

    root_path = dm_settings.execution_root_path
    assert (root_path / "DisciplineWithFiles_execution").is_dir()
    assert (root_path / "ChildDiscipline_execution").is_dir()


def test_scenario_with_non_ascii_name(dm_settings, snapshot):
    """Verify an error is raised when the sanitized name is empty."""
    discipline = DisciplineWithFiles()
    design_space = create_design_space()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    scenario = create_scenario(
        discipline,
        "y",
        design_space,
        formulation_settings_model=DisciplinaryOpt_Settings(),
        name="优化场景",
    )
    with assert_exception(ValueError, snapshot):
        scenario.execute(LHS_Settings(n_samples=1))


def test_scenario_named_after_the_trace_registry_directory(dm_settings):
    """Verify that a root-level observee named `gemseo-traces` does not clash.

    The trace registry writes under the root path, in the very namespace in
    which the manager creates the execution directories, and it does so before
    any of them exists. Its directory name therefore starts with a dot, which
    no sanitized observee name can produce, so that the bare `mkdir` of
    `DirectoryManager.start_directory` cannot hit it.
    """
    discipline = DisciplineWithFiles()
    design_space = create_design_space()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    scenario = create_scenario(
        discipline,
        "y",
        design_space,
        formulation_settings_model=DisciplinaryOpt_Settings(),
        name="gemseo-traces",
    )

    scenario.execute(LHS_Settings(n_samples=1))

    root_path = dm_settings.execution_root_path
    assert (
        root_path / "gemseo-traces" / "DOE_sample_1" / "DisciplineWithFiles_execution"
    ).is_dir()
    # The registry is named after the class, not after the observee.
    assert (root_path / ".gemseo-traces" / "MDOScenario" / "0.trace.yml").exists()


def test_discipline_exception(dm_settings, snapshot):
    """Verify that a the observation end is done when a discipline fails."""

    msg = "Crash!"

    class CrashingDiscipline(Discipline):
        def _run(self, input_data):
            raise RuntimeError(msg)

    disc = CrashingDiscipline()

    with patch.object(
        disc._workflow_observer,
        "end",
        wraps=disc._workflow_observer.end,
    ) as observer_end_mock:
        with assert_exception(RuntimeError, snapshot):
            disc.execute()
        # Make sure that the observer end call is done after the exception.
        observer_end_mock.assert_called()

    assert (
        dm_settings.execution_root_path / f"{CrashingDiscipline.__name__}_execution"
    ).exists()


def test_cannot_disable_once_enabled(dm_settings, snapshot):
    """Verify that the manager cannot be disabled once it has been enabled."""
    DisciplineWithFiles().execute()

    with assert_exception(ValueError, snapshot):
        dm_settings.enable = False


def test_discipline_keyboard_interrupt(dm_settings):
    """Verify that a BaseException propagates unchanged through observation."""

    class InterruptedDiscipline(Discipline):
        def _run(self, input_data):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        InterruptedDiscipline().execute()


def test_failing_scenario_with_solution_policy(dm_settings, snapshot):
    """Verify that ending the directories does not mask the original error."""
    dm_settings.clean_up_policy = CleanUpPolicy.KEEP_SOLUTION_ONLY

    class CrashingDiscipline(Discipline):
        def __init__(self):
            super().__init__()
            self.input_grammar.update_from_names(["x"])
            self.output_grammar.update_from_names(["y"])
            self.default_input_data = {"x": array([1.0])}

        def _run(self, input_data):
            msg = "Crash!"
            raise RuntimeError(msg)

    design_space = create_design_space()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    scenario = create_scenario(
        CrashingDiscipline(),
        "y",
        design_space,
        formulation_settings_model=DisciplinaryOpt_Settings(),
    )

    with assert_exception(RuntimeError, snapshot):
        scenario.execute(LHS_Settings(n_samples=2))

    assert Path.cwd() == dm_settings.execution_root_path


def test_failing_scenario_with_keep_last_policy(dm_settings):
    """Verify the cleanup of a scenario that fails before any iteration.

    The DOE library rejects the samples before evaluating any of them, so the
    scenario directory contains no iteration directory when the cleanup runs.
    """
    dm_settings.clean_up_policy = CleanUpPolicy.KEEP_LAST_ONLY
    discipline = DisciplineWithFiles()
    design_space = create_design_space()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    scenario = create_scenario(
        discipline,
        "y",
        design_space,
        formulation_settings_model=DisciplinaryOpt_Settings(),
    )

    # The samples do not match the design space dimension.
    with pytest.raises(ValueError):
        scenario.execute(CustomDOE_Settings(samples=array([[1.0, 2.0]])))

    # The error propagates, the empty scenario directory is left untouched.
    assert (dm_settings.execution_root_path / "MDOScenario").is_dir()
    assert Path.cwd() == dm_settings.execution_root_path


def test_solution_policy_with_non_iteration_managed_directory(dm_settings):
    """Verify the solution scan over a managed dir without an iteration suffix.

    A discipline executed from a DOE callback runs with the scenario directory
    as the current working directory: its execution directory is managed but
    carries no iteration suffix, so the solution scan skips it and it is
    removed like any non-optimum directory.
    """
    dm_settings.clean_up_policy = CleanUpPolicy.KEEP_SOLUTION_ONLY
    discipline = DisciplineWithFiles()
    callback_discipline = DisciplineWithFiles()
    design_space = create_design_space()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    scenario = create_scenario(
        discipline,
        "y",
        design_space,
        formulation_settings_model=DisciplinaryOpt_Settings(),
    )

    def execute_discipline(index, data) -> None:
        callback_discipline.execute()

    scenario.execute(LHS_Settings(n_samples=1, callbacks=[execute_discipline]))

    scenario_path = dm_settings.execution_root_path / "MDOScenario"
    assert (scenario_path / "DOE_sample_1").is_dir()
    assert not (scenario_path / "DisciplineWithFiles_execution").exists()


def test_solution_policy_keeps_everything_when_optimum_is_not_in_the_database(
    dm_settings,
):
    """Verify that a database miss on the optimum design keeps every directory.

    `OptimizationHistory.optimum` returns an empty design vector when no
    feasible point carries the value of the objective, and
    `Database.get_iteration` raises `KeyError` for such a vector (a database
    that is non-empty, but has no feasible optimum): the solution-based
    policies must then keep everything rather than mask the exception being
    propagated (see `DirectoryManager.__get_removals_solution`).
    """
    dm_settings.clean_up_policy = CleanUpPolicy.KEEP_SOLUTION_ONLY

    class _FakeDatabase:
        """A non-empty database that never holds the optimum design."""

        def get_iteration(self, design: Any) -> int:
            """Raise as the real database does for a design it does not hold.

            Args:
                design: The design vector to look up.

            Returns:
                Never returns.
            """
            msg = "unknown design"
            raise KeyError(msg)

    problem = SimpleNamespace(database=_FakeDatabase(), optimum=(None, array([])))
    observer = ScenarioWorkflowObserver.__new__(ScenarioWorkflowObserver)
    observer.object_ = SimpleNamespace(formulation=SimpleNamespace(problem=problem))

    manager = DirectoryManager()
    scenario_path = manager.start_directory(observer, "FakeScenario")
    for index in (1, 2):
        iteration_observer = SimpleNamespace()
        manager.start_directory(iteration_observer, f"Optimizer_iteration_{index}")
        manager.end_directory(iteration_observer)

    manager.end_directory(observer)

    assert (scenario_path / "Optimizer_iteration_1").is_dir()
    assert (scenario_path / "Optimizer_iteration_2").is_dir()


def test_history_view_skipped_for_short_history(dm_settings):
    """Verify that no history view is plotted with 2 iterations or less."""
    dm_settings.save_history_backup = True
    dm_settings.backup_settings.plot = True
    discipline = DisciplineWithFiles()
    design_space = create_design_space()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    scenario = create_scenario(
        discipline,
        "y",
        design_space,
        formulation_settings_model=DisciplinaryOpt_Settings(),
    )
    scenario.execute(LHS_Settings(n_samples=2))

    scenario_path = dm_settings.execution_root_path / "MDOScenario"
    assert (scenario_path / "backup.h5").exists()
    assert not list(scenario_path.glob("*.png"))


def test_unknown_directory_manager_setting(snapshot):
    """Verify that an unknown setting name raises instead of being ignored."""
    with assert_exception(ValidationError, snapshot):
        Settings(enabel=True)


def test_default_execution_root_path(tmp_wd, monkeypatch):
    """Verify that the default root is the cwd at use time, not at import time."""
    work_path = tmp_wd / "work"
    work_path.mkdir()
    monkeypatch.chdir(work_path)
    previous_settings = _configuration.directory_manager
    dm_settings = _configuration.directory_manager = Settings()
    dm_settings.enable = True
    try:
        DisciplineWithFiles().execute()
    finally:
        # The manager cannot be disabled once enabled: restore the previous
        # (disabled) settings instance instead of toggling enable off.
        _configuration.directory_manager = previous_settings

    assert (work_path / "DisciplineWithFiles_execution").is_dir()


def test_enabling_resets_only_the_directory_manager(dm_settings):
    """Verify that enabling resets the manager but not other multitons."""
    manager = DirectoryManager()
    # Capture the currently cached factory rather than the module-level
    # grammar_factory: a prior test may have cleared the whole multiton cache
    # (it is shared by all multitons), in which case the module-level singleton
    # is no longer the cached instance.
    grammar_factory = GrammarFactory()

    # Re-assigning enable evicts only the directory manager cache entry.
    dm_settings.enable = True

    assert DirectoryManager() is not manager
    assert GrammarFactory() is grammar_factory


def test_execution_root_path_creation(tmp_wd):
    """Verify the creation of the execution_root_path."""
    dm_settings = Settings()
    dm_settings.execution_root_path = tmp_wd / "foo"

    assert not dm_settings.execution_root_path.exists()

    dm_settings.enable = True
    assert dm_settings.execution_root_path.exists()

    # Re-trigger the model validator: re-creating the existing directory
    # shall not fail.
    dm_settings.enable = True

    dm_settings.execution_root_path = tmp_wd / "bar"
    assert dm_settings.execution_root_path.exists()

    # Pointing at an already existing directory raises, like the execution
    # subdirectories created at run time.
    existing_path = tmp_wd / "existing"
    existing_path.mkdir()
    with pytest.raises(FileExistsError):
        dm_settings.execution_root_path = existing_path


@pytest.mark.parametrize(
    "name",
    ["gemseo-traces", ".gemseo-traces", "..gemseo-traces", "_.gemseo-traces._"],
)
def test_secure_filename_never_returns_the_trace_registry_directory_name(name):
    """Verify that a sanitized name can never collide with the trace registry.

    The trace registry lives in a `.gemseo-traces` directory under the execution
    root, next to the directories named after the observed objects, whose names
    go through `secure_filename`. What keeps the two apart is the trailing
    `strip("._")` of `secure_filename`, a vendored copy of the werkzeug
    function: were it dropped by a re-synchronization with werkzeug, an object
    named `.gemseo-traces` would be given the registry directory and the
    execution would fail with a `FileExistsError`.

    Args:
        name: The name of an observed object, sanitizing to the registry
            directory name but for its leading dot.
    """
    sanitized_name = secure_filename(name)

    assert sanitized_name == "gemseo-traces"
    assert sanitized_name != ".gemseo-traces"
