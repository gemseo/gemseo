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
from gemseo.util.discipline import DummyDiscipline
from gemseo.util.global_configuration import _configuration
from gemseo.util.platform import PLATFORM_IS_WINDOWS
from gemseo.util.testing.helper import assert_exception

from ...formulation.bilevel_test_helper import create_sobieski_bilevel_bcd_scenario
from ...formulation.bilevel_test_helper import create_sobieski_bilevel_scenario
from .directory_manager_test_helper import build_monolevel_scenario
from .directory_manager_test_helper import create_disc_from_exe
from .directory_manager_test_helper import read_paths_from_txt

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo.scenario.evaluation import EvaluationScenario
    from gemseo.util.typing import StrKeyMapping

REF_DIR_ROOT_PATH = Path(__file__).parent / "reference_directories"
BASE_DIR = Path("root")
PLATFORM = "windows" if PLATFORM_IS_WINDOWS else "linux"


@pytest.fixture(autouse=True)
def deterministic_slsqp(monkeypatch):
    """Make SLSQP deterministic by replacing ScipyOpt._run with a simple loop.

    This evaluates max_iter equally-spaced points, producing the same directory
    structure regardless of SciPy/NumPy version.
    """

    def mock_run(self, problem):
        from gemseo.space.util import get_value_and_bounds

        _, l_b, u_b = get_value_and_bounds(
            problem.design_space, self._settings.normalize_design_space
        )
        max_iter = self._settings.max_iter
        require_grad = self.ALGORITHM_INFOS[self._algo_name].require_gradient

        constraints = self._get_right_sign_constraints(problem)

        for i in range(max_iter):
            t = (i + 1) / (max_iter + 1)
            x = l_b + t * (u_b - l_b)
            problem.objective.evaluate(x)
            for constraint in constraints:
                constraint.evaluate(x)
            if require_grad:
                problem.objective.jac(x)
                for constraint in constraints:
                    constraint.jac(x)

        return "Deterministic mock", 0

    monkeypatch.setattr(
        "gemseo.optimization.scipy_local.scipy_local.ScipyOpt._run",
        mock_run,
    )


def assert_directory_tree(ref_file_path: Path) -> None:
    """Validate the tree against a reference one."""
    root_path = _configuration.directory_manager.execution_root_path
    ref_dir_paths = read_paths_from_txt(ref_file_path, root_path)
    actual_dir_paths = {path for path in root_path.rglob("*") if path.is_dir()}
    assert ref_dir_paths == actual_dir_paths, (
        f"Missing dirs: {ref_dir_paths - actual_dir_paths}"
    )


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


@pytest.mark.parametrize(
    (
        "scenario_type",
        "formulation_settings_model",
        "settings_model",
        "reference_directories",
    ),
    [
        (
            "MDO",
            MDF_Settings(
                main_mda_settings=MDAChain_Settings(
                    inner_mda_settings=MDAGaussSeidel_Settings(max_mda_iter=3)
                ),
            ),
            SLSQP_Settings(max_iter=5),
            "mdo_mdf_sobieski_slsqp_{}_paths_{}.txt",
        ),
        (
            "MDO",
            IDF_Settings(),
            SLSQP_Settings(max_iter=5),
            "mdo_idf_sobieski_slsqp_{}_paths_{}.txt",
        ),
        (
            "MDO",
            MDF_Settings(
                main_mda_settings=MDAChain_Settings(
                    inner_mda_settings=MDAJacobi_Settings(max_mda_iter=3)
                )
            ),
            NLOPT_COBYLA_Settings(max_iter=5),
            "mdo_mdf_sobieski_cobyla_{}_paths_{}.txt",
        ),
        (
            "MDO",
            IDF_Settings(),
            NLOPT_COBYLA_Settings(max_iter=5),
            "mdo_idf_sobieski_cobyla_{}_paths_{}.txt",
        ),
        (
            "DOE",
            MDF_Settings(
                main_mda_settings=MDAChain_Settings(
                    inner_mda_settings=MDAJacobi_Settings(max_mda_iter=3)
                )
            ),
            LHS_Settings(n_samples=5),
            "doe_mdf_sobieski_{}_paths_{}.txt",
        ),
        (
            "DOE",
            IDF_Settings(),
            LHS_Settings(n_samples=5),
            "doe_idf_sobieski_{}_paths_{}.txt",
        ),
    ],
)
@parametrized_clean_up_policy
def test_monolevel_scenarios_all_policies(
    dm_settings,
    scenario_type,
    formulation_settings_model,
    settings_model,
    reference_directories,
    clean_up_policy,
):
    """Test the creation of directories for the corresponding policy for a mono-level
    scenario."""
    dm_settings.clean_up_policy = clean_up_policy

    scenario = build_monolevel_scenario(formulation_settings_model)
    scenario.execute(settings_model)

    ref_file_path = (
        REF_DIR_ROOT_PATH
        / formulation_settings_model.target_class_name
        / scenario_type
        / PLATFORM
        / reference_directories.format(PLATFORM, clean_up_policy)
    )
    assert_directory_tree(ref_file_path)


@pytest.mark.parametrize(
    ("mda_clean_up_policy", "reference_directories"),
    [
        (
            MDACleanUpPolicy.KEEP_ALL,
            "mda_jacobi_sobieski_{}_paths_{}.txt",
        ),
        (
            MDACleanUpPolicy.KEEP_LAST_ONLY,
            "mda_jacobi_sobieski_{}_paths_{}.txt",
        ),
    ],
)
def test_mda_clean_up_policies_for_mono_level_scenarios(
    dm_settings, mda_clean_up_policy, reference_directories
):
    """Test the clean policies for the MDAs with a mono-level scenario."""
    dm_settings.clean_up_policy = CleanUpPolicy.KEEP_BASELINE_AND_SOLUTION
    dm_settings.mda_clean_up_policy = mda_clean_up_policy

    scenario = build_monolevel_scenario(
        MDF_Settings(
            main_mda_settings=MDAChain_Settings(
                inner_mda_settings=MDAJacobi_Settings(max_mda_iter=3)
            )
        ),
    )
    scenario.execute(SLSQP_Settings(max_iter=3))

    ref_file_path = (
        REF_DIR_ROOT_PATH
        / "mda"
        / reference_directories.format(PLATFORM, mda_clean_up_policy)
    )
    assert_directory_tree(ref_file_path)


@parametrized_clean_up_policy
@pytest.mark.skipif(
    PLATFORM_IS_WINDOWS,
    reason="Currently fails on Windows with pytest, works without pytest",
)
def test_directory_manager_with_multiprocessing(
    dm_settings, generate_sobieski_bilevel_scenario, clean_up_policy
):
    """Test the correct creation of directories when using multiprocessing."""
    dm_settings.clean_up_policy = clean_up_policy

    settings_model = LHS_Settings(n_samples=4, n_processes=4)
    scenario = generate_sobieski_bilevel_scenario(
        main_mda_settings=MDAJacobi_Settings(max_mda_iter=2),
    )
    scenario.execute(settings_model)

    reference_directories = "doe_bilevel_sobieski_parallel_{}_paths_{}.txt"
    ref_file_path = (
        REF_DIR_ROOT_PATH
        / "bilevel"
        / "DOE"
        / PLATFORM
        / reference_directories.format(PLATFORM, clean_up_policy)
    )
    assert_directory_tree(ref_file_path)


def test_directory_manager_with_spawn_multiprocessing(dm_settings, monkeypatch):
    """Verify the directories created by workers of the spawn start method."""
    monkeypatch.setattr(
        CallableParallelExecution, "MULTI_PROCESSING_START_METHOD", "spawn"
    )
    discipline = create_discipline("AnalyticDiscipline", expressions={"y": "2*x"})
    design_space = create_design_space()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
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
    ("scenario_type", "settings_model", "reference_directories"),
    [
        (
            "MDO",
            NLOPT_COBYLA_Settings(max_iter=5),
            "mdo_bilevel_sobieski_{}_paths_{}.txt",
        ),
        ("DOE", LHS_Settings(n_samples=5), "doe_bilevel_sobieski_{}_paths_{}.txt"),
    ],
)
@parametrized_clean_up_policy
@pytest.mark.xfail(
    PLATFORM_IS_WINDOWS,
    reason="Windows can't handle directory paths that are too long.",
)
def test_all_policies_sobieski_bilevel(
    dm_settings,
    generate_sobieski_bilevel_scenario,
    scenario_type,
    settings_model,
    reference_directories,
    clean_up_policy,
):
    """Test the directory creation for the bilevel formulation."""
    dm_settings.clean_up_policy = clean_up_policy

    scenario = generate_sobieski_bilevel_scenario(
        main_mda_settings=MDAChain_Settings(max_mda_iter=2),
    )
    scenario.execute(settings_model)

    ref_file_path = (
        REF_DIR_ROOT_PATH
        / "bilevel"
        / scenario_type
        / PLATFORM
        / reference_directories.format(PLATFORM, clean_up_policy)
    )
    assert_directory_tree(ref_file_path)


@pytest.mark.parametrize(
    ("scenario_type", "settings_model", "reference_directories"),
    [
        (
            "MDO",
            NLOPT_COBYLA_Settings(max_iter=3),
            "mdo_bilevel_bcd_sobieski_{}_paths_{}.txt",
        ),
        ("DOE", LHS_Settings(n_samples=3), "doe_bilevel_bcd_sobieski_{}_paths_{}.txt"),
    ],
)
@parametrized_clean_up_policy
@pytest.mark.xfail(
    PLATFORM_IS_WINDOWS,
    reason="Windows can't handle directory paths that are too long.",
)
def test_all_policies_bilevel_bcd_sobieski(
    dm_settings,
    generate_sobieski_bilevel_bcd_scenario,
    scenario_type,
    settings_model,
    reference_directories,
    clean_up_policy,
):
    """Test the directory creation for the bilevel bcd formulation."""
    dm_settings.clean_up_policy = clean_up_policy

    short_names = PLATFORM == "windows"

    scenario = generate_sobieski_bilevel_bcd_scenario(
        short_names=short_names,
    )
    scenario.formulation._mda1.inner_mdas[0].settings = MDAJacobi_Settings(
        max_mda_iter=2
    )
    scenario.formulation._mda2.inner_mdas[0].settings = MDAJacobi_Settings(
        max_mda_iter=2
    )
    scenario.formulation._bcd_mda.settings = MDAGaussSeidel_Settings(max_mda_iter=2)
    for scenario_adapter in scenario.formulation._scenario_adapters:
        scenario_adapter.scenario.set_algorithm(SLSQP_Settings(max_iter=3))
        scenario_adapter.scenario.formulation.mda.settings = MDAGaussSeidel_Settings(
            max_mda_iter=2
        )
    if short_names:
        scenario.formulation._mda1.inner_mdas[0].name = "MDA1"
        scenario.formulation._mda2.inner_mdas[0].name = "MDA2"

    scenario.execute(settings_model)

    ref_file_path = (
        REF_DIR_ROOT_PATH
        / "bcd"
        / scenario_type
        / PLATFORM
        / reference_directories.format(PLATFORM, clean_up_policy)
    )
    ref_dir_paths = read_paths_from_txt(ref_file_path, dm_settings.execution_root_path)
    actual_dir_paths = {
        path for path in dm_settings.execution_root_path.rglob("*") if path.is_dir()
    }
    # There can be legitimate variations in the execution of a BCD scenario when it is
    # executed in different machines, mostly because of the gradient-based optimizer at
    # the sub-scenario level. We test only that the generated directory tree
    # includes at least the reference directories.
    # In case of failure, check the generated directories and verify that the executed
    # workflow is consistent with the directory tree, then update the reference file.
    assert ref_dir_paths.issubset(ref_dir_paths), (
        f"Extra dirs: {ref_dir_paths - actual_dir_paths}"
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
    design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
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
    design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
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
    design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
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
    design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
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
    PLATFORM_IS_WINDOWS,
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
    design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)

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
    design_space.add_variable("a", lower_bound=0.0, upper_bound=10.0, value=1.0)
    design_space.add_variable("b", lower_bound=0.0, upper_bound=10.0, value=1.0)
    design_space.add_variable("c", lower_bound=0.0, upper_bound=10.0, value=1.0)
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
    design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    scenario = create_scenario(
        discipline,
        "y",
        design_space,
        formulation_settings_model=DisciplinaryOpt_Settings(),
        name="优化场景",
    )
    with assert_exception(ValueError, snapshot):
        scenario.execute(LHS_Settings(n_samples=1))


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
    design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
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
    design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
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
    design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
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


def test_history_view_skipped_for_short_history(dm_settings):
    """Verify that no history view is plotted with 2 iterations or less."""
    dm_settings.save_history_backup = True
    dm_settings.backup_settings.plot = True
    discipline = DisciplineWithFiles()
    design_space = create_design_space()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
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
    # GRAMMAR_FACTORY: a prior test may have cleared the whole multiton cache
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
