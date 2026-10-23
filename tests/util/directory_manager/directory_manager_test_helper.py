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

"""Provide useful functions for DirectoryManager testing."""

from __future__ import annotations

import re
from numbers import Real
from operator import itemgetter
from typing import TYPE_CHECKING
from typing import Final

from numpy import asarray
from numpy.linalg import norm

from gemseo import MDOScenario
from gemseo.core.discipline.discipline import Discipline
from gemseo.core.function.array_function import ArrayFunction
from gemseo.discipline.wrapper.disc_from_exe import DiscFromExe
from gemseo.formulation.idf_settings import IDF_Settings
from gemseo.problem.mdo.sobieski.discipline import SobieskiAerodynamics
from gemseo.problem.mdo.sobieski.discipline import SobieskiMission
from gemseo.problem.mdo.sobieski.discipline import SobieskiPropulsion
from gemseo.problem.mdo.sobieski.discipline import SobieskiStructure
from gemseo.problem.mdo.sobieski.standalone.design_space import SobieskiDesignSpace
from gemseo.space.design import DesignSpace
from gemseo.util._directory_manager.settings import CleanUpPolicy
from gemseo.util._directory_manager.settings import MDACleanUpPolicy
from gemseo.util._directory_manager.settings import Settings
from gemseo.util.global_configuration import _configuration

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping
    from pathlib import Path

    import pytest
    from pydantic import BaseModel

    from gemseo.formulation.core.base_settings import BaseFormulationSettings
    from gemseo.scenario.evaluation import EvaluationScenario
    from gemseo.util.typing import RealOrComplexArray
    from gemseo.util.typing import StrKeyMapping


def build_monolevel_scenario(
    formulation_settings_model: BaseFormulationSettings,
    **args,
) -> EvaluationScenario:
    """Build the scenario for SSBJ.

    Args:
        formulation_settings_model: The formulation settings model.

    Returns:
        The MDOScenario.
    """
    disciplines = [
        SobieskiPropulsion(),
        SobieskiAerodynamics(),
        SobieskiMission(),
        SobieskiStructure(),
    ]

    design_space = SobieskiDesignSpace()
    scenario = MDOScenario(
        disciplines=disciplines,
        design_space=design_space,
        formulation_settings=formulation_settings_model,
        **args,
    )
    scenario.add_objective("y_4", minimize=False)
    for c_name in ["g_1", "g_2", "g_3"]:
        scenario.add_constraint(
            c_name, constraint_type=ArrayFunction.ConstraintType.INEQ
        )
    return scenario


class DummyDiscipline1(Discipline):
    """A discipline that does nothing."""

    def __init__(
        self,
        name: str = "",
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """
        Args:
            input_names: The names of the input variables, if any.
            output_names: The names of the output variables, if any.
        """  # noqa: D205 D212 D415
        super().__init__(name=name)
        self.io.input_grammar.update_from_names(input_names)
        self.io.output_grammar.update_from_names(output_names)

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        y = input_data["a"] * 2

        return {"y": y}


class DummyDiscipline2(DummyDiscipline1):
    """A discipline where the `_run` method calls for the parent's `_run` method."""

    def __init__(
        self,
        name: str = "",
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """
        Args:
            input_names: The names of the input variables, if any.
            output_names: The names of the output variables, if any.
        """  # noqa: D205 D212 D415
        super().__init__(name=name)
        self.io.input_grammar.update_from_names(input_names)
        self.io.output_grammar.update_from_names(output_names)

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        z = input_data["b"] * super()._run(input_data)

        return {"z": z}


def create_scenario_with_inheriting_disciplines():
    """Create a scenario with one discipline that inherits from another."""
    discipline_a = DummyDiscipline1(
        name="DisciplineA", input_names="a", output_names="y"
    )

    discipline_b = DummyDiscipline2(
        name="DisciplineB", input_names="b", output_names="z"
    )
    disciplines = [discipline_a, discipline_b]

    ds = DesignSpace()
    ds.add_real_variable("a", lower_bound=-1.0, upper_bound=1.0, value=0.0)
    ds.add_real_variable("b", lower_bound=-1.0, upper_bound=1.0, value=0.0)

    scenario = MDOScenario(
        disciplines=disciplines, design_space=ds, formulation_settings=IDF_Settings()
    )
    scenario.add_objective("z", minimize=False)
    return scenario


_excluded_directory_names: Final[frozenset[str]] = frozenset({
    ".gemseo-traces",
    ".gemseo-trace.arrays",
})
"""The names of the directories always excluded from a directory tree.

They hold the traced array files of the data-lineage feature and are not part
of the workflow structure that the cleanup policies operate on.
"""

_leaf_discipline_suffixes: Final[tuple[str, str]] = ("_execution", "_linearization")
"""The suffixes of a leaf discipline directory name."""

_scenario_child_name_re: Final[re.Pattern[str]] = re.compile(
    r"(?:Optimizer_iteration|DOE_sample)_(\d+)$"
)
"""The pattern of a scenario directory's managed child directory name."""

_homonym_suffix_re: Final[re.Pattern[str]] = re.compile(r"#\d+$")
"""The pattern of the homonym disambiguation suffix of a directory name."""


def _strip_homonym_suffix(name: str) -> str:
    """Strip the trailing homonym disambiguation suffix from a directory name.

    The manager appends a `#<n>` suffix (e.g. `MDAJacobi#0`,
    `Foo_execution#0`) when the same observee is executed more than once
    under the same parent directory.

    Args:
        name: A directory name, possibly suffixed with `#<n>`.

    Returns:
        `name` without its trailing `#<n>` suffix, if any.
    """
    return _homonym_suffix_re.sub("", name)


def get_directory_tree(root_path: Path, structural_only: bool = False) -> list[str]:
    """Return the sorted relative paths of all the directories under a root.

    Args:
        root_path: The root directory to walk.
        structural_only: Whether to drop the leaf discipline directories
            (execution and linearization directories), keeping only the
            directories that reflect the scenario and MDA structure.

    Returns:
        The sorted POSIX-style paths, relative to `root_path`, of every
        directory found under it.
    """
    tree = []
    for path in root_path.rglob("*"):
        if not path.is_dir():
            continue

        relative_path = path.relative_to(root_path)
        parts = relative_path.parts
        if any(part in _excluded_directory_names for part in parts):
            continue

        if structural_only and any(
            _strip_homonym_suffix(part).endswith(_leaf_discipline_suffixes)
            for part in parts
        ):
            continue

        tree.append(relative_path.as_posix())

    return sorted(tree)


def _split_parent(path: str) -> str:
    """Return the POSIX parent of a directory tree path.

    Args:
        path: A POSIX relative path, as returned by `get_directory_tree`.

    Returns:
        The parent path, or the empty string if `path` has a single
        component (i.e. it is a top-level directory).
    """
    parent, _, _ = path.rpartition("/")
    return parent


def _split_name(path: str) -> str:
    """Return the last component of a directory tree path.

    Args:
        path: A POSIX relative path, as returned by `get_directory_tree`.

    Returns:
        The last path component.
    """
    return path.rpartition("/")[2]


def _group_children_by_parent(tree: Iterable[str]) -> dict[str, list[str]]:
    """Group the paths of a directory tree by their parent path.

    Args:
        tree: The sorted relative paths of a directory tree, as returned by
            `get_directory_tree`.

    Returns:
        A mapping from a parent path (the empty string for the top level, as
        returned by `_split_parent`) to the list of its direct children.
    """
    children_by_parent: dict[str, list[str]] = {}
    for path in tree:
        children_by_parent.setdefault(_split_parent(path), []).append(path)
    return children_by_parent


def _classify_directories(
    children_by_parent: Mapping[str, list[str]],
) -> tuple[set[str], set[str]]:
    """Identify the scenario and the MDA directories of a directory tree.

    A directory is a scenario directory when at least one of its direct
    children is named `Optimizer_iteration_<n>` or `DOE_sample_<n>`. A
    directory is an MDA directory when at least one of its direct children is
    named after itself, suffixed with `_iteration_<n>` (e.g. the `MDAJacobi`
    directory has an `MDAJacobi_iteration_0` child): this is how an MDA
    solver (Jacobi, Gauss-Seidel, Newton-Raphson, ...) nested anywhere in the
    tree, including inside an `MDAChain`, is recognized, since only the
    solver itself, not the chain, is given a directory.

    Args:
        children_by_parent: The direct children of each directory, as
            returned by `_group_children_by_parent`.

    Returns:
        * The paths of the scenario directories.
        * The paths of the MDA directories.
    """
    scenario_directories = set()
    mda_directories = set()
    for parent, children in children_by_parent.items():
        names = [_split_name(child) for child in children]
        if any(_scenario_child_name_re.fullmatch(name) for name in names):
            scenario_directories.add(parent)

        if parent:
            prefix = f"{_strip_homonym_suffix(_split_name(parent))}_iteration_"
            if any(
                name.startswith(prefix) and name[len(prefix) :].isdigit()
                for name in names
            ):
                mda_directories.add(parent)

    return scenario_directories, mda_directories


def _select_scenario_children(
    child_paths: Iterable[str],
    policy: CleanUpPolicy,
    is_top_level: bool,
    optimum_iteration: int,
    actual_tree: set[str],
) -> list[str]:
    """Return the children of a scenario directory kept by a cleanup policy.

    Args:
        child_paths: The direct children of the scenario directory.
        policy: The scenario cleanup policy.
        is_top_level: Whether the scenario directory is the top-level one, of
            which `optimum_iteration` is the exact optimum iteration.
        optimum_iteration: The database iteration of the top-level scenario's
            optimum. Ignored for a directory that is not the top-level one.
        actual_tree: The directory tree actually produced by running a fresh
            scenario under `policy`, used to resolve which iteration a nested
            scenario directory's own (unknown) optimum kept, for the
            solution-based policies.

    Returns:
        The children to keep. For `KEEP_ALL`, every other child (a managed
        directory that does not carry an iteration suffix) is kept too, along
        with its whole subtree; every other policy prunes them, since it
        keeps only the last / solution / baseline-and-solution iteration
        directories.
    """
    numbered = []
    other = []
    for path in child_paths:
        match = _scenario_child_name_re.fullmatch(_split_name(path))
        if match is None:
            other.append(path)
        else:
            numbered.append((int(match.group(1)), path))

    if policy == CleanUpPolicy.KEEP_ALL:
        return [path for _, path in numbered] + other

    if policy == CleanUpPolicy.KEEP_LAST_ONLY:
        kept = [max(numbered, key=itemgetter(0))[1]] if numbered else []
    elif is_top_level:
        if policy == CleanUpPolicy.KEEP_SOLUTION_ONLY:
            kept = [path for suffix, path in numbered if suffix == optimum_iteration]
        else:
            _, baseline_path = min(numbered, key=itemgetter(0))
            kept = list({
                baseline_path,
                *(path for suffix, path in numbered if suffix == optimum_iteration),
            })
    else:
        candidates = {path for _, path in numbered}
        kept_in_actual = candidates & actual_tree
        assert kept_in_actual, (
            f"The oracle could not identify which of {candidates} the actual "
            f"run kept under {policy}: none of them is in the actual tree."
        )
        if policy == CleanUpPolicy.KEEP_SOLUTION_ONLY:
            assert len(kept_in_actual) == 1, (
                f"{policy} must keep exactly one sub-scenario iteration "
                f"directory, the actual run kept {kept_in_actual}."
            )
        else:
            _, baseline_path = min(numbered, key=itemgetter(0))
            assert baseline_path in kept_in_actual, (
                f"{policy} must keep the baseline directory {baseline_path}, "
                f"the actual run kept {kept_in_actual}."
            )
            assert len(kept_in_actual) <= 2, (
                f"{policy} must keep at most the baseline directory and one "
                f"other iteration, the actual run kept {kept_in_actual}."
            )
        kept = list(kept_in_actual)

    return kept


def _select_mda_children(
    dir_path: str, child_paths: Iterable[str], mda_policy: MDACleanUpPolicy
) -> list[str]:
    """Return the children of an MDA directory kept by a cleanup policy.

    Args:
        dir_path: The path of the MDA directory.
        child_paths: The direct children of the MDA directory.
        mda_policy: The MDA cleanup policy.

    Returns:
        The children to keep. For `KEEP_ALL`, every other child (a managed
        directory that does not carry an iteration suffix, e.g. a discipline
        executed once before the solver loop) is kept too; `KEEP_LAST_ONLY`
        prunes them, like `_select_scenario_children` does for its own
        non-suffixed children, since the manager removes every managed
        sub-directory but the last one regardless of whether it is an MDA or
        a scenario directory (`DirectoryManager.__get_removals_keep_last`).
    """
    prefix = f"{_strip_homonym_suffix(_split_name(dir_path))}_iteration_"
    numbered = []
    other = []
    for path in child_paths:
        name = _split_name(path)
        tail = name[len(prefix) :] if name.startswith(prefix) else ""
        if tail.isdigit():
            numbered.append((int(tail), path))
        else:
            other.append(path)

    if mda_policy == MDACleanUpPolicy.KEEP_ALL:
        return [path for _, path in numbered] + other

    return [max(numbered, key=itemgetter(0))[1]] if numbered else []


def derive_policy_tree(
    keep_all_tree: Iterable[str],
    policy: CleanUpPolicy,
    optimum_iteration: int,
    actual_tree: Iterable[str],
    mda_policy: MDACleanUpPolicy = MDACleanUpPolicy.KEEP_ALL,
) -> list[str]:
    """Derive the directory tree expected under a cleanup policy.

    This is an oracle, independent from
    [DirectoryManager][gemseo.util._directory_manager.manager.DirectoryManager]:
    it is written from the cleanup policy definitions (see the module
    docstring of `gemseo.util._directory_manager.settings`), not from the
    manager's implementation, so that comparing its prediction against an
    actual run is a meaningful check of the manager's behavior.

    For the top-level scenario directory (the first path component of
    `keep_all_tree`), the derivation is exact: the solution-based policies use
    `optimum_iteration`. For a nested scenario directory (e.g. a BiLevel
    sub-scenario), `KEEP_LAST_ONLY` is still exact (the largest suffix among
    its `KEEP_ALL` children); for the solution-based policies, the optimum of
    that nested execution is not known here, so the child actually kept by
    `actual_tree` is looked up, checked for consistency (it must be one of the
    `KEEP_ALL` children, in the expected number), and adopted: this is why,
    for the solution-based policies, `check_cleanup_policy` also compares
    `actual_tree` to an exact, per-policy syrupy snapshot in addition to this
    oracle's prediction, since the oracle by itself cannot pin which nested
    sub-scenario iteration is kept.

    Args:
        keep_all_tree: The directory tree produced by a `KEEP_ALL` /
            `MDACleanUpPolicy.KEEP_ALL` run, as returned by
            `get_directory_tree`.
        policy: The scenario cleanup policy to derive the tree for.
        optimum_iteration: The database iteration of the top-level scenario's
            optimum.
        actual_tree: The directory tree actually produced by running a fresh
            scenario under `policy` and `mda_policy`, as returned by
            `get_directory_tree`. Only used to resolve the solution-based
            policies for nested scenario directories.
        mda_policy: The MDA cleanup policy to derive the tree for.

    Returns:
        The sorted paths of the directories expected to remain.
    """
    children_by_parent = _group_children_by_parent(keep_all_tree)
    scenario_directories, mda_directories = _classify_directories(children_by_parent)
    actual_set = set(actual_tree)

    def _prune(dir_path: str, is_top_level: bool) -> set[str]:
        """Return the kept paths of a directory and its descendants.

        Args:
            dir_path: The directory path to prune.
            is_top_level: Whether `dir_path` is the top-level scenario
                directory.

        Returns:
            The kept paths, including `dir_path` itself.
        """
        kept = {dir_path} if dir_path else set()
        children = children_by_parent.get(dir_path, [])
        if dir_path in scenario_directories:
            selected = _select_scenario_children(
                children, policy, is_top_level, optimum_iteration, actual_set
            )
        elif dir_path in mda_directories:
            selected = _select_mda_children(dir_path, children, mda_policy)
        else:
            selected = children

        for child in selected:
            kept |= _prune(child, False)

        return kept

    result: set[str] = set()
    for top_level_directory in children_by_parent.get("", []):
        result |= _prune(top_level_directory, True)

    return sorted(result)


def assert_policies_distinguishable(
    scenario: EvaluationScenario, *, require_feasible: bool = True
) -> None:
    """Verify that the cleanup policies produce genuinely different trees.

    The `KEEP_SOLUTION_ONLY` and `KEEP_BASELINE_AND_SOLUTION` policies are
    meaningfully tested only if the optimum they single out is neither the
    first database iteration (kept by `KEEP_BASELINE_AND_SOLUTION` anyway) nor
    the last one (kept by `KEEP_LAST_ONLY` anyway), and is unambiguously
    better than the runner-up point, so that the optimum iteration is stable
    across platforms and dependency versions.

    Args:
        scenario: The executed top-level scenario.
        require_feasible: Whether the optimum must be feasible. Disable this
            for a formulation whose coupling consistency constraints cannot
            be satisfied by the deterministic sweep (e.g. IDF, where the
            coupling variables are free design variables that the sweep does
            not solve for): the optimum iteration is still required to be
            interior, but it may then be infeasible.

    Raises:
        AssertionError: If `require_feasible` and the optimum is infeasible;
            if the optimum is the first or the last database iteration; or if
            it is not clearly separated from the second best point. When the
            optimum is feasible, the gap is checked on the objective of the
            two best *feasible* points; when it is not (only possible with
            `require_feasible=False`), `OptimizationHistory.optimum` instead
            selects the point with the smallest constraint violation, so the
            gap is checked on the constraint violation of the two least
            infeasible points of the whole database. The sweep points or the
            number of iterations of the test must then be changed so that the
            optimum lands on a distinct, interior iteration; otherwise the
            cleanup policies under test would produce identical or unstable
            directory trees.
    """
    problem = scenario.formulation.problem
    database = problem.database
    _, x_opt, is_feasible, *_ = problem.optimum
    if require_feasible:
        assert is_feasible, "The optimum of the scenario is infeasible."

    optimum_iteration = database.get_iteration(x_opt)
    n_iterations = len(database)
    assert 1 < optimum_iteration < n_iterations, (
        f"The optimum iteration ({optimum_iteration} out of {n_iterations}) "
        "is the first or the last one."
    )

    if is_feasible:
        _, outputs = problem.history.feasible_points
        # The database stores the standardized objective (negated for a
        # maximization problem), not the original one returned by
        # `problem.objective_name`.
        objective_name = problem.standardized_objective_name
        values = sorted(_as_scalar(output[objective_name]) for output in outputs)
        quantity = "feasible objective"
    else:
        # No feasible point exists: `problem.optimum` picked the point with
        # the smallest constraint violation instead (see
        # `OptimizationHistory.optimum`), so the separation must be checked
        # on that same quantity, over the whole database.
        values = sorted(
            problem.history.check_design_point_is_feasible(x)[1] for x in database
        )
        quantity = "constraint violation"

    assert len(values) > 1, (
        "At least two points are needed to check that the optimum is well separated."
    )
    best, runner_up = values[0], values[1]
    relative_gap = abs(runner_up - best) / max(abs(best), abs(runner_up), 1e-30)
    assert relative_gap > 1e-6, (
        f"The best {quantity} ({best}) is too close to the runner-up "
        f"({runner_up}), a relative gap of {relative_gap:.2e}."
    )


def _as_scalar(value: RealOrComplexArray | float) -> float:
    """Return a scalar view of a (possibly vector) objective value.

    Args:
        value: The raw objective value, as stored in the database.

    Returns:
        The scalar objective value, using the Euclidean norm for a
        vector-valued objective, as
        [OptimizationHistory.optimum][gemseo.optimization.history.OptimizationHistory.optimum]
        does.
    """
    if isinstance(value, Real):
        return float(value)
    array_value = asarray(value)
    return float(array_value[0]) if array_value.size == 1 else float(norm(array_value))


def _run_fresh(
    build_scenario: Callable[[], EvaluationScenario],
    algo_settings_model: BaseModel,
    root_path: Path,
    policy: CleanUpPolicy,
    mda_policy: MDACleanUpPolicy,
    *,
    structural_only: bool = False,
) -> tuple[EvaluationScenario, list[str], int]:
    """Build, execute and measure a scenario under a fresh execution root.

    A new `Settings` instance is enabled with `root_path` (which must not
    exist yet) and the given policies, becoming the active directory manager
    configuration; `build_scenario` is then called and the resulting scenario
    executed under it.

    Args:
        build_scenario: A callable building a fresh, unexecuted scenario.
        algo_settings_model: The algorithm settings to execute the scenario
            with. It is copied before execution, so that the same model
            instance can be passed by every caller.
        root_path: The (not yet existing) execution root for this run.
        policy: The scenario cleanup policy to run under.
        mda_policy: The MDA cleanup policy to run under.
        structural_only: Whether the returned tree should compare only the
            scenario and MDA structure, dropping the leaf discipline
            directories.

    Returns:
        * The executed scenario.
        * The directory tree it produced.
        * The database iteration of its optimum, needed by `derive_policy_tree`.

    Raises:
        AssertionError: If `algo_settings_model` is a DOE's settings (it has
            an `n_samples` attribute) and the database does not hold exactly
            `n_samples` points: a failed sample is skipped by the DOE, which
            shifts the `DOE_sample_<n>` directory suffixes away from the
            database iterations that `derive_policy_tree` and the snapshots
            assume they match.
    """
    settings = Settings()
    settings.enable = True
    settings.execution_root_path = root_path
    settings.clean_up_policy = policy
    settings.mda_clean_up_policy = mda_policy
    _configuration.directory_manager = settings

    scenario = build_scenario()
    scenario.execute(algo_settings_model.model_copy(deep=True))
    tree = get_directory_tree(root_path, structural_only=structural_only)
    problem = scenario.formulation.problem
    n_samples = getattr(algo_settings_model, "n_samples", None)
    if n_samples is not None:
        assert len(problem.database) == n_samples, (
            f"{len(problem.database)} database iterations for {n_samples} "
            "DOE samples: a failed sample was skipped, which shifts the "
            "`DOE_sample_<n>` directory suffixes away from the database "
            "iterations."
        )
    optimum_iteration = problem.database.get_iteration(problem.optimum[1])
    return scenario, tree, optimum_iteration


ReferenceKey = tuple[str, tuple[tuple[str, str], ...]]
"""The type of a `KEEP_ALL` reference key: test name and parametrization."""


# This module-level cache is cross-test state, kept on purpose to save run time:
# a KEEP_ALL run is the most expensive part of these tests and is shared by the
# items of every policy of a configuration. The price is that the first item of
# a configuration to run, which depends on the collection order and on the xdist
# distribution, builds the reference and runs `assert_policies_distinguishable`,
# so a failure of the reference build is reported on that item, and a run
# selecting a single policy (e.g. `-k KEEP_SOLUTION_ONLY`) builds the reference
# itself. A fixture cannot replace it: the configuration is a test-level
# parametrization and the reference depends on the function-scoped `tmp_wd` and
# `dm_settings` fixtures and on the scenario builder defined in each test.
_keep_all_references: Final[dict[ReferenceKey, tuple[list[str], int]]] = {}
"""The KEEP_ALL reference tree and optimum iteration, cached per configuration.

Keyed by the reference key returned by `build_reference_key`, i.e. by every
parametrization of a test but its `policy`, so that the (expensive) `KEEP_ALL`
run of a given configuration is executed at most once per test process,
shared by the test items of every policy exercising that same configuration,
instead of once per test item. A configuration is stored only once its
`KEEP_ALL` run has passed `assert_policies_distinguishable`; a failing run is
never cached, so that a later test item sharing the same key retries it.
"""


def build_reference_key(request: pytest.FixtureRequest) -> ReferenceKey:
    """Build the `_keep_all_references` cache key of a per-policy test item.

    The key identifies a test configuration independently of which policy it
    is currently exercising, so that every policy's test item sharing the
    same configuration (the same parametrization but for `policy`) reuses the
    same `KEEP_ALL` reference.

    Args:
        request: The pytest request of the test item, parametrized with
            `parametrized_policy` (its `callspec.params` must hold a
            `"policy"` entry, alongside any other parametrization).

    Returns:
        The test's `originalname`, paired with the sorted, `repr`-ed
        parametrization values other than `policy`. Tests parametrized only
        on `policy` therefore get a key equivalent to their bare
        `originalname`.
    """
    params = request.node.callspec.params
    return (
        request.node.originalname,
        tuple(
            (name, repr(value))
            for name, value in sorted(params.items())
            if name != "policy"
        ),
    )


def check_cleanup_policy(
    build_scenario: Callable[[], EvaluationScenario],
    algo_settings_model: BaseModel,
    tmp_wd: Path,
    snapshot,
    policy: CleanUpPolicy,
    reference_key: ReferenceKey,
    *,
    structural_only: bool = False,
    require_feasible: bool = True,
) -> None:
    """Execute a scenario under one cleanup policy and check the resulting tree.

    The `KEEP_ALL` / `MDACleanUpPolicy.KEEP_ALL` reference tree and optimum
    iteration of `reference_key`'s configuration are fetched from
    `_keep_all_references`, building and caching them first if this is the
    first policy of that configuration to run (see `build_reference_key`).

    For `policy == CleanUpPolicy.KEEP_ALL`, the reference tree itself is
    compared to the syrupy snapshot. For every other policy, a fresh scenario
    is built and executed under its own execution root (see `_run_fresh`),
    checked for determinism against the reference's optimum iteration, its
    tree compared to its own syrupy snapshot (an exact, per-policy pin of
    which nested sub-scenario iteration is kept, since the oracle alone
    cannot resolve that, see `derive_policy_tree`), and cross-checked against
    the oracle's prediction.

    Args:
        build_scenario: A callable building a fresh, unexecuted scenario
            identical to the one used by every other policy of the same
            configuration, so that a new instance can be executed under each.
        algo_settings_model: The algorithm settings to execute the scenario
            with.
        tmp_wd: The working directory under which the execution roots are
            created.
        snapshot: The syrupy snapshot assertion fixture.
        policy: The scenario cleanup policy under test.
        reference_key: This test configuration's cache key, as returned by
            `build_reference_key`.
        structural_only: Whether to compare only the scenario and MDA
            structure of the tree, dropping the leaf discipline directories.
        require_feasible: Passed to `assert_policies_distinguishable`, only
            used while building the reference; disable it for a formulation
            whose coupling consistency constraints cannot be satisfied by the
            deterministic sweep (e.g. IDF).
    """
    reference = _keep_all_references.get(reference_key)
    if reference is None:
        scenario, keep_all_tree, optimum_iteration = _run_fresh(
            build_scenario,
            algo_settings_model,
            tmp_wd / "ref",
            CleanUpPolicy.KEEP_ALL,
            MDACleanUpPolicy.KEEP_ALL,
            structural_only=structural_only,
        )
        assert_policies_distinguishable(scenario, require_feasible=require_feasible)
        reference = keep_all_tree, optimum_iteration
        _keep_all_references[reference_key] = reference

    keep_all_tree, optimum_iteration = reference

    if policy == CleanUpPolicy.KEEP_ALL:
        assert keep_all_tree == snapshot
        return

    _, actual_tree, fresh_optimum_iteration = _run_fresh(
        build_scenario,
        algo_settings_model,
        tmp_wd / "run",
        policy,
        MDACleanUpPolicy.KEEP_ALL,
        structural_only=structural_only,
    )
    assert fresh_optimum_iteration == optimum_iteration, (
        "The scenario execution is not deterministic: the optimum "
        f"iteration was {optimum_iteration} under KEEP_ALL and "
        f"{fresh_optimum_iteration} under {policy}."
    )
    assert actual_tree == snapshot
    assert actual_tree == derive_policy_tree(
        keep_all_tree, policy, optimum_iteration, actual_tree
    )


def create_disc_from_exe(file_path: Path) -> DiscFromExe:
    """Create an executable discipline for testing."""
    sum_path = str(file_path / "sum_data.py")
    exec_cmd = f"python {sum_path} -i input.json -o output.json"

    disc: DiscFromExe = DiscFromExe(
        input_template=str(file_path / "input.json.template"),
        output_template=str(file_path / "output.json.template"),
        root_directory="",
        command_line=exec_cmd,
        input_filename="input.json",
        output_filename="output.json",
    )

    return disc
