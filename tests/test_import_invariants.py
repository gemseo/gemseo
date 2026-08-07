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
"""Tests enforcing the v7 core-layout import dependency rules.

This module AST-scans every module-level import under ``src/gemseo``
(skipping ``if TYPE_CHECKING:`` bodies and function-local imports, which are
lazy by design) and enforces:

1. ``gemseo.core`` does not depend on any domain package.
2. Domain-core SPI packages (``gemseo.optimization.core``, ``gemseo.doe.core``,
   ...) do not depend on sibling top-level domains, except for a short,
   measured and explicitly commented list of exceptions.
3. The overall per-top-level-package dependency graph does not silently grow:
   new second-segment edges must be added to the allowlist below explicitly.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING
from typing import NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterator

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "gemseo"


class _Finding(NamedTuple):
    """A single module-level import of a ``gemseo`` module."""

    file: str
    lineno: int
    module: str
    segment: str
    package: str


def _is_type_checking(test: ast.expr) -> bool:
    """Return whether an ``if`` test is (an alias of) ``typing.TYPE_CHECKING``."""
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _module_dotted_name(path: Path) -> tuple[str, bool]:
    """Return the dotted module name of ``path`` and whether it is a package."""
    parts = list(path.relative_to(SRC_ROOT.parent).with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    return ".".join(parts), is_package


def _resolve_relative_import(
    module_dotted_name: str,
    is_package: bool,
    level: int,
    submodule: str | None,
) -> str:
    """Resolve a relative ``from`` import to an absolute dotted module name."""
    base = module_dotted_name if is_package else module_dotted_name.rsplit(".", 1)[0]
    base_parts = base.split(".") if base else []
    levels_up = level - 1
    if levels_up:
        base_parts = (
            base_parts[: len(base_parts) - levels_up]
            if levels_up <= len(base_parts)
            else []
        )
    base = ".".join(base_parts)
    if submodule:
        return f"{base}.{submodule}" if base else submodule
    return base


def _collect_module_level_imports(
    body: list[ast.stmt],
    module_dotted_name: str,
    is_package: bool,
) -> Iterator[tuple[int, str]]:
    """Yield the ``(lineno, dotted_module_name)`` of module-level imports in ``body``.

    Only top-level statements and statements nested in module-level ``if``/
    ``try`` blocks are visited, except the bodies of ``if TYPE_CHECKING:``
    blocks. Class bodies are visited (class-body imports count), but function
    and async-function bodies are not (function-local imports are lazy by
    design and are intentionally excluded).
    """
    for stmt in body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                yield stmt.lineno, alias.name
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.level:
                module_name = _resolve_relative_import(
                    module_dotted_name, is_package, stmt.level, stmt.module
                )
            else:
                module_name = stmt.module or ""
            yield stmt.lineno, module_name
        elif isinstance(stmt, ast.If):
            if _is_type_checking(stmt.test):
                yield from _collect_module_level_imports(
                    stmt.orelse, module_dotted_name, is_package
                )
            else:
                yield from _collect_module_level_imports(
                    stmt.body, module_dotted_name, is_package
                )
                yield from _collect_module_level_imports(
                    stmt.orelse, module_dotted_name, is_package
                )
        elif isinstance(stmt, ast.Try):
            yield from _collect_module_level_imports(
                stmt.body, module_dotted_name, is_package
            )
            for handler in stmt.handlers:
                yield from _collect_module_level_imports(
                    handler.body, module_dotted_name, is_package
                )
            yield from _collect_module_level_imports(
                stmt.orelse, module_dotted_name, is_package
            )
            yield from _collect_module_level_imports(
                stmt.finalbody, module_dotted_name, is_package
            )
        elif isinstance(stmt, ast.ClassDef):
            yield from _collect_module_level_imports(
                stmt.body, module_dotted_name, is_package
            )
        # FunctionDef and AsyncFunctionDef bodies are intentionally skipped.


def _second_segment(dotted_module_name: str) -> str:
    """Return the second dotted segment of a ``gemseo.*`` module name."""
    parts = dotted_module_name.split(".")
    return parts[1] if len(parts) > 1 else "<root>"


def _file_package(path: Path) -> str:
    """Return the top-level package of ``path``, or ``<root>`` for direct children."""
    parts = path.relative_to(SRC_ROOT).parts
    return parts[0] if len(parts) > 1 else "<root>"


def _scan_gemseo_imports() -> tuple[_Finding, ...]:
    """AST-scan every ``*.py`` file under ``src/gemseo`` for module-level imports."""
    findings = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        module_dotted_name, is_package = _module_dotted_name(path)
        file_display = path.relative_to(SRC_ROOT.parent).as_posix()
        package = _file_package(path)
        for lineno, imported_module in _collect_module_level_imports(
            tree.body, module_dotted_name, is_package
        ):
            if imported_module == "gemseo" or imported_module.startswith("gemseo."):
                segment = _second_segment(imported_module)
                findings.append(
                    _Finding(file_display, lineno, imported_module, segment, package)
                )
    return tuple(findings)


ALL_IMPORTS = _scan_gemseo_imports()


def test_core_imports_no_domain() -> None:
    """``gemseo.core`` must only depend on the shared foundations."""
    # `space` and `dataset` are a TEMPORARY exception for gemseo.core, pending
    # the dataset-inversion MR that removes this foundation-layer dependency.
    allowed_segments = {"core", "util", "space", "dataset"}
    violations = [
        (finding.file, finding.lineno, finding.module)
        for finding in ALL_IMPORTS
        if finding.package == "core" and finding.segment not in allowed_segments
    ]
    assert not violations, (
        "gemseo.core must not import domain packages, found violations "
        f"(file, lineno, imported_module): {violations}"
    )


# Domain-core SPI packages and the top-level second-segment they belong to
# (their "own domain"; for domains nested under another one, e.g.
# uncertainty.sensitivity.core, the own domain and the parent domain are the
# same second-segment: "uncertainty").
DOMAIN_CORE_PACKAGES: dict[str, str] = {
    "doe.core": "doe",
    "linear.core": "linear",
    "ode.core": "ode",
    "optimization.core": "optimization",
    "mda.core": "mda",
    "formulation.core": "formulation",
    "post.core": "post",
    "uncertainty.distribution.core": "uncertainty",
    "uncertainty.reliability.core": "uncertainty",
    "uncertainty.sensitivity.core": "uncertainty",
    "uncertainty.statistic.core": "uncertainty",
    "machine_learning.core": "machine_learning",
    "machine_learning.regression.core": "machine_learning",
    "machine_learning.classification.core": "machine_learning",
    "machine_learning.clustering.core": "machine_learning",
    "machine_learning.linear_model_fitting.core": "machine_learning",
    "machine_learning.transformer.core": "machine_learning",
    "machine_learning.resampling.core": "machine_learning",
}

# Measured, explicit exceptions: a domain-core SPI package that genuinely needs
# a sibling top-level domain. Do not widen a rule silently; add a commented,
# measured entry instead.
DOMAIN_CORE_EXCEPTIONS: dict[str, set[str]] = {
    # BaseDOELibrary exposes gemseo.optimization.result.OptimizationResult as
    # its default result class.
    "doe.core": {"optimization"},
    # Constraints.AggregationFunction reuses ConstraintAggregation from
    # gemseo.discipline.constraint_aggregation.
    "optimization.core": {"discipline"},
    # BaseMDASolverSettings reuses BaseLinearSolverSettings and LGMRES_Settings
    # from gemseo.linear to configure an MDA's inner linear solver.
    "mda.core": {"linear"},
    # BaseFormulation post-processes a scenario with ScenarioResult (from
    # gemseo.scenario.scenario_result); BaseMDOFormulation reuses Constraints
    # from gemseo.optimization.core.
    "formulation.core": {"scenario", "optimization"},
    # BasePost operates on a gemseo.optimization.problem.OptimizationProblem.
    "post.core": {"optimization"},
    # BaseSensitivityAnalysis drives DOE-based scenarios (gemseo.doe.factory,
    # gemseo.formulation.mdf_settings, gemseo.scenario.evaluation) and renders
    # its results with gemseo.post.dataset plot classes.
    "uncertainty.sensitivity.core": {"doe", "formulation", "post", "scenario"},
    # BaseResampler plots its folds with gemseo.post.dataset.scatter.
    "machine_learning.resampling.core": {"post"},
}


def test_domain_core_imports_no_sibling_domain() -> None:
    """Domain-core SPI packages must not depend on sibling top-level domains."""
    base_allowed_segments = {"core", "util", "space", "dataset"}
    violations = []
    for package, own_domain in DOMAIN_CORE_PACKAGES.items():
        allowed_segments = (
            base_allowed_segments
            | {own_domain}
            | DOMAIN_CORE_EXCEPTIONS.get(package, set())
        )
        prefix = f"gemseo/{package.replace('.', '/')}/"
        violations.extend(
            (finding.file, finding.lineno, finding.module)
            for finding in ALL_IMPORTS
            if finding.file.startswith(prefix)
            and finding.segment not in allowed_segments
        )
    assert not violations, (
        "domain core SPI packages must not import sibling top-level domains, "
        f"found violations (file, lineno, imported_module): {violations}"
    )


# Frozen per-top-level-package dependency graph: the set of second-segments
# each top-level package is allowed to import from, as measured on the
# current tree with the scanner above.
# TODO: shrink these edges in follow-up MRs.
PACKAGE_DEPENDENCY_ALLOWLIST: dict[str, frozenset[str]] = {
    "<root>": frozenset({
        "_deprecation",
        "core",
        "dataset",
        "doe",
        "machine_learning",
        "problem",
        "scenario",
        "util",
    }),
    "_deprecation": frozenset({"_deprecation", "util"}),
    "core": frozenset({"core", "dataset", "space", "util"}),
    "dataset": frozenset({"core", "dataset", "util"}),
    "discipline": frozenset({
        "core",
        "discipline",
        "machine_learning",
        "ode",
        "optimization",
        "post",
        "util",
    }),
    "doe": frozenset({"core", "doe", "optimization", "space", "util"}),
    "enum": frozenset({"util"}),
    "formulation": frozenset({
        "core",
        "discipline",
        "formulation",
        "mda",
        "optimization",
        "scenario",
        "util",
    }),
    "linear": frozenset({"core", "linear", "util"}),
    "machine_learning": frozenset({
        "<root>",
        "core",
        "dataset",
        "doe",
        "formulation",
        "machine_learning",
        "post",
        "scenario",
        "space",
        "uncertainty",
        "util",
    }),
    "mda": frozenset({"core", "discipline", "linear", "mda", "util"}),
    "ode": frozenset({"core", "ode", "util"}),
    "optimization": frozenset({
        "core",
        "dataset",
        "discipline",
        "doe",
        "optimization",
        "space",
        "util",
    }),
    "post": frozenset({
        "core",
        "dataset",
        "machine_learning",
        "optimization",
        "post",
        "util",
    }),
    "problem": frozenset({
        "<root>",
        "core",
        "dataset",
        "discipline",
        "formulation",
        "mda",
        "ode",
        "optimization",
        "post",
        "problem",
        "scenario",
        "space",
        "uncertainty",
        "util",
    }),
    "scenario": frozenset({
        "core",
        "doe",
        "formulation",
        "optimization",
        "post",
        "scenario",
        "util",
    }),
    "space": frozenset({"core", "optimization", "space", "uncertainty", "util"}),
    "uncertainty": frozenset({
        "core",
        "dataset",
        "doe",
        "formulation",
        "post",
        "scenario",
        "uncertainty",
        "util",
    }),
    "util": frozenset({
        "<root>",
        "core",
        "dataset",
        "formulation",
        "mda",
        "optimization",
        "problem",
        "scenario",
        "space",
        "util",
    }),
}


def test_package_dependency_allowlist() -> None:
    """The measured package-dependency graph must stay within the frozen allowlist."""
    measured_segments: dict[str, set[str]] = {}
    for finding in ALL_IMPORTS:
        measured_segments.setdefault(finding.package, set()).add(finding.segment)

    violations = []
    for package, segments in measured_segments.items():
        extra_segments = segments - PACKAGE_DEPENDENCY_ALLOWLIST.get(
            package, frozenset()
        )
        if extra_segments:
            violations.extend(
                (finding.file, finding.lineno, finding.module)
                for finding in ALL_IMPORTS
                if finding.package == package and finding.segment in extra_segments
            )
    assert not violations, (
        "new cross-package import edges are missing from the frozen dependency "
        f"allowlist, found violations (file, lineno, imported_module): {violations}"
    )
