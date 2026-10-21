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
"""Factory of tests for the lazy re-export of names from a package.

Counterpart to the runtime helper
[install_lazy_reexport][gemseo.util.package_import.install_lazy_reexport]:
the mapping under test is read from the package's own ``_name_to_location``
and each expected object is resolved with the same shared
[resolve_lazy_export][gemseo.util.package_import.resolve_lazy_export].
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Final

import pytest

from gemseo.util.package_import import get_location_module_name
from gemseo.util.package_import import resolve_lazy_export
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType

_subprocess_timeout: Final[float] = 120.0
"""The time limit, in seconds, of the fresh interpreter of a test."""


def _is_type_checking_guard(test: ast.expr) -> bool:
    """Return whether an `if` test is (an alias of) `typing.TYPE_CHECKING`.

    Args:
        test: The test expression of the `if` statement.

    Returns:
        Whether the test refers to `TYPE_CHECKING`.
    """
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _get_target_names(target: ast.expr) -> list[str]:
    """Return the names bound by the target of an assignment.

    Args:
        target: The target of the assignment.

    Returns:
        The names bound by the target, walking tuple and list targets recursively;
        none for an attribute or a subscript target.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Starred):
        return _get_target_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for elt in target.elts for name in _get_target_names(elt)]
    return []


def _get_static_import_names(package: ModuleType) -> tuple[set[str], set[str]]:
    """Return the names bound under `if TYPE_CHECKING:` in a package.

    Args:
        package: The package whose `__init__.py` is scanned.

    Returns:
        The names bound by the module-level `if TYPE_CHECKING:` block,
        by an import or an assignment,
        and the names read anywhere in the module
        (an imported name that is read is an annotation import,
        not the static mirror of a lazy re-export).
    """
    # A regular package always has a file; only namespace packages do not.
    assert package.__file__ is not None
    tree = ast.parse(Path(package.__file__).read_text(encoding="utf-8"))
    static_names: set[str] = set()
    for stmt in tree.body:
        if not (isinstance(stmt, ast.If) and _is_type_checking_guard(stmt.test)):
            continue
        for sub_stmt in stmt.body:
            if isinstance(sub_stmt, (ast.Import, ast.ImportFrom)):
                static_names.update(
                    alias.asname or alias.name.split(".")[0] for alias in sub_stmt.names
                )
            elif isinstance(sub_stmt, ast.Assign):
                for target in sub_stmt.targets:
                    static_names.update(_get_target_names(target))
            elif isinstance(sub_stmt, ast.AnnAssign):
                static_names.update(_get_target_names(sub_stmt.target))
    referenced_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return static_names, referenced_names


def make_lazy_reexport_tests(
    package: ModuleType,
    extra_all: Iterable[str] = (),
    deferred_sample: str | None = None,
) -> dict[str, object]:
    """Build the standard lazy-re-export test functions for a package.

    The mapping under test is read from ``package._name_to_location``.

    Args:
        package: The imported package exposing names lazily via
            [install_lazy_reexport][gemseo.util.package_import.install_lazy_reexport].
        extra_all: Names exposed by `__all__` that are not lazily re-exported
            (e.g. eagerly imported names).
        deferred_sample: The name of a lazily re-exported entry used to assert, in a
            fresh interpreter, that its defining module stays out of `sys.modules`
            until the name is first accessed. If `None`, this stronger check,
            the only one starting an interpreter, is skipped.

    Returns:
        A mapping from test function name to test function,
        meant to be spread into a test module via `globals().update(...)`.
    """
    name_to_location: dict[str, str] = package._name_to_location
    names = list(name_to_location)
    extra_all = tuple(extra_all)

    @pytest.mark.parametrize(("name", "location"), name_to_location.items())
    def test_lazy_reexport(name, location) -> None:
        """Check that a name is re-exported from the package."""
        expected = resolve_lazy_export(package.__name__, location, name)
        assert getattr(package, name) is expected

    def test_all() -> None:
        """Check that `__all__` exposes exactly the expected names."""
        assert set(package.__all__) == set(names) | set(extra_all)

    def test_mapping_is_sorted() -> None:
        """Check that the names of the mapping are sorted."""
        assert names == sorted(names)

    def test_reexport_is_lazy() -> None:
        """Check that the names are served lazily, not eagerly bound.

        The `__init__.py` of the package is executed again in a fresh module object,
        as the resolved names are cached in the namespace of the package
        by the first access; so are its module-level side effects, which must be
        idempotent.
        """
        spec = package.__spec__
        if spec is None or spec.loader is None:
            pytest.fail(
                f"the package {package.__name__!r} has no spec or loader "
                "to execute it again"
            )
        fresh_package = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh_package)
        assert not [name for name in names if name in vars(fresh_package)]

    if deferred_sample is not None:
        module_name = get_location_module_name(
            package.__name__, name_to_location[deferred_sample]
        )

        def test_deferred_module_import() -> None:
            """Check that the module of `deferred_sample` is imported on first access.

            Run in a fresh interpreter, so that probing `sys.modules` does not depend
            on and mutate the import state of the current test session.
            """
            code = (
                "import sys\n"
                "from importlib import import_module\n"
                f"package = import_module({package.__name__!r})\n"
                f"assert {module_name!r} not in sys.modules\n"
                f"getattr(package, {deferred_sample!r})\n"
                f"assert {module_name!r} in sys.modules\n"
            )
            result = subprocess.run(  # noqa: S603
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=_subprocess_timeout,
            )
            assert result.returncode == 0, result.stderr

    def test_unknown_attribute(snapshot) -> None:
        """Check that accessing an unknown attribute raises `AttributeError`."""
        with assert_exception(AttributeError, snapshot):
            package.NotAClass  # noqa: B018

    @pytest.mark.parametrize("name", names)
    def test_dir(name) -> None:
        """Check that a name is exposed by `dir`."""
        assert name in dir(package)

    def test_static_names_match_lazy_reexports() -> None:
        """Check that the `if TYPE_CHECKING:` block mirrors the lazy re-exports.

        The `if TYPE_CHECKING:` block of the package gives mypy and IDEs static
        visibility of the names served at runtime by `_name_to_location`; the two
        must not drift apart.
        """
        static_names, referenced_names = _get_static_import_names(package)
        missing = set(names) - static_names
        assert not missing, (
            f"lazily re-exported names missing from the TYPE_CHECKING imports of "
            f"{package.__name__!r}: {sorted(missing)}"
        )
        unknown = static_names - set(names) - set(extra_all) - referenced_names
        assert not unknown, (
            f"TYPE_CHECKING imports of {package.__name__!r} not matching any "
            f"exported name: {sorted(unknown)}"
        )

    tests: dict[str, object] = {
        "test_lazy_reexport": test_lazy_reexport,
        "test_all": test_all,
        "test_mapping_is_sorted": test_mapping_is_sorted,
        "test_reexport_is_lazy": test_reexport_is_lazy,
        "test_unknown_attribute": test_unknown_attribute,
        "test_dir": test_dir,
        "test_static_names_match_lazy_reexports": (
            test_static_names_match_lazy_reexports
        ),
    }
    if deferred_sample is not None:
        tests["test_deferred_module_import"] = test_deferred_module_import
    return tests
