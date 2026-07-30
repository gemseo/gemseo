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
the mapping under test is read from the package's own ``_NAME_TO_LOCATION``
and each expected object is resolved with the same shared
[resolve_lazy_export][gemseo.util.package_import.resolve_lazy_export].
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gemseo.util.package_import import resolve_lazy_export
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType


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


def _get_static_import_names(package: ModuleType) -> tuple[set[str], set[str]]:
    """Return the names imported under `if TYPE_CHECKING:` in a package.

    Args:
        package: The package whose `__init__.py` is scanned.

    Returns:
        The names bound by the module-level `if TYPE_CHECKING:` imports,
        and the names referenced anywhere in the module
        (an imported name that is referenced is an annotation import,
        not the static mirror of a lazy re-export).
    """
    # A regular package always has a file; only namespace packages do not.
    assert package.__file__ is not None
    tree = ast.parse(Path(package.__file__).read_text(encoding="utf-8"))
    imported_names: set[str] = set()
    for stmt in tree.body:
        if not (isinstance(stmt, ast.If) and _is_type_checking_guard(stmt.test)):
            continue
        for sub_stmt in stmt.body:
            if isinstance(sub_stmt, (ast.Import, ast.ImportFrom)):
                imported_names.update(
                    alias.asname or alias.name.split(".")[0] for alias in sub_stmt.names
                )
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    return imported_names, referenced_names


def make_lazy_reexport_tests(
    package: ModuleType,
    extra_all: Iterable[str] = (),
    deferred_sample: str | None = None,
) -> dict[str, object]:
    """Build the standard lazy-re-export test functions for a package.

    The mapping under test is read from ``package._NAME_TO_LOCATION``.

    Args:
        package: The imported package exposing names lazily via
            [install_lazy_reexport][gemseo.util.package_import.install_lazy_reexport].
        extra_all: Names exposed by `__all__` that are not lazily re-exported
            (e.g. eagerly imported names).
        deferred_sample: The name of a lazily re-exported entry used to assert, in a
            fresh interpreter, that its defining module stays out of `sys.modules`
            until the name is first accessed. If `None`, this stronger check is skipped.

    Returns:
        A mapping from test function name to test function,
        meant to be spread into a test module via `globals().update(...)`.
    """
    name_to_location: dict[str, str] = package._NAME_TO_LOCATION
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

    def test_reexport_is_lazy() -> None:
        """Check that the names are served lazily, not eagerly bound."""
        namespace = vars(package)
        for name in names:
            assert name not in namespace

    def test_unknown_attribute(snapshot) -> None:
        """Check that accessing an unknown attribute raises `AttributeError`."""
        with assert_exception(AttributeError, snapshot):
            package.NotAClass  # noqa: B018

    @pytest.mark.parametrize("name", names)
    def test_dir(name) -> None:
        """Check that a name is exposed by `dir`."""
        assert name in dir(package)

    def test_static_names_match_lazy_reexports() -> None:
        """Check that the `if TYPE_CHECKING:` imports mirror the lazy re-exports.

        The `if TYPE_CHECKING:` block of the package gives mypy and IDEs static
        visibility of the names served at runtime by `_NAME_TO_LOCATION`; the two
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

    tests = {
        "test_lazy_reexport": test_lazy_reexport,
        "test_all": test_all,
        "test_reexport_is_lazy": test_reexport_is_lazy,
        "test_unknown_attribute": test_unknown_attribute,
        "test_dir": test_dir,
        "test_static_names_match_lazy_reexports": (
            test_static_names_match_lazy_reexports
        ),
    }

    if deferred_sample is not None:
        name = deferred_sample
        module_path = name_to_location[name].partition(":")[0]
        if module_path.split(".", 1)[0] == "gemseo":
            full_module = module_path
        else:
            full_module = f"{package.__name__}.{module_path}"

        def test_reexport_defers_import() -> None:
            """Check that the module is imported only on first attribute access.

            Run in a fresh interpreter so that probing `sys.modules` does not
            mutate the import state of the current test session.
            """
            code = (
                "import sys\n"
                "from importlib import import_module\n"
                f"package = import_module({package.__name__!r})\n"
                f"assert {full_module!r} not in sys.modules\n"
                f"getattr(package, {name!r})\n"
                f"assert {full_module!r} in sys.modules\n"
            )
            result = subprocess.run(  # noqa: S603
                [sys.executable, "-c", code], capture_output=True, text=True
            )
            assert result.returncode == 0, result.stderr

        tests["test_reexport_defers_import"] = test_reexport_defers_import

    return tests
