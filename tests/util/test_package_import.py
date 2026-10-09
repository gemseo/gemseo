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
"""Tests for the lazy re-export helper."""

from __future__ import annotations

import ast
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

import gemseo
from gemseo.util.package_import import install_lazy_reexport
from gemseo.util.package_import import resolve_lazy_export
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from typing import Any

SOURCE_DIR = Path(gemseo.__file__).parent


@pytest.fixture
def fake_package(monkeypatch) -> ModuleType:
    """A fake package with a lazily re-exported class, installed in ``sys.modules``."""
    package = ModuleType("fake_lazy_pkg")
    package.__path__ = []
    submodule = ModuleType("fake_lazy_pkg.sub")

    class Outer:
        class Inner:
            pass

    submodule.Foo = type("Foo", (), {})
    submodule.Outer = Outer
    package.sub = submodule
    monkeypatch.setitem(sys.modules, "fake_lazy_pkg", package)
    monkeypatch.setitem(sys.modules, "fake_lazy_pkg.sub", submodule)
    install_lazy_reexport(
        package.__dict__,
        {
            "Foo": "sub",
            "Bar": "sub:Foo",
            "Inner": "sub:Outer.Inner",
            "Missing": "sub",
            "BadChain": "sub:Outer.Nope",
            "Absolute": "gemseo.util.package_import:resolve_lazy_export",
        },
        extra_all=["extra"],
    )
    return package


def test_resolve_lazy_export(fake_package) -> None:
    """Check the resolution of relative, absolute and chained locations."""
    sub = fake_package.sub
    assert resolve_lazy_export("fake_lazy_pkg", "sub", "Foo") is sub.Foo
    assert resolve_lazy_export("fake_lazy_pkg", "sub:Foo", "Bar") is sub.Foo
    assert resolve_lazy_export("fake_lazy_pkg", "sub:Outer.Inner", "x") is (
        sub.Outer.Inner
    )
    assert (
        resolve_lazy_export("x", "gemseo.util.package_import:resolve_lazy_export", "y")
        is resolve_lazy_export
    )


def test_getattr_resolves_and_caches(fake_package) -> None:
    """Check that a resolved name is cached in the namespace."""
    assert "Foo" not in fake_package.__dict__
    foo = fake_package.Foo
    assert foo is fake_package.sub.Foo
    assert fake_package.__dict__["Foo"] is foo
    assert fake_package.Bar is foo
    assert fake_package.Absolute is resolve_lazy_export
    assert fake_package.Inner is fake_package.sub.Outer.Inner


def test_all(fake_package) -> None:
    """Check that ``__all__`` includes the extra names."""
    assert fake_package.__all__ == [
        "Foo",
        "Bar",
        "Inner",
        "Missing",
        "BadChain",
        "Absolute",
        "extra",
    ]


def test_dir_without_duplicates(fake_package) -> None:
    """Check that ``__dir__`` has no duplicate, before and after caching."""
    names = dir(fake_package)
    assert len(names) == len(set(names))
    assert "Foo" in names
    fake_package.Foo  # noqa: B018
    names = dir(fake_package)
    assert len(names) == len(set(names))
    assert names == sorted(names)
    assert "Foo" in names


def test_unknown_name(fake_package, snapshot) -> None:
    """Check that an unknown name raises an AttributeError."""
    with assert_exception(AttributeError, snapshot):
        fake_package.unknown  # noqa: B018

    assert not hasattr(fake_package, "unknown")


@pytest.mark.parametrize("name", ["Missing", "BadChain"])
def test_bad_location(fake_package, name, snapshot) -> None:
    """Check that a wrong location raises an ImportError chained to its cause."""
    with assert_exception(ImportError, snapshot) as exc_info:
        getattr(fake_package, name)

    assert isinstance(exc_info.value.__cause__, AttributeError)
    assert name not in fake_package.__dict__


def test_from_import_bad_location(fake_package) -> None:
    """Check that ``from package import name`` keeps the cause of the error."""
    with pytest.raises(ImportError) as exc_info:
        from fake_lazy_pkg import Missing  # noqa: F401

    assert isinstance(exc_info.value.__cause__, AttributeError)


def _get_package_names() -> list[str]:
    """Return the names of the packages calling ``install_lazy_reexport``."""
    names = []
    for path in sorted(SOURCE_DIR.rglob("__init__.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "install_lazy_reexport"
            for node in ast.walk(tree)
        ):
            parts = path.parent.relative_to(SOURCE_DIR.parent).parts
            names.append(".".join(parts))
    return names


PACKAGE_NAMES = _get_package_names()


def _get_mapping_keys(package_name: str) -> list[str]:
    """Return the keys of ``_name_to_location`` in definition order."""
    path = SOURCE_DIR.parent.joinpath(*package_name.split("."), "__init__.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if any(getattr(t, "id", None) == "_name_to_location" for t in targets):
            mapping: Any = node.value
            # Unwrap MappingProxyType({...}).
            if isinstance(mapping, ast.Call):
                mapping = mapping.args[0]
            return [key.value for key in mapping.keys]
    msg = f"No _name_to_location in {package_name}"
    raise AssertionError(msg)


def _get_exported_type_checking_names(package_name: str) -> set[str]:
    """Return the names exported by the ``if TYPE_CHECKING:`` block of a package.

    These are the imported names whose line carries ``# noqa: F401``
    (the alias name is used when present; for a parenthesized import,
    the comment is on the alias line or on the ``from`` line)
    and the names assigned in the block (e.g. ``Name = _Class.Name``).
    The other imports only serve type hints.
    """
    path = SOURCE_DIR.parent.joinpath(*package_name.split("."), "__init__.py")
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    names = set()
    for node in ast.parse(source).body:
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            continue
        for statement in node.body:
            if isinstance(statement, ast.ImportFrom):
                for alias in statement.names:
                    # The noqa is on the alias line of a parenthesized import,
                    # or on the line of the "from" keyword.
                    if any(
                        "noqa: F401" in lines[lineno - 1]
                        for lineno in (alias.lineno, statement.lineno)
                    ):
                        names.add(alias.asname or alias.name)
            elif isinstance(statement, ast.Assign):
                names.update(t.id for t in statement.targets)
            elif isinstance(statement, ast.AnnAssign):
                names.add(statement.target.id)
    return names


def test_discovered_packages() -> None:
    """Check that the discovery of the lazily re-exporting packages works."""
    assert len(PACKAGE_NAMES) > 30
    assert "gemseo.enum" in PACKAGE_NAMES
    assert "gemseo.mda" in PACKAGE_NAMES


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_package_exports_resolve(package_name) -> None:
    """Check that every exported name resolves to the object of its location."""
    package = import_module(package_name)
    mapping = package._name_to_location
    for name, location in mapping.items():
        expected = resolve_lazy_export(package_name, location, name)
        assert getattr(package, name) is expected


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_package_mapping_is_sorted(package_name) -> None:
    """Check that the keys of the mapping are sorted by character code."""
    keys = _get_mapping_keys(package_name)
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_package_type_checking_block(package_name) -> None:
    """Check that the TYPE_CHECKING block exposes exactly the exported names."""
    keys = set(_get_mapping_keys(package_name))
    assert _get_exported_type_checking_names(package_name) == keys
