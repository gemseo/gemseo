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
    from collections.abc import Iterator

SOURCE_DIR = Path(gemseo.__file__).parent
TESTS_DIR = Path(__file__).parents[1]


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
    assert "Foo" in names


def test_unknown_name(fake_package, snapshot) -> None:
    """Check that an unknown name raises an AttributeError."""
    with assert_exception(AttributeError, snapshot):
        fake_package.unknown  # noqa: B018

    assert not hasattr(fake_package, "unknown")


@pytest.mark.parametrize("name", ["Missing", "BadChain"])
def test_bad_location(fake_package, name, snapshot) -> None:
    """Check that a wrong attribute raises an AttributeError chained to its cause."""
    with assert_exception(AttributeError, snapshot) as exc_info:
        getattr(fake_package, name)

    assert isinstance(exc_info.value.__cause__, AttributeError)
    assert name not in fake_package.__dict__
    assert not hasattr(fake_package, name)
    assert getattr(fake_package, name, None) is None


def test_from_import_bad_location(fake_package) -> None:
    """Check that ``from package import name`` raises an ImportError.

    CPython turns the AttributeError of the module ``__getattr__`` into its own
    ImportError, without the cause.
    """
    with pytest.raises(ImportError, match="cannot import name 'Missing'"):
        from fake_lazy_pkg import Missing  # noqa: F401


@pytest.fixture
def tmp_package(tmp_path, monkeypatch) -> Iterator[ModuleType]:
    """A package on disk re-exporting names from submodules that cannot be imported."""
    package_dir = tmp_path / "lazy_tmp_pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        "from gemseo.util.package_import import install_lazy_reexport\n"
        "install_lazy_reexport(globals(), {'Broken': 'broken', 'Typo': 'typo'})\n",
        encoding="utf-8",
    )
    (package_dir / "broken.py").write_text(
        "raise AttributeError('Raised while importing broken.')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(tmp_path)
    yield import_module("lazy_tmp_pkg")
    for name in [name for name in sys.modules if name.startswith("lazy_tmp_pkg")]:
        del sys.modules[name]


@pytest.mark.parametrize(
    ("name", "error"), [("Broken", AttributeError), ("Typo", ModuleNotFoundError)]
)
def test_module_import_error_propagates(tmp_package, name, error, snapshot) -> None:
    """Check that an error raised by the import of the module is not wrapped."""
    with assert_exception(error, snapshot) as exc_info:
        getattr(tmp_package, name)

    assert exc_info.value.__cause__ is None


def _get_package_names() -> list[str]:
    """Return the names of the packages calling ``install_lazy_reexport``.

    Returns:
        The names of the packages.
    """
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


def _get_tested_package_names() -> set[str]:
    """Return the names of the packages passed to ``make_lazy_reexport_tests``.

    Returns:
        The names of the packages, read from the first argument of the calls,
        written as a dotted name like ``gemseo.mda``, in the test modules.
    """
    names = set()
    for path in TESTS_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "make_lazy_reexport_tests"
                and node.args
            ):
                names.add(ast.unparse(node.args[0]))
    return names


def test_discovered_packages() -> None:
    """Check that the discovery of the lazily re-exporting packages works."""
    package_names = _get_package_names()
    assert len(package_names) > 30
    assert "gemseo.enum" in package_names
    assert "gemseo.mda" in package_names


def test_packages_are_tested() -> None:
    """Check that every lazily re-exporting package has its factory-made tests."""
    untested = set(_get_package_names()) - _get_tested_package_names()
    assert not untested, sorted(untested)
