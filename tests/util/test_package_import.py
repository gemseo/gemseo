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
from gemseo._deprecation import _DeprecatedModule
from gemseo._deprecation import _install_attribute_aliases
from gemseo.util.package_import import get_location_module_name
from gemseo.util.package_import import getattr_wrapper_marker
from gemseo.util.package_import import install_lazy_reexport
from gemseo.util.package_import import resolve_lazy_export
from gemseo.util.testing.helper import assert_exception
from gemseo.util.testing.package_import import _get_static_import_names
from gemseo.util.testing.package_import import make_lazy_reexport_tests

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


@pytest.mark.parametrize("wrapped", [False, True], ids=["lazy", "wrapped"])
def test_from_import_bad_location(fake_package, wrapped, snapshot) -> None:
    """Check that ``from package import name`` raises an ImportError.

    It keeps the message and the cause, which CPython would replace by its own
    generic ImportError if the module ``__getattr__`` raised an AttributeError.

    The lazy ``__getattr__`` can be wrapped by the one of the deprecation aliases,
    which delegates the names it has no rename for to it.
    """
    if wrapped:
        _install_attribute_aliases(fake_package)
        assert fake_package.__getattr__.__module__ == "gemseo._deprecation"

    with assert_exception(ImportError, snapshot) as exc_info:
        from fake_lazy_pkg import Missing  # noqa: F401

    assert isinstance(exc_info.value.__cause__, AttributeError)


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


@pytest.mark.parametrize(
    ("name", "error"), [("Broken", ImportError), ("Typo", ModuleNotFoundError)]
)
def test_from_import_module_import_error(tmp_package, name, error, snapshot) -> None:
    """Check the error of ``from package import name`` when the module import fails.

    CPython would replace an ``AttributeError`` raised by the import of the module
    by its own generic ImportError, so it is chained to an ``ImportError``;
    the other errors propagate unchanged.
    """
    namespace: dict[str, object] = {}
    with assert_exception(error, snapshot) as exc_info:
        exec(f"from lazy_tmp_pkg import {name}", namespace)  # noqa: S102

    if error is ImportError:
        assert isinstance(exc_info.value.__cause__, AttributeError)
    else:
        assert exc_info.value.__cause__ is None


@pytest.fixture
def deprecated_module(fake_package) -> ModuleType:
    """A stand-in of a renamed module delegating to the fake package."""
    module = _DeprecatedModule("fake_old_pkg")
    module.__dict__["_deprecation_target"] = fake_package
    return module


def test_deprecated_module_bad_location(deprecated_module, snapshot) -> None:
    """Check that a stand-in keeps the error of a lazy name it cannot resolve."""
    with assert_exception(AttributeError, snapshot) as exc_info:
        deprecated_module.BadChain  # noqa: B018

    assert isinstance(exc_info.value.__cause__, AttributeError)


def test_deprecated_module_unknown_name(deprecated_module, snapshot) -> None:
    """Check that a stand-in reports an unknown name with its own name."""
    with assert_exception(AttributeError, snapshot) as exc_info:
        deprecated_module.unknown  # noqa: B018

    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("marked", [False, True], ids=["unmarked", "marked"])
def test_from_import_through_third_party_wrapper(
    fake_package, marked, snapshot
) -> None:
    """Check that a wrapper module setting the marker is seen through.

    Without the marker, the frame of the wrapper is taken for the one looking the
    name up, so CPython replaces the AttributeError by its generic ImportError.
    """
    wrapper = ModuleType("third_party_wrapper")
    if marked:
        setattr(wrapper, getattr_wrapper_marker, True)
    exec(  # noqa: S102
        "def wrap(lazy_getattr):\n"
        "    def __getattr__(name):\n"
        "        return lazy_getattr(name)\n"
        "    return __getattr__\n",
        wrapper.__dict__,
    )
    fake_package.__getattr__ = wrapper.wrap(fake_package.__getattr__)

    with assert_exception(ImportError, snapshot) as exc_info:
        from fake_lazy_pkg import Missing  # noqa: F401

    assert isinstance(exc_info.value.__cause__, AttributeError) is marked


def test_get_static_import_names(tmp_path) -> None:
    """Check the names bound by the ``if TYPE_CHECKING:`` block of a package."""
    path = tmp_path / "__init__.py"
    path.write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    import a.b\n"
        "    from c import D as E\n"
        "    F = G = 1\n"
        "    H, (I, *J) = 1, (2, 3)\n"
        "    [K] = [1]\n"
        "    L: int = 1\n"
        "    M: int\n"
        "    N.attribute = 1\n"
        "    O[0]: int = 1\n"
        "if typing.TYPE_CHECKING:\n"
        "    Q = 1\n"
        "if other:\n"
        "    R = 1\n"
        "print(P)\n",
        encoding="utf-8",
    )
    package = ModuleType("package")
    package.__file__ = str(path)

    static_names, referenced_names = _get_static_import_names(package)

    assert static_names == set("aEFGHIJKLMQ")
    assert referenced_names == {
        "TYPE_CHECKING",
        "int",
        "N",
        "O",
        "typing",
        "other",
        "print",
        "P",
    }


def test_reexport_is_lazy_detects_an_eager_binding(
    tmp_path, monkeypatch, request
) -> None:
    """Check that a name bound eagerly by the package fails the laziness test."""
    package_dir = tmp_path / "eager_tmp_pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        "from gemseo.util.package_import import install_lazy_reexport\n"
        "_name_to_location = {'Foo': 'sub'}\n"
        "install_lazy_reexport(globals(), _name_to_location)\n"
        "Foo = 1\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(tmp_path)
    # A monkeypatch of `sys.modules` would restore the imported package at teardown,
    # as the entry is absent before the import and only created by it.
    request.addfinalizer(lambda: sys.modules.pop("eager_tmp_pkg", None))
    package = import_module("eager_tmp_pkg")

    test_reexport_is_lazy = make_lazy_reexport_tests(package)["test_reexport_is_lazy"]

    with pytest.raises(AssertionError):
        test_reexport_is_lazy()


def test_get_location_module_name() -> None:
    """Check the full name of the module of absolute and relative locations."""
    assert get_location_module_name("pkg", "sub.module:A.b") == "pkg.sub.module"
    assert get_location_module_name("pkg", "gemseo.module") == "gemseo.module"


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
