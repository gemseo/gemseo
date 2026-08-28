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
"""Backward compatibility for modules and classes renamed or moved across releases.

Many modules, packages and classes have been renamed or moved (see the `modules:` and
`attributes:` sections of `bump-version.yml` in this package, and the tables computed
from them in [aliases][gemseo._deprecation.aliases]). To keep scripts written against a
previous release working for one deprecation cycle,
[install][gemseo._deprecation.install] registers a [importlib.abc.MetaPathFinder][] that
intercepts imports of the old names, redirects them to the new location, applies any
class rename, and emits a `DeprecationWarning` pointing at the new name.

When only a class was renamed and its module kept its own name, there is no import to
intercept; [install][gemseo._deprecation.install] then adds a module-level `__getattr__`
resolving the old attribute names in that module's own namespace.

`gemseo/__init__.py` calls [install][gemseo._deprecation.install] on import, so
`from gemseo.old.path import OldName` keeps working even though the old module no longer
exists on disk.

A stand-in delegates reads to its new location but owns its own namespace: binding an
attribute through an old path, e.g. in a test monkey-patching
`gemseo.util.string_tools.MultiLineString`, only rebinds it on the stand-in and is not
seen by the code importing the new path. Patch the new path instead.
"""

from __future__ import annotations

import sys
import warnings
from importlib import import_module
from importlib.abc import Loader
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import find_spec as _find_spec
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Final

from gemseo._deprecation.aliases import ATTRIBUTE_RENAMES
from gemseo._deprecation.aliases import DISSOLVED_PACKAGES
from gemseo._deprecation.aliases import LIVE_ALIASED_MODULES
from gemseo._deprecation.aliases import MODULE_RENAMES
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

_PACKAGE_PREFIX: Final[str] = "gemseo."

_installed: bool = False


def _resolve_new_name(old_name: str) -> str | None:
    """Return the new module name for an old one, or `None` if it is not renamed.

    The [MODULE_RENAMES][gemseo._deprecation.aliases.MODULE_RENAMES] entry whose old
    name is the longest prefix of `old_name` is used, so a specific module rename wins
    over a broader package rename.

    Args:
        old_name: The old fully-qualified module name.

    Returns:
        The new fully-qualified module name, or `None` if `old_name` is not renamed.
    """
    new_name = old_name
    seen: set[str] = set()
    while new_name not in seen:
        seen.add(new_name)
        best_old = None
        for old_prefix in MODULE_RENAMES:
            if (new_name == old_prefix or new_name.startswith(f"{old_prefix}.")) and (
                best_old is None or len(old_prefix) > len(best_old)
            ):
                best_old = old_prefix
        if best_old is None:
            break
        new_name = MODULE_RENAMES[best_old] + new_name[len(best_old) :]
    return None if new_name == old_name else new_name


def _warn_attribute_rename(module_name: str, name: str, new_name: str) -> None:
    """Warn that a module-level attribute was renamed.

    Args:
        module_name: The old fully-qualified name of the module defining the attribute.
        name: The old attribute name.
        new_name: The fully-qualified new name to point the user at.
    """
    warnings.warn(
        f"The attribute {name!r} of the module {module_name!r} is deprecated; "
        f"use {new_name!r} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _qualify(holder_name: str, new_name: str) -> str:
    """Return the fully-qualified new name of a renamed attribute.

    Args:
        holder_name: The fully-qualified name of the module holding the new name.
        new_name: The new name, already fully qualified when the attribute moved to
            another module.

    Returns:
        The fully-qualified new name.
    """
    if new_name.startswith(_PACKAGE_PREFIX):
        return new_name
    return f"{holder_name}.{new_name}"


def _get_renamed_attribute(module: ModuleType, new_name: str) -> Any:
    """Return the object a rename points to.

    Args:
        module: The module holding the new name, unless the latter is fully qualified.
        new_name: The new name, fully qualified when the attribute moved to another
            module.

    Returns:
        The renamed object.

    Raises:
        AttributeError: When the new name does not exist.
    """
    if not new_name.startswith(_PACKAGE_PREFIX):
        return getattr(module, new_name)
    new_module_name, _, attribute_name = new_name.rpartition(".")
    return getattr(import_module(new_module_name), attribute_name)


class _DeprecatedModule(ModuleType):
    """A stand-in for a renamed module that delegates to its new location."""

    def __getattr__(self, name: str) -> Any:
        # Reached only when normal lookup on this module's namespace fails.
        target = self.__dict__["_deprecation_target"]
        new_name = ATTRIBUTE_RENAMES.get(self.__name__, {}).get(name)
        if new_name is None:
            new_name = name
        else:
            # The module warning only names the new module, which does not carry the
            # old attribute name; point at the renamed attribute itself.
            _warn_attribute_rename(
                self.__name__, name, _qualify(target.__name__, new_name)
            )
        try:
            return _get_renamed_attribute(target, new_name)
        except AttributeError:
            msg = f"module {self.__name__!r} has no attribute {name!r}"
            raise AttributeError(msg) from None

    def __dir__(self) -> list[str]:
        # Expose the new location's names, as the stand-in namespace is empty.
        return [*self.__dict__, *dir(self.__dict__["_deprecation_target"])]


class _DeprecatedModuleLoader(Loader):
    """Load a renamed module from its new location and warn about the move."""

    def __init__(self, old_name: str, new_name: str) -> None:
        """
        Args:
            old_name: The deprecated fully-qualified module name being imported.
            new_name: The new fully-qualified module name to redirect to.
        """  # noqa: D205 D212
        self._old_name = old_name
        self._new_name = new_name

    def create_module(self, spec: ModuleSpec) -> ModuleType:  # noqa: D102
        target = import_module(self._new_name)
        module = _DeprecatedModule(self._old_name)
        module.__doc__ = target.__doc__
        module.__dict__["_deprecation_target"] = target
        # Keep the old package path importable so its submodules also redirect.
        target_path = getattr(target, "__path__", None)
        if target_path is not None:
            module.__path__ = target_path
        # Keep star imports from the old path working: the stand-in namespace is empty,
        # so without `__all__` the star import would bind nothing at all.
        target_all = getattr(target, "__all__", None)
        if target_all is None:
            target_all = [name for name in vars(target) if not name.startswith("_")]
        module.__dict__["__all__"] = list(target_all)
        return module

    def exec_module(self, module: ModuleType) -> None:  # noqa: D102
        warnings.warn(
            f"The module {self._old_name!r} is deprecated; "
            f"use {self._new_name!r} instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class _DissolvedPackage(ModuleType):
    """A stand-in for a package whose submodules were dissolved into others."""

    def __getattr__(self, name: str) -> Any:
        # Reached only when normal lookup on this module's namespace fails.
        if name == "__all__":
            # Computed on demand so that a star import from the old package binds the
            # names it used to, without importing every new location upfront.
            return self.__dir__()
        for new_name in self.__dict__["_deprecation_targets"]:
            try:
                return getattr(import_module(new_name), name)
            except AttributeError:  # noqa: PERF203
                continue
        msg = f"module {self.__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    def __dir__(self) -> list[str]:
        # Expose the new locations' names, as the stand-in namespace is empty.
        names: set[str] = set()
        for new_name in self.__dict__["_deprecation_targets"]:
            names.update(
                name
                for name in dir(import_module(new_name))
                if not name.startswith("_")
            )
        return sorted(names)


class _DissolvedPackageLoader(Loader):
    """Load a dissolved package as a stand-in module and warn about the dissolution."""

    def __init__(self, old_name: str, new_names: tuple[str, ...]) -> None:
        """
        Args:
            old_name: The deprecated fully-qualified package name being imported.
            new_names: The fully-qualified names of the packages that the old
                package's submodules were dissolved into, in resolution order.
        """  # noqa: D205 D212
        self._old_name = old_name
        self._new_names = new_names

    def create_module(self, spec: ModuleSpec) -> ModuleType:  # noqa: D102
        module = _DissolvedPackage(self._old_name)
        module.__dict__["_deprecation_targets"] = self._new_names
        # Keep the old package importable as a package; its known submodules are
        # intercepted by name by the meta-path finder, unknown ones fail normally.
        module.__path__ = []
        return module

    def exec_module(self, module: ModuleType) -> None:  # noqa: D102
        warnings.warn(
            f"The module {self._old_name!r} is deprecated; "
            f"import its contents from the new locations instead: "
            f"{pretty_repr(self._new_names, sort=False)}.",
            DeprecationWarning,
            stacklevel=2,
        )


def _install_attribute_aliases(module: ModuleType) -> None:
    """Make the old names of renamed attributes resolve on a module that still exists.

    A module-level `__getattr__` is installed in the module namespace; it is reached
    only when the normal lookup fails, so it never shadows a live name. Names the
    module has no rename for are delegated to any `__getattr__` it already defines
    (e.g. the lazy re-export of a package), so that the errors raised by the latter
    are not mistaken for an unknown name.

    Args:
        module: The module whose old attribute names must keep working.
    """
    module_name = module.__name__
    renames = ATTRIBUTE_RENAMES[module_name]
    previous_getattr = module.__dict__.get("__getattr__")

    def __getattr__(name: str) -> Any:  # noqa: N807
        new_name = renames.get(name)
        if new_name is None:
            if previous_getattr is not None:
                return previous_getattr(name)
            msg = f"module {module_name!r} has no attribute {name!r}"
            raise AttributeError(msg)
        _warn_attribute_rename(module_name, name, _qualify(module_name, new_name))
        return _get_renamed_attribute(module, new_name)

    module.__dict__["__getattr__"] = __getattr__


class _AttributeAliasLoader(Loader):
    """Load a module normally, then alias the old names of its renamed attributes."""

    def __init__(self, loader: Loader) -> None:
        """
        Args:
            loader: The loader that actually loads the module.
        """  # noqa: D205 D212
        self._loader = loader

    def __getattr__(self, name: str) -> Any:
        # Delegate the rest of the loader protocol (get_source, get_filename, ...) to
        # the wrapped loader; reached only when normal lookup on this object fails.
        return getattr(self.__dict__["_loader"], name)

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:  # noqa: D102
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:  # noqa: D102
        self._loader.exec_module(module)
        _install_attribute_aliases(module)


class _DeprecatedModuleFinder(MetaPathFinder):
    """Redirect imports of renamed `gemseo` modules to their new location."""

    _finding: set[str]
    """The modules whose real spec is being looked up, to avoid re-entering."""

    def __init__(self) -> None:  # noqa: D107
        self._finding = set()

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:  # noqa: D102
        if not fullname.startswith(_PACKAGE_PREFIX):
            return None
        if fullname in DISSOLVED_PACKAGES:
            return ModuleSpec(
                fullname,
                _DissolvedPackageLoader(fullname, DISSOLVED_PACKAGES[fullname]),
                is_package=True,
            )
        new_name = _resolve_new_name(fullname)
        if new_name is not None:
            return ModuleSpec(fullname, _DeprecatedModuleLoader(fullname, new_name))
        return self._find_alias_spec(fullname)

    def _find_alias_spec(self, fullname: str) -> ModuleSpec | None:
        """Return the spec of a live module whose attributes were renamed.

        The module is loaded by its real loader, wrapped so that the old attribute
        names keep resolving.

        Args:
            fullname: The fully-qualified name of the module being imported.

        Returns:
            The wrapped spec, or `None` when the module has no renamed attribute.
        """
        if fullname not in LIVE_ALIASED_MODULES or fullname in self._finding:
            return None
        self._finding.add(fullname)
        try:
            spec = _find_spec(fullname)
        finally:
            self._finding.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _AttributeAliasLoader(spec.loader)
        return spec


def install() -> None:
    """Register the deprecated-import finder.

    Also alias the old names of the renamed attributes of the modules that kept their
    own name and are already imported, and register warning filters so the emitted
    `DeprecationWarning` is shown once, regardless of the default filters: without them,
    warnings triggered by imports in library code (rather than `__main__`) would be
    silenced. These filters take precedence over the default ones, so they are only
    registered when the user did not configure warnings themselves, e.g. with
    `-W error::DeprecationWarning` or `PYTHONWARNINGS`; their choice must win.

    Idempotent: calling it more than once has no effect.
    """
    global _installed
    if _installed:
        return
    if not sys.warnoptions:
        warnings.filterwarnings(
            "once",
            message=r"The module 'gemseo\..*' is deprecated",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "once",
            message=r"The attribute '.*' of the module 'gemseo\..*' is deprecated",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "once",
            message=r"The class 'gemseo\..*' is deprecated",
            category=DeprecationWarning,
        )
    sys.meta_path.insert(0, _DeprecatedModuleFinder())
    for module_name in LIVE_ALIASED_MODULES:
        # The finder only sees the modules imported from now on; the ones already
        # imported (by gemseo itself) are aliased here.
        module = sys.modules.get(module_name)
        if module is not None:
            _install_attribute_aliases(module)
    _installed = True
