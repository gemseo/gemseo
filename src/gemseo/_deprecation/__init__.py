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

The names whose `attributes:` value is `TODO <text>` are the exception: their migration
cannot be automated, as the new name does not behave as the old one on its own, so
importing one raises an `ImportError` ending with the text, which tells how to migrate,
instead of silently resolving to a name of a different meaning.

Renamed class attributes (see the `classes:` section of `bump-version.yml` and
[class_attribute_renames][gemseo._deprecation.aliases.class_attribute_renames]) are
aliased too: [install][gemseo._deprecation.install] sets a data descriptor for each
old attribute name on the class defining it, so both `Cls.OLD` and `instance.OLD`
keep working with a `DeprecationWarning`, and a subclass body that still assigns the
old name (e.g. `OLD = value` in the class statement) is remapped to the new name. The
old name of an enumeration member is also resolved by the lookup by name, e.g.
`Enum["OLD"]`, but is not listed among the members. A dotted entry, e.g.
`DifferentiationMethod.USER_GRAD`, aliases the attribute of a class nested in the
class, e.g. the member of an enumeration.
The one path this does not cover is an assignment on the class object itself after
it is defined, e.g. `Cls.OLD = value` or `monkeypatch.setattr(Cls, "OLD", ...)`: this
silently shadows the descriptor instead of going through it, so patch the new name
instead.

An old package that was dissolved, i.e. that has no rename of its own but whose
submodules were renamed, is derived from the `modules:` section (see
[dissolved_packages][gemseo._deprecation.aliases.dissolved_packages]). Importing it
gives an empty stand-in that warns and lists the new locations of its former
submodules; it resolves no name of its own, as the old package defined none, while its
submodules are redirected as any renamed module.

`gemseo/__init__.py` calls [install][gemseo._deprecation.install] on import, so
`from gemseo.old.path import OldName` keeps working even though the old module no longer
exists on disk.

A stand-in delegates reads to its new location but owns its own namespace: binding an
attribute through an old path, e.g. in a test monkey-patching
`gemseo.util.string_tools.MultiLineString`, only rebinds it on the stand-in and is not
seen by the code importing the new path. Patch the new path instead.
"""

from __future__ import annotations

import dis
import sys
import warnings
from enum import Enum
from enum import EnumMeta
from importlib import import_module
from importlib.abc import Loader
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import find_spec as _find_spec
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Final
from typing import NoReturn

from gemseo._deprecation.aliases import attribute_renames
from gemseo._deprecation.aliases import class_attribute_renames
from gemseo._deprecation.aliases import dissolved_packages
from gemseo._deprecation.aliases import find_ancestor
from gemseo._deprecation.aliases import live_aliased_modules
from gemseo._deprecation.aliases import manual_migrations
from gemseo._deprecation.aliases import module_renames
from gemseo._deprecation.aliases import removed_attributes
from gemseo._deprecation.aliases import removed_modules
from gemseo._deprecation.aliases import unmigrated_attributes
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any

_package_prefix: Final[str] = "gemseo."

_plugin_prefix: Final[str] = "gemseo_"
"""The prefix of the name of the package of a GEMSEO plugin, e.g. `gemseo_excel`."""

_import_from_opcode: Final[int] = dis.opmap["IMPORT_FROM"]
"""The opcode importing a name in a `from module import name` statement."""

_installed: bool = False


def _resolve_new_name(old_name: str) -> str | None:
    """Return the new module name for an old one, or `None` if it is not renamed.

    The [module_renames][gemseo._deprecation.aliases.module_renames] entry whose old
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
        for old_prefix in module_renames:
            if (new_name == old_prefix or new_name.startswith(f"{old_prefix}.")) and (
                best_old is None or len(old_prefix) > len(best_old)
            ):
                best_old = old_prefix
        if best_old is None:
            break
        new_name = module_renames[best_old] + new_name[len(best_old) :]
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


def _is_imported_from() -> bool:
    """Return whether the attribute being looked up is imported by a `from` statement.

    The frame looking the attribute up is the first one outside this module, the
    frames of this module being those of the `__getattr__` of a module and of its
    helpers. It imports the attribute when it is executing the `IMPORT_FROM`
    instruction of a `from module import name` statement.

    Returns:
        Whether the attribute is imported by a `from module import name` statement.
    """
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
    return (
        frame is not None and frame.f_code.co_code[frame.f_lasti] == _import_from_opcode
    )


def _raise_unresolvable_attribute(msg: str) -> NoReturn:
    """Raise the error of a module-level attribute that cannot be resolved.

    An `AttributeError` is raised, so that `hasattr` and `getattr` with a default keep
    working, as they only swallow an `AttributeError`, except under a
    `from module import name` statement, which would replace an `AttributeError` by a
    generic "cannot import name" `ImportError`, losing the message: an `ImportError` is
    raised there instead.

    Args:
        msg: The error message.

    Raises:
        ImportError: Under a `from module import name` statement.
        AttributeError: Otherwise.
    """
    if _is_imported_from():
        raise ImportError(msg)
    raise AttributeError(msg)


def _raise_for_unmigrated_attribute(module_name: str, name: str) -> None:
    """Raise when a module-level attribute raises instead of resolving.

    Such an attribute is either removed with no replacement or one whose migration
    cannot be automated.

    Args:
        module_name: The old fully-qualified name of the module defining the attribute.
        name: The old attribute name.

    Raises:
        ImportError: When the migration of the attribute cannot be automated (a
            `TODO <text>` entry of the `attributes:` section), whatever the way it is
            looked up, the message ending with the text; when it was removed with no
            replacement (a `null` entry of the `attributes:` section), under a
            `from module import name` statement.
        AttributeError: When the attribute was removed with no replacement, otherwise
            (see
            [_raise_unresolvable_attribute][gemseo._deprecation._raise_unresolvable_attribute]).
    """
    migration = manual_migrations.get(module_name, {}).get(name)
    if migration is not None:
        msg = (
            f"The attribute {name!r} of the module {module_name!r} was removed; "
            f"{migration}."
        )
        raise ImportError(msg)
    if name in removed_attributes.get(module_name, ()):
        msg = (
            f"The attribute {name!r} of the module {module_name!r} was removed, "
            "with no replacement."
        )
        _raise_unresolvable_attribute(msg)


def _raise_for_removed_module(name: str) -> None:
    """Raise when a module was removed with no replacement.

    It is called by the finder, so `importlib.util.find_spec` also raises for such a
    module instead of returning `None`, as it does for a module whose parent package
    is missing; returning `None` would lose the message of an actual import.

    The error carries no module `name`: the import machinery silently swallows the
    `ModuleNotFoundError` of a `from package import module` whose `name` is that of
    the module, so that the message would be replaced by a generic "cannot import
    name" one.

    Args:
        name: The fully-qualified name of the module being imported.

    Raises:
        ModuleNotFoundError: When the module, or a package containing it, is a `null`
            entry of the `modules:` section. Without it, the rename of an ancestor
            package would redirect the import to a module that does not exist either,
            with a message naming that module instead.
    """
    removed = find_ancestor(name, removed_modules)
    if removed is not None:
        msg = f"The module {removed!r} was removed, with no replacement."
        raise ModuleNotFoundError(msg)


def _warn_class_attribute_rename(
    class_name: str, name: str, new_name: str, stacklevel: int = 3
) -> None:
    """Warn that a class attribute was renamed.

    Args:
        class_name: The name of the class defining the attribute.
        name: The old attribute name.
        new_name: The new attribute name.
        stacklevel: The stack level of the warning, counted from this function.
    """
    warnings.warn(
        f"The attribute {name!r} of the class {class_name!r} is deprecated; "
        f"use {new_name!r} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def _get_missing_plugin_message(
    old: str, new_name: str, error: ModuleNotFoundError
) -> str | None:
    """Return the message of a name moved to a plugin that is not installed.

    Args:
        old: The description of the old name, e.g. `"The module 'gemseo.x'"`.
        new_name: The fully-qualified new name.
        error: The error raised when importing the module of the new name.

    Returns:
        The message, or `None` when `error` is not due to the package of a plugin
        missing, e.g. when it is due to a missing dependency of the plugin.
    """
    package = new_name.partition(".")[0]
    if not package.startswith(_plugin_prefix) or error.name != package:
        return None
    return (
        f"{old} moved to {new_name!r}, of the {package.replace('_', '-')} plugin, "
        "which is not installed."
    )


def _get_renamed_attribute(
    module: ModuleType, module_name: str, name: str, new_name: str
) -> Any:
    """Return the object a rename points to.

    `new_name` comes from
    [attribute_renames][gemseo._deprecation.aliases.attribute_renames], where it is
    always fully qualified, in any package, e.g. GEMSEO or a plugin. The lookup goes
    through `module` rather than a fresh [import_module][importlib.import_module] when
    the module part of `new_name` is `module` itself: `module` may be a stand-in whose
    attribute access is not a plain import (e.g.
    [_DeprecatedModule][gemseo._deprecation._DeprecatedModule] delegating to a
    monkeypatched target), and re-importing by name would silently bypass that.

    `new_name` may also be that of a module, e.g. for a module re-exported by a
    package, which is imported when its package does not hold it yet.

    Args:
        module: The module the attribute is looked up on when it stayed in its own
            (possibly renamed) module; ignored when the attribute moved elsewhere.
        module_name: The old fully-qualified name of the module defining the
            attribute, for the error message only.
        name: The old attribute name, for the error message only.
        new_name: The fully-qualified new name.

    Returns:
        The renamed object.

    Raises:
        AttributeError: When the new name does not exist, or when it is in a plugin
            that is not installed, except under a `from module import name`
            statement.
        ImportError: When the new name is in a plugin that is not installed, under a
            `from module import name` statement (see
            [_raise_unresolvable_attribute][gemseo._deprecation._raise_unresolvable_attribute]).
    """
    new_module_name, _, attribute_name = new_name.rpartition(".")
    if new_module_name == module.__name__:
        return getattr(module, attribute_name)
    try:
        new_module = import_module(new_module_name)
    except ModuleNotFoundError as error:
        msg = _get_missing_plugin_message(
            f"The attribute {name!r} of the module {module_name!r}", new_name, error
        )
        if msg is None:
            raise
        _raise_unresolvable_attribute(msg)
    try:
        return getattr(new_module, attribute_name)
    except AttributeError:
        # A submodule is an attribute of its package only once imported.
        if not hasattr(new_module, "__path__") or _find_spec(new_name) is None:
            raise
    return import_module(new_name)


class _DeprecatedModule(ModuleType):
    """A stand-in for a renamed module that delegates to its new location."""

    def __getattr__(self, name: str) -> Any:
        # Reached only when normal lookup on this module's namespace fails.
        target = self.__dict__["_deprecation_target"]
        _raise_for_unmigrated_attribute(self.__name__, name)
        new_name = attribute_renames.get(self.__name__, {}).get(name)
        if new_name is None:
            try:
                return getattr(target, name)
            except AttributeError:
                msg = f"module {self.__name__!r} has no attribute {name!r}"
                raise AttributeError(msg) from None
        # The module warning only names the new module, which does not carry the old
        # attribute name; point at the renamed attribute itself.
        _warn_attribute_rename(self.__name__, name, new_name)
        return _get_renamed_attribute(target, self.__name__, name, new_name)

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
        try:
            target = import_module(self._new_name)
        except ModuleNotFoundError as error:
            msg = _get_missing_plugin_message(
                f"The module {self._old_name!r}", self._new_name, error
            )
            if msg is None:
                raise
            raise ModuleNotFoundError(msg, name=self._new_name) from error
        module = _DeprecatedModule(self._old_name)
        module.__doc__ = target.__doc__
        module.__dict__["_deprecation_target"] = target
        # Keep the old package path importable so its submodules also redirect.
        target_path = getattr(target, "__path__", None)
        if target_path is not None:
            module.__path__ = target_path
        # Keep star imports from the old path working: the stand-in namespace is empty,
        # so without `__all__` the star import would bind nothing at all. The old names
        # of the attributes renamed in the old module are added, as the target does
        # not hold them, while the names that raise instead of being resolved are left
        # out, lest the star import fail partway.
        target_all = getattr(target, "__all__", None)
        if target_all is None:
            target_all = [name for name in vars(target) if not name.startswith("_")]
        renamed_names = [
            name
            for name in attribute_renames.get(self._old_name, {})
            if not name.startswith("_")
        ]
        unmigrated_names = unmigrated_attributes.get(self._old_name, ())
        module.__dict__["__all__"] = [
            name
            for name in dict.fromkeys((*target_all, *renamed_names))
            if name not in unmigrated_names
        ]
        return module

    def exec_module(self, module: ModuleType) -> None:  # noqa: D102
        warnings.warn(
            f"The module {self._old_name!r} is deprecated; "
            f"use {self._new_name!r} instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class _DissolvedPackage(ModuleType):
    """A stand-in for a package whose submodules were dissolved into others.

    The package defined no name of its own (it held only a docstring), so the stand-in
    resolves none either: its submodules are imported through the meta-path finder,
    which redirects them to their new location, and any other name raises an
    `AttributeError`, so that `from old_package import submodule` falls back to
    importing the submodule. A star import binds nothing.
    """

    def __getattr__(self, name: str) -> Any:
        # Reached only when normal lookup on this module's namespace fails.
        _raise_for_unmigrated_attribute(self.__name__, name)
        msg = f"module {self.__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)


class _DissolvedPackageLoader(Loader):
    """Load a dissolved package as an empty stand-in and warn about the dissolution."""

    def __init__(self, old_name: str, new_names: tuple[str, ...]) -> None:
        """
        Args:
            old_name: The deprecated fully-qualified package name being imported.
            new_names: The fully-qualified names of the packages that the old
                package's submodules were dissolved into, only listed in the warning.
        """  # noqa: D205 D212
        self._old_name = old_name
        self._new_names = new_names

    def create_module(self, spec: ModuleSpec) -> ModuleType:  # noqa: D102
        module = _DissolvedPackage(self._old_name)
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
    are not mistaken for an unknown name. The old names of the attributes removed with
    no replacement raise an `AttributeError` saying so, as on the stand-ins of the
    renamed modules, and those whose migration cannot be automated (the `TODO <text>`
    entries of the `attributes:` section) an `ImportError`.

    Args:
        module: The module whose old attribute names must keep working.
    """
    module_name = module.__name__
    renames = attribute_renames.get(module_name, {})
    previous_getattr = module.__dict__.get("__getattr__")

    def __getattr__(name: str) -> Any:  # noqa: N807
        _raise_for_unmigrated_attribute(module_name, name)
        new_name = renames.get(name)
        if new_name is None:
            if previous_getattr is not None:
                return previous_getattr(name)
            msg = f"module {module_name!r} has no attribute {name!r}"
            raise AttributeError(msg)
        _warn_attribute_rename(module_name, name, new_name)
        return _get_renamed_attribute(module, module_name, name, new_name)

    module.__dict__["__getattr__"] = __getattr__


_class_aliases_attribute: Final[str] = "_deprecation_class_aliases"
"""The name of the attribute of a class holding the renames aliased on it.

It maps the old names to the new ones, and is set in the class's own namespace.
"""


class _RenamedClassAttribute:
    """A data descriptor resolving the old name of a renamed class attribute.

    Being a data descriptor (it defines both `__get__` and `__set__`) makes it take
    precedence over an instance's own `__dict__`, so reads and writes of the old
    name through an instance are routed through it too.
    """

    def __init__(self, class_name: str, old_name: str, new_name: str) -> None:
        """
        Args:
            class_name: The name of the class defining the attribute.
            old_name: The old attribute name.
            new_name: The new attribute name.
        """  # noqa: D205 D212
        self._class_name = class_name
        self._old_name = old_name
        self._new_name = new_name

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        # Reached only when normal lookup finds no closer-matching attribute.
        _warn_class_attribute_rename(self._class_name, self._old_name, self._new_name)
        target = owner if instance is None else instance
        try:
            return getattr(target, self._new_name)
        except AttributeError:
            # Some renamed names are instance attributes, unreachable on the class.
            msg = (
                f"{self._class_name!r} has no attribute {self._new_name!r} "
                f"(renamed from {self._old_name!r})"
            )
            raise AttributeError(msg) from None

    def __set__(self, instance: Any, value: Any) -> None:
        _warn_class_attribute_rename(self._class_name, self._old_name, self._new_name)
        setattr(instance, self._new_name, value)


# A dict, as the enumeration's own member map, not a slower UserDict.
class _AliasingMemberMap(dict[str, Enum]):  # noqa: FURB189
    """The member map of an enumeration, resolving the old names of renamed members.

    It replaces the `_member_map_` of the enumeration, which the lookup of a member by
    name, e.g. `VariableType["FLOAT"]`, reads. An old name is only resolved by
    `__missing__`: it is neither contained nor iterated, so it is not listed among the
    members, e.g. by `__members__`.
    """

    renames: dict[str, str]
    """The old member names -> the new ones."""

    def __init__(self, class_name: str, members: Mapping[str, Enum]) -> None:
        """
        Args:
            class_name: The name of the enumeration.
            members: The members of the enumeration, by name.
        """  # noqa: D205 D212
        super().__init__(members)
        self.renames = {}
        self._class_name = class_name

    def __missing__(self, name: str) -> Enum:
        new_name = self.renames.get(name)
        if new_name is None:
            raise KeyError(name)
        # The stack is this method, the lookup of the enumeration, then the caller.
        _warn_class_attribute_rename(self._class_name, name, new_name, stacklevel=4)
        return self[new_name]


def _class_declares_attribute(cls: type, name: str) -> bool:
    """Return whether a class actually declares an attribute.

    An attribute counts as declared when it resolves on the class (a live class
    attribute, property, classmethod, ...), when it appears in the `__annotations__`
    of any class in the MRO (a bare class-level annotation, typically only ever
    assigned on instances in `__init__`), or when it is a pydantic model field. This
    tells apart a `classes:` table entry meant for `cls` from one that was written
    for an unrelated class that merely happens to share `cls`'s name.

    Args:
        cls: The class to inspect.
        name: The attribute name to look for.

    Returns:
        Whether `cls` declares `name`.
    """
    with warnings.catch_warnings():
        # Only checking declaration, not a real access to the name; guarded all the
        # same so a warning raised by some unrelated custom `__getattr__` cannot be
        # triggered or wasted under the "once" filter.
        warnings.simplefilter("ignore", DeprecationWarning)
        if hasattr(cls, name):
            return True
        if name in getattr(cls, "model_fields", {}):
            return True
    return any(name in vars(klass).get("__annotations__", {}) for klass in cls.__mro__)


def _alias_class_attributes(cls: type, renames: Mapping[str, str]) -> None:
    """Make the old names of a class's renamed attributes keep working.

    A data descriptor is set on the class for each renamed attribute whose old name
    is not still in use and whose new name `cls` actually declares (see
    [_class_declares_attribute][gemseo._deprecation._class_declares_attribute]), so
    it is reached only when nothing closer shadows it, and a table entry meant for a
    different class of the same name is not applied to `cls`. A subclass whose body
    still assigns an old name is handled by chaining `__init_subclass__`: the old
    binding is remapped to the new name before any other `__init_subclass__`
    (defined on `cls` or inherited) runs, subject to the same declared-new-name
    condition. The old name of a member of an enumeration is also resolved by the
    lookup of a member by name, e.g. `VariableType["FLOAT"]`, through an
    [_AliasingMemberMap][gemseo._deprecation._AliasingMemberMap], even when it is
    not aliased as an attribute because it resolves to an attribute inherited from a
    base class, e.g. the method `str.center` of a `StrEnum`, which a descriptor would
    shadow on the members.

    It can be called several times on the same class, e.g. for a nested class held by
    several classes, each listing its own renames of it, or after a module reload: the
    renames are recorded in the class's own namespace, and only those not aliased yet
    are added.

    Args:
        cls: The class whose old attribute names must keep working.
        renames: The mapping from old attribute name to new attribute name.
    """
    aliased: dict[str, str] | None = cls.__dict__.get(_class_aliases_attribute)
    if aliased is None:
        aliased = {}
        _chain_init_subclass(cls, aliased)
        setattr(cls, _class_aliases_attribute, aliased)

    new_renames = {
        old_name: new_name
        for old_name, new_name in renames.items()
        if old_name not in aliased
    }
    new_name_is_declared = {
        new_name: _class_declares_attribute(cls, new_name)
        for new_name in new_renames.values()
    }
    member_renames: dict[str, str] = {}
    for old_name, new_name in new_renames.items():
        if not new_name_is_declared[new_name]:
            # The table entry was written for another class that happens to share
            # cls's name; applying it here would alias an attribute cls never had.
            continue
        aliased[old_name] = new_name
        if (
            isinstance(cls, EnumMeta)
            and new_name in cls._member_map_
            and old_name not in cls._member_map_
        ):
            # The lookup by name only reads the members, so an old name resolving to
            # an inherited attribute, e.g. the method `str.center` of a `StrEnum`, is
            # aliased there all the same.
            member_renames[old_name] = new_name
        with warnings.catch_warnings():
            # This lookup is only checking liveness, not a real access to the old
            # name; it must not itself trigger (and thereby waste, under the "once"
            # filter) the warning of a descriptor set by an earlier, unrelated call.
            warnings.simplefilter("ignore", DeprecationWarning)
            old_name_is_live = hasattr(cls, old_name)
        if old_name_is_live:
            # The old name is still live: a stale table entry must not shadow it.
            continue
        setattr(cls, old_name, _RenamedClassAttribute(cls.__name__, old_name, new_name))

    if member_renames:
        member_map = cls._member_map_  # type: ignore[attr-defined]
        if not isinstance(member_map, _AliasingMemberMap):
            member_map = _AliasingMemberMap(cls.__name__, member_map)
            cls._member_map_ = member_map  # type: ignore[attr-defined]
        member_map.renames.update(member_renames)


def _chain_init_subclass(cls: type, renames: Mapping[str, str]) -> None:
    """Remap the old names assigned by the body of a subclass to the new ones.

    Args:
        cls: The class whose subclasses must have their old names remapped.
        renames: The mapping from old attribute name to new attribute name, read when
            a subclass is created, so that the renames added later apply too.
    """
    previous = cls.__dict__.get("__init_subclass__")

    def __init_subclass__(subclass: type, **kwargs: Any) -> None:  # noqa: N807
        subclass_vars = vars(subclass)
        for old_name, new_name in renames.items():
            if old_name in subclass_vars and new_name not in subclass_vars:
                _warn_class_attribute_rename(cls.__name__, old_name, new_name)
                setattr(subclass, new_name, subclass_vars[old_name])
        if previous is not None:
            previous.__func__(subclass, **kwargs)
        else:
            super(cls, subclass).__init_subclass__(**kwargs)

    cls.__init_subclass__ = classmethod(__init_subclass__)  # type: ignore[assignment]


def _install_class_attribute_aliases(module: ModuleType) -> None:
    """Make the old names of the renamed attributes of a module's classes keep working.

    Only the classes actually defined in the module (not merely re-exported by it)
    are aliased, so that each class is aliased exactly once, from its defining
    module.

    Args:
        module: The module whose classes' old attribute names must keep working.
    """
    module_name = module.__name__
    for obj in list(vars(module).values()):
        if (
            isinstance(obj, type)
            and obj.__module__ == module_name
            and obj.__name__ in class_attribute_renames
        ):
            _alias_class_and_nested_attributes(
                obj, class_attribute_renames[obj.__name__]
            )


def _alias_class_and_nested_attributes(cls: type, renames: Mapping[str, str]) -> None:
    """Make the old names of the renamed attributes of a class and its nested ones work.

    A dotted old name, e.g. `DifferentiationMethod.USER_GRAD`, is that of an attribute
    of the class this path leads to from `cls`, e.g. a member of an enumeration held by
    `cls`; it is aliased on that class, which is left alone when the path does not lead
    to a class. The renames of a nested class held by several classes are merged, so
    that each of them may list its own.

    Args:
        cls: The class whose old attribute names must keep working.
        renames: The mapping from old attribute name to new attribute name.
    """
    own_renames: dict[str, str] = {}
    nested_renames: dict[str, dict[str, str]] = {}
    for old_name, new_name in renames.items():
        path, _, old_attribute_name = old_name.rpartition(".")
        if path:
            new_attribute_name = new_name.rpartition(".")[2]
            nested_renames.setdefault(path, {})[old_attribute_name] = new_attribute_name
        else:
            own_renames[old_name] = new_name
    if own_renames:
        _alias_class_attributes(cls, own_renames)
    for path, path_renames in nested_renames.items():
        nested_class: Any = cls
        with warnings.catch_warnings():
            # Only resolving the nested class, not a real access to a name.
            warnings.simplefilter("ignore", DeprecationWarning)
            for name in path.split("."):
                nested_class = getattr(nested_class, name, None)
        if isinstance(nested_class, type):
            _alias_class_attributes(nested_class, path_renames)


class _AliasLoader(Loader):
    """Load a module normally, then alias the old names of its renamed attributes.

    Both the module's own renamed attributes (when it is one of
    [live_aliased_modules][gemseo._deprecation.aliases.live_aliased_modules]) and the
    renamed attributes of the classes it defines are aliased; a live module may have
    either, both or neither.
    """

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
        if module.__name__ in live_aliased_modules:
            _install_attribute_aliases(module)
        _install_class_attribute_aliases(module)


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
        if not fullname.startswith(_package_prefix):
            return None
        _raise_for_removed_module(fullname)
        if fullname in dissolved_packages:
            return ModuleSpec(
                fullname,
                _DissolvedPackageLoader(fullname, dissolved_packages[fullname]),
                is_package=True,
            )
        new_name = _resolve_new_name(fullname)
        if new_name is not None:
            return ModuleSpec(fullname, _DeprecatedModuleLoader(fullname, new_name))
        return self._find_alias_spec(fullname)

    def _find_alias_spec(self, fullname: str) -> ModuleSpec | None:
        """Return the spec of a live module, wrapped to alias its renamed attributes.

        The module is loaded by its real loader, wrapped so that the old names of
        its own renamed attributes (if it is one of `live_aliased_modules`) and of
        its classes' renamed attributes keep resolving. Every live `gemseo` module is
        wrapped, as a class rename cannot be resolved to its defining module ahead of
        time.

        Args:
            fullname: The fully-qualified name of the module being imported.

        Returns:
            The wrapped spec, or `None` when `fullname` is already being looked up
            (a re-entrant call) or has no real spec to wrap.
        """
        if fullname in self._finding:
            return None
        self._finding.add(fullname)
        try:
            spec = _find_spec(fullname)
        finally:
            self._finding.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _AliasLoader(spec.loader)
        return spec


def install() -> None:
    """Register the deprecated-import finder.

    Also alias the old names of the renamed attributes of the modules and classes
    that are already imported, and register warning filters so the emitted
    `DeprecationWarning` is shown once, regardless of the default filters: without them,
    a warning attributed to a module of the library (rather than to `__main__`) would be
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
        warnings.filterwarnings(
            "once",
            message=r"The attribute '.*' of the class '.*' is deprecated",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "once",
            message=r"The variable type '.*' is deprecated",
            category=DeprecationWarning,
        )
    sys.meta_path.insert(0, _DeprecatedModuleFinder())
    # The finder only sees the modules imported from now on; the ones already
    # imported (by gemseo itself) are aliased here.
    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue
        if module_name != "gemseo" and not module_name.startswith(_package_prefix):
            continue
        if module_name in live_aliased_modules:
            _install_attribute_aliases(module)
        _install_class_attribute_aliases(module)
    _installed = True
