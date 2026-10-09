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
"""Shared helper for the lazy re-export of names from a package."""

from __future__ import annotations

import dis
import sys
from importlib import import_module
from typing import TYPE_CHECKING
from typing import Any
from typing import Final
from typing import NoReturn

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import MutableMapping

_import_from_opcode: Final[int] = dis.opmap["IMPORT_FROM"]
"""The opcode importing a name in a `from module import name` statement."""

getattr_wrapper_marker: Final[str] = "_gemseo_getattr_wrapper"
"""The name of the global marking a module that looks attributes up for its callers.

A module defining a module-level `__getattr__`, or a helper called by one, sets this
global to `True` so that
[is_imported_from][gemseo.util.package_import.is_imported_from] skips its frames
when it searches for the frame looking the attribute up.
This module and `gemseo._deprecation`, whose `__getattr__` wraps the lazy one
of this module, set it;
a plugin wrapping a module-level `__getattr__` can set it too.
"""

_gemseo_getattr_wrapper: Final[bool] = True
"""Whether this module looks attributes up for its callers.

See [getattr_wrapper_marker][gemseo.util.package_import.getattr_wrapper_marker].
"""


def is_imported_from() -> bool:
    """Return whether the attribute being looked up is imported by a `from` statement.

    The frame looking the attribute up is the first one outside the modules
    marked by the global named
    [getattr_wrapper_marker][gemseo.util.package_import.getattr_wrapper_marker],
    i.e. the modules defining module-level `__getattr__` and their helpers.
    It imports the attribute when it is executing the `IMPORT_FROM` instruction
    of a `from module import name` statement.

    Returns:
        Whether the attribute is imported by a `from module import name` statement.
    """
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get(getattr_wrapper_marker, False):
        frame = frame.f_back
    return (
        frame is not None and frame.f_code.co_code[frame.f_lasti] == _import_from_opcode
    )


def raise_unresolvable_attribute(
    msg: str, cause: BaseException | None = None
) -> NoReturn:
    """Raise the error of a module-level attribute that cannot be resolved.

    An `AttributeError` is raised, so that `hasattr` and `getattr` with a default keep
    working, as they only swallow an `AttributeError`, except under a
    `from module import name` statement, which would replace an `AttributeError` by a
    generic "cannot import name" `ImportError`, losing the message and the cause:
    an `ImportError` is raised there instead
    (see [is_imported_from][gemseo.util.package_import.is_imported_from]).

    Args:
        msg: The error message.
        cause: The error the raised one is chained to, if any.

    Raises:
        ImportError: Under a `from module import name` statement.
        AttributeError: Otherwise.
    """
    error_type = ImportError if is_imported_from() else AttributeError
    if cause is None:
        raise error_type(msg)
    raise error_type(msg) from cause


def get_location_module_name(package_name: str, location: str) -> str:
    """Return the full name of the module part of a location.

    Args:
        package_name: The name of the package requesting the resolution.
        location: The location of the object, formatted as
            ``"module[:attribute.chain]"``. The module part is absolute
            when its first segment is ``gemseo``, otherwise relative to
            ``package_name``.

    Returns:
        The full name of the module.
    """
    module_path = location.partition(":")[0]
    if module_path.split(".", 1)[0] == "gemseo":
        return module_path
    return f"{package_name}.{module_path}"


def resolve_lazy_export(package_name: str, location: str, name: str) -> Any:
    """Resolve a ``"module[:attr.chain]"`` location to its object.

    When an attribute of the location does not exist,
    an error chained to the original one is raised:
    an ``ImportError`` under a ``from package import name`` statement,
    which would otherwise replace the message and the cause
    by a generic "cannot import name" one,
    and an ``AttributeError`` otherwise,
    so that ``hasattr`` and ``getattr`` with a default keep working
    (see [raise_unresolvable_attribute]
    [gemseo.util.package_import.raise_unresolvable_attribute]).

    An error raised while importing the module of the location,
    e.g. a ``ModuleNotFoundError`` for a misspelt module,
    propagates unchanged,
    except an ``AttributeError`` under a ``from package import name`` statement,
    which is chained to an ``ImportError`` for the same reason.

    Args:
        package_name: The name of the package requesting the resolution.
        location: The location of the object, formatted as
            ``"module[:attribute.chain]"``. The module part is imported absolutely
            when its first segment is ``gemseo``, otherwise relative to
            ``package_name``. The attribute part is dot-split and walked; when
            omitted, it defaults to ``name``.
        name: The exported name, used as the attribute when ``location`` has no
            ``":attribute"`` part.

    Returns:
        The object resolved from its defining module.

    Raises:
        AttributeError: If an attribute of the chain does not exist,
            or if the import of the module raises one,
            except under a ``from package import name`` statement.
        ImportError: If an attribute of the chain does not exist,
            or if the import of the module raises an ``AttributeError``,
            under a ``from package import name`` statement.
    """
    module_name = get_location_module_name(package_name, location)
    try:
        module = import_module(module_name)
    except AttributeError as error:
        if not is_imported_from():
            raise
        msg = (
            f"The import of the module {module_name!r} of the name {name!r} "
            f"exported by {package_name!r} raised an error ({error})."
        )
        raise ImportError(msg) from error
    try:
        return _get_attribute_chain(module, location.partition(":")[2] or name)
    except AttributeError as error:
        msg = (
            f"The location {location!r} of the name {name!r} "
            f"exported by {package_name!r} cannot be resolved ({error})."
        )
        raise_unresolvable_attribute(msg, error)


def _get_attribute_chain(obj: Any, attribute_chain: str) -> Any:
    """Walk a dot-separated chain of attributes from an object.

    Args:
        obj: The object the walk starts from.
        attribute_chain: The dot-separated names of the attributes to walk.

    Returns:
        The object at the end of the chain.

    Raises:
        AttributeError: If an attribute of the chain does not exist.
    """
    for attribute in attribute_chain.split("."):
        obj = getattr(obj, attribute)
    return obj


def install_lazy_reexport(
    namespace: MutableMapping[str, Any],
    name_to_location: Mapping[str, str],
    extra_all: Iterable[str] = (),
) -> None:
    """Install ``__all__``, ``__dir__`` and ``__getattr__`` into a package namespace.

    Meant to be called from a package ``__init__.py`` as
    ``install_lazy_reexport(globals(), _name_to_location)``.

    The installed ``__getattr__`` caches each resolved object in ``namespace``,
    so that the following accesses are plain attribute lookups.
    As with an eager ``from module import Name``,
    the package then keeps the object it resolved first:
    a later change of the attribute in its defining module, e.g. a monkeypatch,
    is not seen through the package.

    The installed ``__getattr__`` raises an ``AttributeError``
    for a name that is not exported,
    and the errors of [resolve_lazy_export]
    [gemseo.util.package_import.resolve_lazy_export] for an exported name
    whose location cannot be resolved.

    Args:
        namespace: The ``globals()`` of the calling package ``__init__.py``.
        name_to_location: Mapping from exported name to
            ``"module[:attribute.chain]"`` location (see [resolve_lazy_export]
            [gemseo.util.package_import.resolve_lazy_export]).
        extra_all: Names exposed by ``__all__`` that are not lazily re-exported
            (e.g. eagerly imported names).
    """
    package_name = namespace["__name__"]
    names = list(name_to_location)

    def __dir__() -> list[str]:  # noqa: N807
        return list({*namespace, *names})

    def __getattr__(name: str) -> Any:  # noqa: N807
        location = name_to_location.get(name)
        if location is None:
            msg = f"module {package_name!r} has no attribute {name!r}"
            raise AttributeError(msg)
        obj = resolve_lazy_export(package_name, location, name)
        namespace[name] = obj
        return obj

    namespace["__all__"] = [*names, *extra_all]
    namespace["__dir__"] = __dir__
    namespace["__getattr__"] = __getattr__
