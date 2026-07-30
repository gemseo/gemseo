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

from importlib import import_module
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import MutableMapping


def resolve_lazy_export(package_name: str, location: str, name: str) -> Any:
    """Resolve a ``"module[:attr.chain]"`` location to its object.

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
    """
    module_path, _, attribute = location.partition(":")
    if module_path.split(".", 1)[0] == "gemseo":
        module = import_module(module_path)
    else:
        module = import_module(f"{package_name}.{module_path}")

    obj: Any = module
    for attr in (attribute or name).split("."):
        obj = getattr(obj, attr)
    return obj


def install_lazy_reexport(
    namespace: MutableMapping[str, Any],
    name_to_location: Mapping[str, str],
    extra_all: Iterable[str] = (),
) -> None:
    """Install ``__all__``, ``__dir__`` and ``__getattr__`` into a package namespace.

    Meant to be called from a package ``__init__.py`` as
    ``install_lazy_reexport(globals(), _NAME_TO_LOCATION)``.

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
        return [*namespace, *names]

    def __getattr__(name: str) -> Any:  # noqa: N807
        location = name_to_location.get(name)
        if location is None:
            msg = f"module {package_name!r} has no attribute {name!r}"
            raise AttributeError(msg)
        return resolve_lazy_export(package_name, location, name)

    namespace["__all__"] = [*names, *extra_all]
    namespace["__dir__"] = __dir__
    namespace["__getattr__"] = __getattr__
