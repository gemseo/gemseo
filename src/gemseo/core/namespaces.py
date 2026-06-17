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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Utility functions and classes to handle namespaces.

The namespaces implementation itself is mainly in
[gemseo.core.grammars][gemseo.core.grammars] and
[gemseo.core.discipline][gemseo.core.discipline].
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator

MutableNamespacesMapping = MutableMapping[str, str | list[str]]
NamespacesMapping = Mapping[str, str | list[str]]

namespaces_separator = ":"
"""The special character for separating namespace and variable name."""


def split_namespace(name: str) -> list[str]:
    """Return the (namespace, name) pair from a data name.

    For instance if data_name = `my:namespace:a` and the separator is `:`,
    returns (`my:namespace`,`a`).

    If there is no namespace prefix in `data_name`, returns `data_name`.

    In case data_name contains the namespace separator but empty name,
    or empty namespace,
    returns the (namespace, name) pair, containing eventually empty strings.

    Args:
        name: The data name containing the namespace name.

    Returns:
        The namespace name and the data name.
    """
    return name.rsplit(namespaces_separator, 1)


def remove_prefix(names: Iterable[str]) -> Iterator[str]:
    """Remove namespaces prefixes from names, if any.

    Args:
        names: The names that may contain namespaces.

    Returns:
        The names without prefixes in its keys.
    """
    return (d.rsplit(namespaces_separator, 1)[-1] for d in names)


def update_namespaces(
    namespaces: MutableNamespacesMapping,
    other_namespaces: NamespacesMapping,
) -> None:
    """Update namespaces with the key/value pairs from other.

    This is the non-nested variant: a name maps to a single namespaced name.
    It is the only variant allowed for leaf disciplines.
    Process disciplines, which may aggregate the same name under several
    namespaces, must use
    [update_nested_namespaces][gemseo.core.namespaces.update_nested_namespaces].

    Args:
        namespaces: The namespaces to update.
        other_namespaces: The namespaces to update from.

    Raises:
        ValueError: If a name is already mapped to a different namespaced name,
            which would require nesting.
    """
    for name, other_ns in other_namespaces.items():
        curr_ns = namespaces.get(name)
        if curr_ns is not None and curr_ns != other_ns:
            msg = (
                f"The name {name!r} is already mapped to the namespaced name "
                f"{curr_ns!r}; mapping it to {other_ns!r} would require nesting, "
                "which is only allowed for process disciplines."
            )
            raise ValueError(msg)
        namespaces[name] = other_ns


def update_nested_namespaces(
    namespaces: MutableNamespacesMapping,
    other_namespaces: NamespacesMapping,
) -> None:
    """Update namespaces with the key/value pairs from other, allowing nesting.

    A name may map to several namespaced names (a list), as happens when a
    process discipline aggregates sub-disciplines that share a name under
    different namespaces.

    Args:
        namespaces: The namespaces to update.
        other_namespaces: The namespaces to update from.
    """
    for name, other_ns in other_namespaces.items():
        curr_ns = namespaces.get(name)
        if curr_ns is None:
            # Copy the list so a later in-place update does not mutate
            # other_namespaces, which may be another grammar's mapping.
            namespaces[name] = (
                list(other_ns) if isinstance(other_ns, list) else other_ns
            )
        elif isinstance(curr_ns, str):
            if isinstance(other_ns, str):
                namespaces[name] = [curr_ns, other_ns]
            else:
                namespaces[name] = [curr_ns, *other_ns]
        elif isinstance(other_ns, str):
            # curr_ns is the list stored in namespaces[name]; mutate it in place.
            curr_ns.append(other_ns)
        else:
            curr_ns.extend(other_ns)
