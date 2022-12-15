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

The namespaces implementation itself is mainly in :mod:`~gemseo.core.grammars` and
:mod:`~gemseo.core.discipline`
"""
from __future__ import annotations

from typing import Any
from typing import Iterable
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Union

NamespacesMapping = MutableMapping[str, Union[str, List[str]]]

namespaces_separator = ":"


def split_namespace(data_name: str) -> list[str, str]:
    """Return the (namespace, name) pair from a data name.

    For instance if data_name = ``my:namesp:ace:a`` and the separator is ``:``,
    returns (``my:namesp:ace``,``a``).

    If there is no namespace prefix in ``data_name``, returns ``data_name``.

    In case data_name contains the namespace separator but empty name,
    or empty namespace,
    returns the (namespace, name) pair, containing eventually empty strings.

    Args:
        data_name: The data name containing the namespace name.

    Returns:
        The namespace name and the data name.
    """
    return data_name.rsplit(namespaces_separator, -1)


def remove_prefix_from_name(name: str) -> str:
    """Remove namespace prefix from the name.

    Args:
        data: The container of data names, or a single data name.

    Returns:
        The data names without prefixes.
    """
    return name.rsplit(namespaces_separator, 1)[-1]


def remove_prefix_from_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """Remove namespaces prefixes from mapping keys if any.

    Args:
        data: The mapping which names may contain namespaces.

    Returns:
        The mapping without prefixes in its keys.
    """
    return {k.rsplit(namespaces_separator, 1)[-1]: v for k, v in data.items()}


def remove_prefix_from_list(names: Iterable[str]) -> list[str]:
    """Remove namespaces prefixes from names, if any.

    Args:
        names: The names that may contain namespaces.

    Returns:
        The names without prefixes in its keys.
    """
    return [d.rsplit(namespaces_separator, 1)[-1] for d in names]


def update_namespaces(
    namespaces: NamespacesMapping, other_namespaces: NamespacesMapping
) -> None:
    """Update namespaces with the key/value pairs from other, overwriting existing keys.

    Args:
        namespaces: The namespaces to update.
        other_namespaces: The namespaces to update from.
    """
    for name, other_ns in other_namespaces.items():
        curr_ns = namespaces.get(name)
        if curr_ns is None:
            namespaces[name] = other_ns
        elif isinstance(curr_ns, str):
            if isinstance(other_ns, str):
                namespaces[name] = [curr_ns, other_ns]
            else:
                namespaces[name] = [curr_ns] + other_ns
        else:
            if isinstance(other_ns, str):
                namespaces[name].append(other_ns)
            else:
                namespaces[name].extend(other_ns)
