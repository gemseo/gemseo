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
"""A griffe extension exposing the lazy re-exports of the GEMSEO packages.

A package calling
[install_lazy_reexport][gemseo.util.package_import.install_lazy_reexport]
sets its `__all__` at runtime,
so griffe, which reads the source statically,
sees its re-exported names as private imports
and the API reference renders none of them.
This extension imports such a package
and gives griffe its runtime `__all__` as the exported names.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING
from typing import Any

from griffe import Extension

if TYPE_CHECKING:
    from griffe import Module


class LazyExports(Extension):
    """Use the runtime `__all__` of the packages with lazy re-exports."""

    def on_module_members(self, *, mod: Module, **kwargs: Any) -> None:
        """Set the exported names of a package with lazy re-exports.

        Args:
            mod: The module.
            **kwargs: The other arguments of the event.
        """
        if not mod.is_init_module or "install_lazy_reexport(" not in mod.source:
            return

        mod.exports = list(import_module(mod.path).__all__)
