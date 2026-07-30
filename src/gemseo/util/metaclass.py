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
"""Some metaclasses."""

from __future__ import annotations

from abc import ABCMeta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # The runtime import returns either `type` or the real metaclass depending on
    # the ``DOCSTRING_INHERITANCE_ENABLE`` environment variable, which confuses
    # static type checkers; import the concrete class for type checking only.
    from docstring_inheritance._internal import GoogleDocstringInheritanceMeta
else:
    from docstring_inheritance import GoogleDocstringInheritanceMeta


class ABCGoogleDocstringInheritanceMeta(ABCMeta, GoogleDocstringInheritanceMeta):
    """A metaclass for creating abstract classes that inherit docstrings."""
