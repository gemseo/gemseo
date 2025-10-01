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
"""Compatibility between different versions of matplotlib."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING
from typing import Final

from matplotlib import pyplot as plt
from packaging.version import parse as parse_version

if TYPE_CHECKING:
    from packaging.version import Version

MATPLOTLIB_VERSION: Final[Version] = parse_version(version("matplotlib"))


def boxplot(x, labels=None, vert=None, ax=None, **kwargs):  # noqa: D103
    plot = (plt if ax is None else ax).boxplot
    if parse_version(version("matplotlib")) < parse_version("3.9"):
        kwargs["labels"] = labels
    else:
        kwargs["tick_labels"] = labels

    if parse_version(version("matplotlib")) < parse_version("3.11"):
        kwargs["vert"] = vert
    else:
        kwargs["orientation"] = "vertical" if vert in [True, None] else "horizontal"

    return plot(x, **kwargs)
