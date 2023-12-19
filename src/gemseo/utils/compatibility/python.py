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
"""Python versions compatibility layer."""

from __future__ import annotations

from sys import version_info
from typing import Final

PYTHON_VERSION: Final[tuple[int, int, int]] = version_info

if PYTHON_VERSION < (3, 10):  # pragma: >=3.10 no cover
    from typing_extensions import ParamSpecArgs  # noqa: F401
    from typing_extensions import ParamSpecKwargs  # noqa: F401

    EllipsisType = type(Ellipsis)
else:  # pragma: <3.10 no cover
    from types import EllipsisType
    from typing import ParamSpecArgs  # noqa: F401
    from typing import ParamSpecKwargs  # noqa: F401

    EllipsisType  # noqa: B018
