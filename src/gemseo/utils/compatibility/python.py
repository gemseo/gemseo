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

from typing_extensions import get_args  # noqa: F401
from typing_extensions import get_origin  # noqa: F401

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

if PYTHON_VERSION < (3, 9):  # pragma: >=3.9 no cover

    def remove_suffix(string: str, suffix: str) -> str:  # noqa: D103
        if string.endswith(suffix):
            return string[: -len(suffix)]
        return string

    _get_origin = get_origin

    def get_origin(tp) -> type:  # noqa: D103
        origin = _get_origin(tp)
        if origin is None:
            return getattr(tp, "__origin__", None)
        return origin

    _get_args = get_args

    def get_args(tp) -> tuple:  # noqa: D103
        args = _get_args(tp)
        if not args:
            return getattr(tp, "__args__", None)
        return args

else:  # pragma: <3.9 no cover

    def remove_suffix(string: str, suffix: str) -> str:  # noqa: D103
        return string.removesuffix(suffix)
