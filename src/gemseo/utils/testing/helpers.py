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
"""Comparison tools for testing."""

from __future__ import annotations

import contextlib
import re
from contextlib import nullcontext
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

import pytest
from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import Generator

    from regex import Pattern
    from syrupy.assertion import SnapshotAssertion

__ABSTRACTMETHODS__: Final[str] = "__abstractmethods__"

# Pydantic embeds its version in error message URLs, e.g.
# `https://errors.pydantic.dev/2.13/v/value_error`.
# Replace the version with a stable placeholder so snapshots survive pydantic
# upgrades.
_PYDANTIC_VERSION_IN_URL: Final[Pattern[str]] = re.compile(
    r"(errors\.pydantic\.dev/)\d+(?:\.\d+)*(/v/)"
)


@contextlib.contextmanager
def concretize_classes(*classes: type) -> None:
    """Context manager forcing classes to be concrete.

    Args:
        *classes: The classes.
    """
    classes_to___abstractmethods__ = {}
    for cls in classes:
        if hasattr(cls, __ABSTRACTMETHODS__):
            classes_to___abstractmethods__[cls] = cls.__abstractmethods__
            del cls.__abstractmethods__

    try:
        yield
    finally:
        for cls, __abstractmethods__ in classes_to___abstractmethods__.items():
            cls.__abstractmethods__ = __abstractmethods__


@contextlib.contextmanager
def assert_exception(
    exception: type[BaseException] | tuple[type[BaseException], ...],
    snapshot: SnapshotAssertion,
) -> Generator[pytest.ExceptionInfo, None, None]:
    """Assert an exception is raised.

    When `pytest` is executed with `--snaphot-update`,
    the exception message is captured
    and stored in the `__snapshots__` directory
    placed in the test module directory.

    Args:
        exception: The expected exception type(s).
        snapshot: The syrupy `snapshot` fixture
            containing the exception message.

    Yields:
        The pytest `ExceptionInfo` for further assertions if needed.
    """
    with pytest.raises(exception) as exc_info:
        yield exc_info

    message = str(exc_info.value)
    if isinstance(exc_info.value, ValidationError):
        message = _PYDANTIC_VERSION_IN_URL.sub(r"\1X.Y\2", message)

    assert message == snapshot


class do_not_raise(nullcontext):  # noqa: N801
    """Return a context manager like `pytest.raises()` but that does nothing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Args:
            *args: The arguments to match the signature of `pytest.raises()`.
            **kwargs: The keyword arguments to match the signature of
                `pytest.raises()`.
        """  # noqa:D205 D212 D415
        super().__init__()
