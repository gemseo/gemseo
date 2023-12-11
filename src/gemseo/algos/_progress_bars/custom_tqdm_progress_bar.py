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
"""A custom tqdm progress bar with improved time units."""

from __future__ import annotations

import logging
import string
from io import StringIO
from io import TextIOWrapper
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Final

import tqdm

if TYPE_CHECKING:
    from numbers import Real

LOGGER = logging.getLogger(__name__)


def _log_status(status: str) -> None:
    """Log the tqdm progress bar status.

    Args:
        status: The progress bar status.
    """
    if " 0%|" in status:
        return

    status = status.rstrip(string.whitespace)
    if status:
        LOGGER.info("  %s", status)


class CustomTqdmProgressBar(tqdm.tqdm):
    """A custom tqdm progress bar with improved time units.

    Use minute, hour and day for slower processes.
    """

    _BAR_FORMAT: ClassVar[str] = (
        "{{desc}} {{percentage:3.0f}}%|{{bar}}| {{n_fmt}}/{{total_fmt}} "
        "[{{elapsed}}<{{remaining}}, {rate}{{postfix}}]"
    )
    """The bar_format used by tqdm.format_meter."""

    _INITIAL_RATE: ClassVar[str] = "? it/sec"

    _RATE_TEMPLATE: ClassVar[str] = "{:5.2f} it/{}"

    __BAR_FORMAT_LABEL: Final[str] = "bar_format"
    __DAY_LABEL: Final[str] = "day"
    __HOUR_LABEL: Final[str] = "hour"
    __MIN_LABEL: Final[str] = "min"
    __SEC_LABEL: Final[str] = "sec"

    __FILE_STREAM_CLASS: Final[type] = StringIO
    """The class used to create the dummy file stream for tqdm."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D102
        # Use a file stream to prevent tqdm from trying to adapt the progress bar
        # to the current terminal because its rendering will vary and because it is
        # not needed since the progress bar goes to a logger.
        kwargs["file"] = self.__FILE_STREAM_CLASS()
        super().__init__(*args, **kwargs)

    @classmethod
    def format_meter(  # noqa: D102
        cls, n: float, total: float, elapsed: float, **kwargs: Any
    ) -> str:
        kwargs[cls.__BAR_FORMAT_LABEL] = cls._BAR_FORMAT.format(
            rate=cls.__get_rate_expression(n, elapsed)
        )
        return tqdm.tqdm.format_meter(n, total, elapsed, **kwargs)

    @classmethod
    def __get_rate_expression(cls, n: Real, elapsed: Real) -> str:
        """Get the string expression of the rate.

        Args:
            n: The number of finished iterations.
            elapsed: The number of seconds passed since start.

        Returns:
            The rate string expression.
        """
        if elapsed == 0:
            return cls._INITIAL_RATE

        rate = n / elapsed
        if rate >= 1:
            return cls._RATE_TEMPLATE.format(rate, cls.__SEC_LABEL)

        rate *= 60
        if rate >= 1:
            return cls._RATE_TEMPLATE.format(rate, cls.__MIN_LABEL)

        rate *= 60
        if rate >= 1:
            return cls._RATE_TEMPLATE.format(rate, cls.__HOUR_LABEL)

        return cls._RATE_TEMPLATE.format(rate * 24, cls.__DAY_LABEL)

    @staticmethod
    def status_printer(file: TextIOWrapper | StringIO) -> Callable[[str], None]:
        """Create the function logging the progress bar statuses.

        Args:
            file: The output stream.
                This argument defined in the parent class is not used.
                Use ``logging`` instead.

        Returns:
            The function logging the progress bar statuses.
        """
        return _log_status

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # A file-like stream cannot be pickled.
        del state["fp"]
        return state

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        # Set back the file-like stream to its state as done in tqdm.__init__.
        self.fp = tqdm.utils.DisableOnWriteError(
            self.__FILE_STREAM_CLASS(), tqdm_instance=self
        )
