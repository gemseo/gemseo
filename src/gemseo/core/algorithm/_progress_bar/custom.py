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
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable
    from io import TextIOWrapper
    from numbers import Real

logger = logging.getLogger(__name__)


def _log_status(status: str) -> None:
    """Log the tqdm progress bar status.

    Args:
        status: The progress bar status.
    """
    if " 0%|" in status:
        return

    status = status.rstrip(string.whitespace)
    if status:
        logger.info("  %s", status)


class CustomTqdmProgressBar(tqdm.tqdm):
    """A custom tqdm progress bar with improved time units.

    Use minute, hour and day for slower processes.
    """

    _bar_format: ClassVar[str] = (
        "{{desc}} {{percentage:3.0f}}%|{{bar}}| {{n_fmt}}/{{total_fmt}} "
        "[{{elapsed}}<{{remaining}}, {rate}{{postfix}}]"
    )
    """The bar_format used by tqdm.format_meter."""

    _initial_rate: ClassVar[str] = "? it/sec"

    _rate_template: ClassVar[str] = "{:5.2f} it/{}"

    __bar_format_label: Final[str] = "bar_format"
    __day_label: Final[str] = "day"
    __hour_label: Final[str] = "hour"
    __min_label: Final[str] = "min"
    __sec_label: Final[str] = "sec"

    __file_stream_class: Final[type] = StringIO
    """The class used to create the dummy file stream for tqdm."""

    _log_one_status_per_percentage: ClassVar[bool] = False
    """Whether to log a status only when its rounded percentage changes.

    When `True`,
    a progress bar with a total logs a status
    only when its rounded percentage differs from the one of the last logged status
    or at its last iteration,
    and logs its last skipped status, if any, when it is closed.
    """

    __last_percentage: str = ""
    """The rounded percentage of the last logged status, if any."""

    __skipped_status_data: dict[str, Any] | None = None
    """The data of the last status when it was skipped, otherwise `None`.

    These data are the `format_dict` of the progress bar at the time of this status,
    so that the status logged at closing has the elapsed time and the rate
    of its iteration.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: D102
        # Use a file stream to prevent tqdm from trying to adapt the progress bar
        # to the current terminal because its rendering will vary and because it is
        # not needed since the progress bar goes to a logger.
        kwargs["file"] = self.__file_stream_class()
        super().__init__(*args, **kwargs)

    def display(self, msg: str | None = None, pos: int | None = None) -> bool:
        """Log a status, unless it is skipped.

        When `_log_one_status_per_percentage` is `True`
        and the progress bar has a total,
        the current status is skipped
        when its rounded percentage is the one of the last logged status
        and its iteration is not the last one;
        its data are kept to log it when the progress bar is closed.

        Args:
            msg: The status to log.
                If `None`, use the current status of the progress bar.
            pos: The position of the progress bar.
                If `None`, use its current position.

        Returns:
            `False` when tqdm hides the status
            because of the position of the progress bar,
            `True` otherwise, including when the status is skipped.
        """
        if msg is None and self._log_one_status_per_percentage and self.total:
            # The percentage as tqdm.format_meter writes it in the status.
            percentage = f"{self.n / self.total * 100:3.0f}"
            if percentage == self.__last_percentage and self.n < self.total:
                self.__skipped_status_data = self.format_dict
                return True

            self.__last_percentage = percentage
            self.__skipped_status_data = None

        return super().display(msg=msg, pos=pos)

    def close(self) -> None:
        """Log the last status when it was skipped, then close the progress bar.

        This status is the last one
        e.g. when an optimizer stops before its last iteration;
        it is logged with the elapsed time and the rate of its iteration.
        """
        if not self.disable and self.__skipped_status_data is not None:
            status = self.format_meter(**self.__skipped_status_data)
            self.__skipped_status_data = None
            with self._lock:
                super().display(msg=status)

        super().close()

    @classmethod
    def format_meter(  # noqa: D102
        cls, n: float, total: float, elapsed: float, **kwargs: Any
    ) -> str:
        kwargs[cls.__bar_format_label] = cls._bar_format.format(
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
            return cls._initial_rate

        rate = n / elapsed
        if rate >= 1:
            return cls._rate_template.format(rate, cls.__sec_label)

        rate *= 60
        if rate >= 1:
            return cls._rate_template.format(rate, cls.__min_label)

        rate *= 60
        if rate >= 1:
            return cls._rate_template.format(rate, cls.__hour_label)

        return cls._rate_template.format(rate * 24, cls.__day_label)

    @staticmethod
    def status_printer(file: TextIOWrapper | StringIO) -> Callable[[str], None]:
        """Create the function logging the progress bar statuses.

        Args:
            file: The output stream.
                This argument defined in the parent class is not used.
                Use `logging` instead.

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
            self.__file_stream_class(), tqdm_instance=self
        )
