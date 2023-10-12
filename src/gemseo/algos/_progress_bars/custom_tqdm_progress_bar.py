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

import io
from numbers import Real
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Final

import tqdm
from tqdm.utils import _unicode
from tqdm.utils import disp_len

from gemseo.algos._progress_bars.tqdm_to_logger import TqdmToLogger


class CustomTqdmProgressBar(tqdm.tqdm):
    """A custom tqdm progress bar with improved time units.

    Use minute, hour and day for slower processes.
    """

    _BAR_FORMAT: ClassVar[
        str
    ] = "{{desc}} {{percentage:3.0f}}%|{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, {rate}{{postfix}}]"  # noqa: B950
    """The bar_format used by tqdm.format_meter."""

    __BAR_FORMAT_LABEL: Final[str] = "bar_format"
    __DAY_LABEL: Final[str] = "day"
    __FP_LABEL: Final[str] = "fp"
    __HOUR_LABEL: Final[str] = "hour"
    _INITIAL_RATE: ClassVar[str] = "? it/sec"
    __MIN_LABEL: Final[str] = "min"
    _RATE_TEMPLATE: ClassVar[str] = "{:5.2f} it/{}"
    __SEC_LABEL: Final[str] = "sec"

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
        n = rate * 60
        if n > 1:
            return cls._RATE_TEMPLATE.format(rate, cls.__SEC_LABEL)

        rate = n
        n = rate * 60
        if n > 1:
            return cls._RATE_TEMPLATE.format(rate, cls.__MIN_LABEL)

        rate = n
        n = rate * 24
        if n > 1:
            return cls._RATE_TEMPLATE.format(rate, cls.__HOUR_LABEL)

        return cls._RATE_TEMPLATE.format(n, cls.__DAY_LABEL)

    def status_printer(
        self, file: io.TextIOWrapper | io.StringIO
    ) -> Callable[[str], None]:
        """Overload the status_printer method to avoid the use of closures.

        Args:
            file: Specifies where to output the progress messages.

        Returns:
            The function to print the status in the progress bar.
        """
        self._last_len = [0]
        return self._print_status

    def _print_status(self, s: str) -> None:
        s_length = disp_len(s)
        self.fp.write(_unicode(f"\r{s}{' ' * max(self._last_len[0] - s_length, 0)}"))
        self.fp.flush()
        self._last_len[0] = s_length

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # A file-like stream cannot be pickled.
        del state[self.__FP_LABEL]
        return state

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        # Set back the file-like stream to its state as done in tqdm.__init__.
        self.fp = tqdm.utils.DisableOnWriteError(TqdmToLogger(), tqdm_instance=self)
