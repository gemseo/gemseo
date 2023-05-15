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
"""Progress bar."""
from __future__ import annotations

import io
import logging
import string
from typing import Any
from typing import Callable

import tqdm
from tqdm.utils import _unicode
from tqdm.utils import disp_len


LOGGER = logging.getLogger(__name__)


class TqdmToLogger(io.StringIO):
    """Redirect tqdm output to the gemseo logger."""

    def write(self, buf: str) -> None:
        """Write buffer."""
        buf = buf.strip(string.whitespace)
        if buf:
            LOGGER.info(buf)


class ProgressBar(tqdm.tqdm):
    """Extend tqdm progress bar with better time units.

    Use hour, day or week for slower processes.
    """

    @classmethod
    def format_meter(  # noqa: D102
        cls, n: float, total: float, elapsed: float, **kwargs: Any
    ) -> str:
        meter = tqdm.tqdm.format_meter(n, total, elapsed, **kwargs)
        if elapsed != 0.0:
            rate, unit = cls.__convert_rate(n, elapsed)
            lstr = meter.split(",")
            lstr[1] = f" {rate:5.2f}{unit}"
            meter = ",".join(lstr)
        # remove the unit suffix that is hard coded in tqdm
        return meter.replace("/s,", ",").replace("/s]", "]")

    @staticmethod
    def __convert_rate(total, elapsed):
        rps = float(total) / elapsed
        if rps >= 0:
            rate = rps
            unit = "sec"

        rpm = rps * 60.0
        if rpm < 60.0:
            rate = rpm
            unit = "min"

        rph = rpm * 60.0
        if rph < 60.0:
            rate = rph
            unit = "hour"

        rpd = rph * 24.0
        if rpd < 24.0:
            rate = rpd
            unit = "day"

        return rate, f" it/{unit}"

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
        len_s = disp_len(s)
        self.fp.write(
            _unicode("\r{}{}".format(s, (" " * max(self._last_len[0] - len_s, 0))))
        )
        self.fp.flush()
        self._last_len[0] = len_s

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # A file-like stream cannot be pickled.
        del state["fp"]
        return state

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        # Set back the file-like stream to its state as done in tqdm.__init__.
        self.fp = tqdm.utils.DisableOnWriteError(TqdmToLogger(), tqdm_instance=self)
