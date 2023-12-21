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
"""Redirection of the tqdm output."""

from __future__ import annotations

import io
import logging
import string

# TODO: API: remove this module in gemseo 6.0.0.

LOGGER = logging.getLogger(__name__)


class TqdmToLogger(io.StringIO):
    """Redirect the tqdm output to the GEMSEO logger."""

    def write(self, buffer_: str) -> None:
        """Log the buffer with INFO level.

        Args:
            buffer_: The buffer.
        """
        buffer_ = buffer_.strip(string.whitespace)
        # Do not log the initialization of the progress bar.
        if buffer_ and " 0%|" not in buffer_:
            LOGGER.info(buffer_)
