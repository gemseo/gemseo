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
"""Deprecated module for progress bar."""

from __future__ import annotations

from gemseo.algos._progress_bars.tqdm_to_logger import TqdmToLogger  # noqa: F401
from gemseo.algos._progress_bars.tqdm_to_logger import (  # noqa: F401
    TqdmToLogger as ProgressBar,
)

# TODO: API: remove this module in gemseo 6.0.0.
