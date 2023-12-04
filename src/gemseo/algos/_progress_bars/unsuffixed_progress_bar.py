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
"""A progress bar not suffixed by metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.algos._progress_bars.progress_bar import ProgressBar

if TYPE_CHECKING:
    from numpy import ndarray


class UnsuffixedProgressBar(ProgressBar):
    """A progress bar not suffixed by metadata."""

    def _set_objective_value(self, x_vect: ndarray | None) -> None:
        self._tqdm_progress_bar.n += 1
        self._tqdm_progress_bar.set_postfix()
