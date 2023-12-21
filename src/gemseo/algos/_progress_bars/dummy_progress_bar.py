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
"""A dummy progress bar."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.algos._progress_bars.base_progress_bar import BaseProgressBar

if TYPE_CHECKING:
    from numpy import ndarray


class DummyProgressBar(BaseProgressBar):
    """A dummy progress bar.

    .. warning:: This progress bar is inactive.
    """

    def set_objective_value(  # noqa D102
        self, x_vect: ndarray | None, current_iter_must_not_be_logged: bool = False
    ) -> None: ...

    def finalize_iter_observer(self) -> None:  # noqa D102
        ...
