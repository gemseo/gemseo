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
"""The base progress bar."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray


class BaseProgressBar(ABC):
    """The base progress bar."""

    @abstractmethod
    def set_objective_value(
        self, x_vect: ndarray | None, current_iter_must_not_be_logged: bool = False
    ) -> None:
        """Set the objective value.

        Args:
            x_vect: The design variables values.
                If ``None``, consider the objective at the last iteration.
            current_iter_must_not_be_logged: Set the objective value
                only if the current iteration is not logged.
        """

    @abstractmethod
    def finalize_iter_observer(self) -> None:
        """Finalize the iteration observer."""
