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
"""A factory of gradient approximators."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.base_factory import BaseFactory
from gemseo.utils.derivatives.base_gradient_approximator import BaseGradientApproximator

if TYPE_CHECKING:
    from gemseo.utils.derivatives.approximation_modes import ApproximationMode


class GradientApproximatorFactory(BaseFactory):
    """A factory of gradient approximators."""

    _CLASS = BaseGradientApproximator
    _PACKAGE_NAMES = ("gemseo.utils.derivatives",)

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        for class_info in tuple(self._names_to_class_info.values()):
            approximation_mode = class_info.class_._APPROXIMATION_MODE
            self._names_to_class_info[approximation_mode] = class_info

    def create(
        self,
        name: ApproximationMode,
        *args: Any,
        **kwargs: Any,
    ) -> BaseGradientApproximator:
        """Create a gradient approximator.

        Args:
            name: The name of the class or the approximation mode.

        Returns:
            The gradient approximator.
        """
        return super().create(name, *args, **kwargs)
