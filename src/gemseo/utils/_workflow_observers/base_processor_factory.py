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
"""Factory for creating workflow observer processors."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeVar

from gemseo.core.base_factory import BaseFactory
from gemseo.utils._workflow_observers.base_processor import BaseProcessor

if TYPE_CHECKING:
    from gemseo.utils._workflow_observers.base_observer import BaseWorkflowObserver
    from gemseo.utils._workflow_observers.interface import CallArguments

T = TypeVar("T", bound=BaseProcessor)


class BaseProcessorFactory(BaseFactory[T]):
    """Factory to create processors."""

    def create(
        self,
        observer: BaseWorkflowObserver,
        init_arguments: CallArguments,
    ) -> T:
        """
        Args:
            observer: The observer to use.
            init_arguments: The arguments to pass to the processor constructor.

        Returns:
            The processor.
        """  # noqa: D205, D212
        for class_name in self.class_names:
            if isinstance(observer, self.get_class(class_name).observer_class):
                return super().create(class_name, observer, init_arguments)
        msg = f"No directory manager found for the observer {type(observer).__name__}."
        raise ValueError(msg)
