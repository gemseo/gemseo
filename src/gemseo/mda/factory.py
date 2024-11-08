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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A factory of MDAs."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Optional
from typing import Union

from pydantic_core import PydanticUndefined

from gemseo.core.base_factory import BaseFactory
from gemseo.core.coupling_structure import CouplingStructure
from gemseo.mda.base_mda import BaseMDA

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

MDAOptionType = Optional[
    Union[float, int, bool, str, Iterable[CouplingStructure], Sequence[BaseMDA]]
]


class MDAFactory(BaseFactory):
    """A factory of MDAs."""

    _CLASS = BaseMDA
    _PACKAGE_NAMES = ("gemseo.mda",)

    def get_options_doc(self, name: str) -> dict[str, str]:
        """Return the constructor documentation of a class.

        Args:
            name: The name of the class.

        Returns:
            The mapping from the argument names to their documentation.
        """
        fields = self.get_class(name).Settings.model_fields
        return {k: v.description for k, v in fields.items()}

    def get_default_option_values(self, name: str) -> StrKeyMapping:
        """Return the constructor kwargs default values of a class.

        Args:
            name: The name of the class.

        Returns:
            The mapping from the argument names to their default values.
        """
        fields = self.get_class(name).Settings.model_fields
        defaults = {}
        for field_name, field_default in fields.items():
            default_value = field_default.default
            if default_value == PydanticUndefined:
                factory = field_default.default_factory
                default_value = factory() if factory != PydanticUndefined else None

            defaults[field_name] = default_value
        return defaults
