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
"""Catalog variable."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from numpy import array
from numpy import atleast_1d
from numpy import int64
from numpy import isfinite
from numpy import isreal
from numpy import mod
from pandas import DataFrame
from pydantic import Field
from pydantic import field_validator

from gemseo.space.catalog._input import CatalogPropertiesType
from gemseo.space.catalog._input import properties_field
from gemseo.space.catalog.catalog import Catalog
from gemseo.space.variable._formatting import format_components
from gemseo.space.variable.base import CoordinateDType
from gemseo.space.variable.base import DataType
from gemseo.space.variable.deterministic import BaseDeterministicVariable
from gemseo.space.variable.numeric import BaseNumericVariable
from gemseo.util._numpy import freeze_array

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from typing import Any
    from typing import Self

    from gemseo.space.variable.numeric import BoundArray
    from gemseo.util.typing import BooleanArray
    from gemseo.util.typing import NumberArray

_normalization_mask: Final[BooleanArray] = freeze_array(array([False]))
"""The normalization mask of a catalog variable (read-only).

A catalog variable is never normalized,
so this mask is the same for every catalog variable;
it is frozen once and for all,
and `CatalogVariable.get_normalization_mask` hands out views of it.
"""


class CatalogVariable(
    BaseDeterministicVariable, BaseNumericVariable, arbitrary_types_allowed=True
):
    """A scalar catalog variable.

    Its domain is the alternatives of a
    [Catalog][gemseo.space.catalog.catalog.Catalog],
    and its value is the **position** of an alternative in that catalog,
    from `0` to the number of alternatives minus one.

    The catalog is the only input of the variable;
    its size, which is one, and its bounds derive from it and are read-only.

    A label of the catalog names an alternative;
    a property of the catalog holds a value per alternative.
    """

    coordinate_type: ClassVar[CoordinateDType] = int64

    type: ClassVar[DataType] = DataType.CATALOG

    catalog: Catalog = Field(
        description="The catalog of the alternatives that the variable can take; "
        "a table or a mapping is also accepted, "
        "as the properties to build the catalog from, "
        "or a mapping whose keys are exactly 'properties' and 'labels', "
        "with a table or a mapping under 'properties', "
        "as the fields of a catalog, e.g. its model_dump."
    )

    @field_validator("catalog", mode="before")
    @classmethod
    def __convert_catalog(cls, value: Catalog | CatalogPropertiesType) -> Catalog:
        """Build a catalog from a table or a mapping.

        A mapping whose keys are exactly the fields of a catalog,
        `"properties"` and `"labels"`,
        and whose `"properties"` value is itself a table or a mapping,
        is read as the fields of a catalog, e.g. a `model_dump` of a catalog,
        so that a dumped catalog variable is validated back into the same one;
        any other mapping is read as the properties of a catalog.
        Pass a [Catalog][gemseo.space.catalog.catalog.Catalog]
        to build one whose properties are named that way.

        Args:
            value: The catalog, its fields, or the properties to build it from.

        Returns:
            The catalog.

        Raises:
            ValidationError: If the catalog cannot be built from the value.
        """
        if isinstance(value, Catalog):
            return value

        if (
            isinstance(value, Mapping)
            and value.keys() == Catalog.model_fields.keys()
            and isinstance(value[properties_field], (DataFrame, Mapping))
        ):
            return Catalog.model_validate(value)

        # Any other table or mapping is taken as the properties of a catalog,
        # so that the common case needs neither an extra import
        # nor an extra keyword;
        # pass a Catalog or the fields of a catalog to also supply the labels.
        return Catalog(properties=value)

    @property
    def size(self) -> int:
        """The size of the variable."""
        return 1

    @property
    def lower_bound(self) -> BoundArray:
        """The lower bound of the variable (read-only)."""
        return self.__create_bound(0)

    @property
    def upper_bound(self) -> BoundArray:
        """The upper bound of the variable (read-only)."""
        return self.__create_bound(len(self.catalog) - 1)

    def __create_bound(self, position: int) -> BoundArray:
        """Create a bound of the variable from a position in its catalog.

        The bound array is built on demand and handed out read-only.

        Args:
            position: The position in the catalog defining the bound.

        Returns:
            The bound of the variable.
        """
        return freeze_array(array([position], dtype=self.coordinate_type))

    def get_normalization_mask(  # noqa: D102
        self, enable_integer_normalization: bool
    ) -> BooleanArray:
        # A view, so that reassigning the shape, the strides or the data type
        # of the mask handed out, which NumPy allows on a read-only array,
        # cannot reach the mask shared by every catalog variable.
        return _normalization_mask.view()

    def find_components_outside_domain(self, value: NumberArray) -> set[int]:  # noqa: D102
        value_0 = atleast_1d(value)[0]
        if value_0 is None:
            return set()

        # A label, or any other non-numeric value, is never a valid position
        # in the catalog; return early to avoid the raw numpy error that
        # `.real` or `isfinite` would otherwise raise on it.
        if not isreal(value_0):
            return {0}

        value_0_real = value_0.real
        # An infinite or NaN position is never a valid position in the catalog;
        # return early to avoid calling mod on it,
        # which would otherwise emit a spurious RuntimeWarning
        # (as opposed to floor, see IntegerVariable.check_finite_bound_components).
        if not isfinite(value_0_real):
            return {0}

        if mod(value_0_real, 1) == 0 and 0 <= value_0_real <= len(self.catalog) - 1:
            return set()

        return {0}

    def get_default_value(self) -> NumberArray:
        """
        Returns:
            The position of the first alternative of the catalog.
        """  # noqa: D205, D212
        return array([0], dtype=self.coordinate_type)

    def filter_components(self, components: Sequence[int]) -> Self:  # noqa: D102
        return self._filter_scalar_components(components)

    def _get_out_of_domain_message(
        self, name: str, value: NumberArray, indices: Iterable[int]
    ) -> str:
        # A catalog variable is scalar, so its only caller,
        # `check_addable_value`, can never pass more than the single component
        # `find_components_outside_domain` may return; no plural wording is needed.
        return (
            f"The following value of variable '{name}' "
            f"is not a position in its catalog with the labels "
            f"{self.catalog.format_labels()}: "
            f"{format_components(value, indices)}."
        )

    def _get_out_of_domain_component_message(
        self, name: str, index: int, value_i: Any
    ) -> str:
        return (
            f"The variable {name} is a catalog variable "
            f"with the labels {self.catalog.format_labels()}; "
            f"got {name}[{index}] = {value_i}."
        )
