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
from functools import cached_property
from typing import ClassVar

from pandas import DataFrame
from pydantic import Field
from pydantic import field_validator

from gemseo.enum import VariableType
from gemseo.space.catalog._input import CatalogPropertiesType
from gemseo.space.catalog._input import properties_field
from gemseo.space.catalog.catalog import Catalog
from gemseo.space.variable.base_categorical import BaseCategoricalVariable


class CatalogVariable(BaseCategoricalVariable):
    """A scalar categorical variable defined by a catalog.

    Its domain is the alternatives of a
    [Catalog][gemseo.space.catalog.catalog.Catalog],
    and its labels are those of this catalog:
    a label names an alternative,
    and a property of the catalog holds a value per alternative.

    The catalog is the only input of the variable;
    its labels and its size, which is one, derive from it and are read-only.
    """

    type: ClassVar[VariableType] = VariableType.CATALOG

    catalog: Catalog = Field(
        description="The catalog of the alternatives that the variable can take; "
        "a table or a mapping is also accepted, "
        "as the properties to build the catalog from, "
        "or a mapping whose keys are exactly 'properties' and 'labels', "
        "with a table or a mapping under 'properties', "
        "as the fields of a catalog, e.g. its model_dump; "
        "the labels and the properties are kept as they are, "
        "so, to use cleaned data, pass a catalog cleaned with Catalog.clean, "
        "e.g. Catalog(properties=table).clean()."
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

    @cached_property
    def labels(self) -> tuple[str, ...]:
        """The labels of the catalog (read-only)."""
        return tuple(self.catalog.labels.tolist())

    def _format_labels(self) -> str:
        return self.catalog.format_labels()
