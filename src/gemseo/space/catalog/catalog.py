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
"""The catalog of alternatives."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import starmap
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any
from typing import Final

from numpy import array
from numpy import array_equal
from numpy import flatnonzero
from numpy import ndarray
from numpy.char import encode
from pandas import DataFrame
from pydantic import BaseModel
from pydantic import Field
from pydantic import WithJsonSchema
from pydantic import model_validator

from gemseo.space.catalog._equality import properties_are_equal
from gemseo.space.catalog._hdf import find_hdf_writing_error
from gemseo.space.catalog._hdf import find_unencodable_utf8_value
from gemseo.space.catalog._hdf import labels_group
from gemseo.space.catalog._hdf import properties_group
from gemseo.space.catalog._hdf import property_names_group
from gemseo.space.catalog._hdf import read_hdf_property
from gemseo.space.catalog._hdf import was_unicode_attribute
from gemseo.space.catalog._input import CatalogFormatter
from gemseo.space.catalog._input import CatalogInput
from gemseo.space.catalog._input import freeze_property_array
from gemseo.space.catalog._input import labels_field
from gemseo.space.catalog._input import properties_field
from gemseo.space.catalog._input import raise_errors
from gemseo.space.variable._formatting import format_elided
from gemseo.space.variable._view_field import ArrayViewField
from gemseo.space.variable._view_field import expose_fields_as_views
from gemseo.util.read_only_mapping import ReadOnlyMapping
from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from typing import Self

    from h5py import Group


_max_formatted_labels: Final[int] = 6
"""The number of labels above which a rendering is elided."""


class Catalog(BaseModel, arbitrary_types_allowed=True, frozen=True, extra="forbid"):
    """A catalog of alternatives.

    A catalog is a table:
    one row per alternative,
    one column per property of the alternatives,
    and one label naming each alternative.
    It is the domain of a
    [CatalogVariable][gemseo.space.variable.catalog.CatalogVariable],
    whose value is the **position** of an alternative, never its label.

    **Building.** A catalog can be built:

    - from a mapping from a property name, which must be a string,
      to a sequence of values or a scalar, which is repeated over the rows,
      e.g. `Catalog(properties={"density": [2.7, 7.8], "cost": 10.0})`;
    - from a [DataFrame][pandas.DataFrame],
      e.g. `Catalog(properties=pandas.read_csv("materials.csv", index_col=0))`,
      whose index gives the labels, unless `labels` is passed,
      and whose column names are converted to strings,
      which must remain distinct;
      a `DataFrame` whose row or column axis is a `MultiIndex` is rejected,
      since a catalog is a flat table;
    - from a mapping with the fields of a catalog,
      through `Catalog.model_validate`;
    - from an HDF group,
      through [read_hdf][gemseo.space.catalog.catalog.Catalog.read_hdf];
    - from another catalog,
      through [model_copy][gemseo.space.catalog.catalog.Catalog.model_copy]
      with new properties or new labels.

    The labels are optional: without them,
    the alternatives are labelled by their positions, `"0"`, `"1"`, ...
    A `str`, `bytes`, `bytearray` or zero-dimensional array is a single label,
    so `labels="alu"` builds a one-row catalog labelled `"alu"`,
    and `labels=""` names a single row rather than meaning no labels.
    Duplicate labels are accepted,
    since an alternative is chosen by its position.

    [DesignSpace.add_catalog_variable][gemseo.space.design.DesignSpace.add_catalog_variable]
    and [CatalogVariable][gemseo.space.variable.catalog.CatalogVariable]
    also accept the properties of a catalog in place of a catalog,
    and build it themselves.

    **Formatting.** Whatever the way it is built, the input is formatted:

    - the leading and trailing whitespace of every string is removed,
      in the properties and in the labels,
      and a byte-string label is decoded from UTF-8;
    - a property holding no value,
      i.e. whose every element is a missing value or an empty string,
      is dropped with a log warning;
      `0` and `False` are values, so their properties are kept;
    - a scalar property is repeated over the rows;
    - the index of a `Series` given as a property is ignored,
      since a property is read positionally;
    - the data type of a property is inferred by NumPy from its values,
      even when the input has the `object` data type,
      so a property mixing strings with other values becomes a property of strings,
      e.g. `[1, "1"]` becomes `["1", "1"]`,
      whose two alternatives are then equal;
      only strings mixed with missing values remain Python objects.
      Pass properties of a single type to keep such alternatives distinct.

    The input is rejected
    if a property is an unordered collection, e.g. a `set`, or a `DataFrame`,
    if a property, or the labels, is not one-dimensional,
    if the properties and the labels do not have the same number of elements,
    if there is no row or no property left,
    if two column names of a `DataFrame` are equal once converted to strings,
    or if a byte-string label is not valid UTF-8.

    The `properties` and `labels` fields are typed as what a catalog holds,
    a read-only mapping from the names of the properties to frozen arrays
    and a frozen array of strings,
    while the constructor also accepts the input types described above.
    The names of the properties, in order, are the keys of `properties`.

    **Immutability.** A catalog is immutable:
    the input is copied, deeply for a property of Python objects,
    its arrays are frozen and handed out as fresh views on every read,
    and [to_dataframe][gemseo.space.catalog.catalog.Catalog.to_dataframe]
    rebuilds a fresh table on each call,
    so mutating either the input or the result cannot change the catalog.
    Copying one hands the catalog itself back,
    `model_copy` without an update included,
    which departs from the pydantic contract
    but shares nothing that could be mutated.

    **Serialization.** A catalog is written to and read from HDF, and nothing else:
    it holds NumPy arrays, which pydantic cannot serialize to JSON,
    so `model_dump_json` raises
    and `model_dump` hands back the arrays as they are.
    """

    # The JSON schemas of the properties and the labels are those of their input
    # types, since pydantic cannot generate one for the types they hold.
    properties: Annotated[
        ReadOnlyMapping[str, ndarray],
        WithJsonSchema({"type": "object", "additionalProperties": True}),
    ] = Field(
        description="""The properties of the alternatives, by name, in order.

A `DataFrame` or a mapping from a property name to a sequence or a scalar
is accepted at construction, and formatted into frozen arrays.""",
    )

    labels: Annotated[
        ndarray,
        WithJsonSchema({"anyOf": [{"type": "array", "items": {}}, {"type": "string"}]}),
    ] = Field(
        # The default is an input, read as the absence of labels,
        # which the formatting replaces with the positions of the alternatives.
        default=(),
        description="""The names of the alternatives, as an array of strings.

A sequence, an array or a pandas `Index` is accepted at construction.
If empty at construction,
the positions of the alternatives are used.
A `str`, `bytes`, `bytearray` or zero-dimensional array is read as a
single label, not as a sequence of its characters, bytes or, for a
zero-dimensional array, of its one element; an empty string is one
label naming a single row, not an absence of labels.""",
    )

    @model_validator(mode="before")
    @classmethod
    def __validate_catalog(cls, data: Any) -> Any:
        """Format the catalog and freeze it.

        The input is first checked against the input types of the fields,
        then formatted into the values the fields hold
        by [CatalogFormatter][gemseo.space.catalog._input.CatalogFormatter].

        Args:
            data: The input of the catalog.

        Returns:
            The input with the labels and the properties formatted and frozen.

        Raises:
            ValidationError: If the properties are missing,
                or if they are neither a `DataFrame` nor a mapping
                from a string,
                or if the labels are neither a sequence, an array
                nor a pandas `Index`.
            ValueError: If the input cannot be formatted.
        """
        if not isinstance(data, Mapping):
            # E.g. an integer, which pydantic rejects with a ValidationError.
            return data

        data = dict(data)
        # Check the input against its types, with the locations and the title
        # that a field annotated with these types would give to the errors.
        catalog_input = CatalogInput.model_validate(data)
        data[labels_field], data[properties_field] = CatalogFormatter.format(
            catalog_input.properties, catalog_input.labels
        )
        return data

    @property
    def _property_names(self) -> tuple[str, ...]:
        """The names of the properties, in order.

        The order takes part in the identity of a catalog:
        it drives equality, hashing and the layout on file.
        """
        return tuple(self.properties)

    def __len__(self) -> int:
        return len(self.labels)

    def to_dataframe(self) -> DataFrame:
        """Return the catalog as a table.

        The table is rebuilt on each call
        and shares no data with the catalog,
        so mutating it cannot change the catalog.

        Returns:
            The catalog as a table.
        """
        return DataFrame(dict(self.properties), index=self.labels, copy=True)

    def format_labels(self, max_length: int = _max_formatted_labels) -> str:
        """Return a readable representation of the labels.

        A long list is elided around its extremes and followed by its length,
        so that a tabular view keeps one line per component.

        Args:
            max_length: The number of labels above which
                the representation is elided.
                A value below `1` is read as `1`,
                since an elided representation keeps at least one label.

        Returns:
            The representation of the labels,
            e.g. `"[aluminium, steel, titanium]"`.
        """
        return format_elided(self.labels, "labels", max_length)

    def get_position(self, label: str) -> int:
        """Return the position of the alternative named by a label.

        The label is stripped of its leading and trailing whitespace,
        as the labels of the catalog are,
        then compared exactly, case included, with each of them.
        A catalog built without labels is labelled by the positions,
        so that `"1"` then names the alternative at position `1`.

        Args:
            label: The label of the alternative.

        Returns:
            The position of the alternative.

        Raises:
            ValueError: If no alternative or several alternatives
                have this label.
        """
        label = label.strip()
        positions = flatnonzero(self.labels == label)
        if len(positions) == 1:
            return int(positions[0])

        if not len(positions):
            msg = (
                f"The catalog has no alternative labelled {label!r}; "
                f"its labels are {self.format_labels()}. "
                "Pass the position of an alternative to choose it by row."
            )
            raise ValueError(msg)

        msg = (
            f"The label {label!r} names several alternatives of the catalog, "
            f"at positions {pretty_str(positions.tolist(), sort=False)}; "
            "pass the position of the alternative to choose one of them."
        )
        raise ValueError(msg)

    def __eq__(self, other: object) -> bool:
        # The default implementation of pydantic compares the __dict__ of the models,
        # which raises on a mapping of arrays;
        # compare the labels and the properties explicitly instead.
        # Return False rather than NotImplemented for a foreign type, which
        # gives up the reflected comparison but keeps BaseVariable.__eq__
        # safe: it reads the comparison of two fields as a plain truth value,
        # and NotImplemented is truthy, so it would read as equal.
        if not isinstance(other, Catalog):
            return False

        if self._property_names != other._property_names:
            return False

        if not array_equal(self.labels, other.labels):
            return False

        other_properties = other.properties
        return all(
            properties_are_equal(property_values, other_properties[name])
            for name, property_values in self.properties.items()
        )

    def __repr__(self) -> str:
        # The default implementation of pydantic renders every value of every
        # property, e.g. 13 kB for a thousand rows, which drowns a traceback or
        # a debugger view; render the shape of the table instead.
        n_rows = len(self)
        plural = "" if n_rows == 1 else "s"
        return (
            f"{self.__class__.__name__}("
            f"{n_rows} alternative{plural} "
            f"{self.format_labels()}, "
            f"properties: {pretty_str(self._property_names, sort=False)})"
        )

    # Pydantic's default __str__ renders every value of every property, same as its
    # default __repr__; reuse the short __repr__ above for str(), print() and f-strings
    # too, so none of them fall back to that default.
    __str__ = __repr__

    def __hash__(self) -> int:
        # Pydantic gives a frozen model a hash over the values of its fields,
        # which raises on the mapping of arrays the properties field holds.
        # Hash only what two equal catalogs are guaranteed to share: the
        # values cannot take part, since equal properties may differ in dtype,
        # e.g. an int8 and an int64 property of the same values, and equal
        # values may differ in bytes, e.g. 0.0 and -0.0.
        return hash((self._property_names, tuple(self.labels)))

    def __copy__(self) -> Self:
        # A catalog is immutable and its arrays are read-only,
        # so a copy can be shared with the original.
        # This also keeps the arrays frozen,
        # since NumPy does not preserve the writeable flag across a copy.
        return self

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        return self

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Return a copy of the catalog, updated with new field values.

        Args:
            update: The new field values, if any.
            deep: Whether to deep-copy the catalog;
                this has no effect since the properties and the labels are
                copied by the validation.

        Returns:
            The catalog itself without an update, otherwise a new catalog.
            New properties without new labels keep the labels of the catalog,
            except a `DataFrame`, labelled by its index,
            and except for a catalog labelled by the positions,
            labelled by the new positions.
        """
        # The base implementation writes the update into the __dict__ of the object
        # returned by __copy__/__deepcopy__, which is this very instance;
        # rebuild through validation instead, so that the original is left alone
        # and the new properties and labels are converted, checked and frozen.
        if not update:
            return self

        payload = dict(update)
        if labels_field not in update and not self.__update_relabels(update):
            payload[labels_field] = self.__dict__[labels_field]

        payload.setdefault(properties_field, self.__dict__[properties_field])
        return self.model_validate(payload)

    def __update_relabels(self, update: Mapping[str, Any]) -> bool:
        """Check whether new properties without new labels relabel the catalog.

        Args:
            update: The new field values.

        Returns:
            Whether the update replaces the properties
            and either the new properties are a `DataFrame`,
            whose index names the alternatives,
            or the catalog is labelled by the positions,
            which then follow the new number of alternatives.
        """
        if properties_field not in update:
            return False

        if isinstance(update[properties_field], DataFrame):
            return True

        return self.labels.tolist() == [str(index) for index in range(len(self))]

    def check_hdf_writable(self) -> None:
        """Check that the catalog can be written to an HDF file.

        Raises:
            ValueError: If a property name is not a valid HDF5 link name,
                i.e. is empty, is `"."` or `".."`, or contains a slash or a
                NUL character, which would make it an invalid or ambiguous
                HDF path, if a property name cannot be encoded to UTF-8,
                if a property holds Python objects, if a property has a dtype
                that an HDF file cannot store, e.g. a `datetime64` or a
                `timedelta64` dtype, if a value of a `"U"` property cannot be
                encoded to UTF-8, or if a label cannot be encoded to UTF-8;
                every offender is named at once, one reason per property.
        """
        error_messages = [
            message
            for message in starmap(find_hdf_writing_error, self.properties.items())
            if message
        ]
        if (label := find_unencodable_utf8_value(self.labels)) is not None:
            error_messages.append(
                f"The label {label!r} of the catalog cannot be encoded to UTF-8 "
                "and cannot be written to an HDF file."
            )

        raise_errors("The catalog cannot be written to an HDF file:", error_messages)

    def write_hdf(self, group: Group) -> None:
        """Write the catalog to an HDF group.

        Args:
            group: The HDF group.

        Raises:
            ValueError: If a property name is not a valid HDF5 link name,
                i.e. is empty, is `"."` or `".."`, or contains a slash or a
                NUL character, if a property name cannot be encoded to UTF-8,
                if a property holds Python objects, if a property has a dtype
                that an HDF file cannot store, if a value of a `"U"` property
                cannot be encoded to UTF-8, or if a label cannot be encoded
                to UTF-8; every offender is named at once.
        """
        self.check_hdf_writable()
        group.create_dataset(labels_group, data=encode(self.labels, "utf-8"))
        # The names are stored explicitly because HDF does not preserve
        # the insertion order of the members of a group,
        # while the order of the properties takes part in the equality of two catalogs.
        group.create_dataset(
            property_names_group, data=encode(array(self._property_names), "utf-8")
        )
        hdf_properties = group.create_group(properties_group)
        for name, property_values in self.properties.items():
            was_unicode = property_values.dtype.kind == "U"
            data = encode(property_values, "utf-8") if was_unicode else property_values
            dataset = hdf_properties.create_dataset(name, data=data)
            dataset.attrs[was_unicode_attribute] = was_unicode

    @classmethod
    def read_hdf(cls, group: Group) -> Self:
        """Read a catalog from an HDF group.

        Args:
            group: The HDF group.

        Returns:
            The catalog.
        """
        hdf_properties = group[properties_group]
        return cls(
            properties={
                name.decode(): read_hdf_property(hdf_properties[name.decode()])
                for name in group[property_names_group][()]
            },
            labels=[label.decode() for label in group[labels_group][()]],
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        # NumPy pickles a frozen array as an independent, writeable array,
        # and pydantic restores the model without re-validating it,
        # so refreeze the arrays here rather than trusting the unpickled ones.
        self.__dict__[labels_field] = freeze_property_array(self.__dict__[labels_field])
        self.__dict__[properties_field] = ReadOnlyMapping({
            name: freeze_property_array(property_values)
            for name, property_values in self.__dict__[properties_field].items()
        })


class _PropertiesViewField(ArrayViewField):
    """A descriptor reading the properties field as views of the frozen properties.

    A [ReadOnlyMapping][gemseo.util.read_only_mapping.ReadOnlyMapping]
    cannot view the arrays it holds when one of them is looked up,
    so the mapping is rebuilt on every read,
    for the reason
    [ArrayViewField][gemseo.space.variable._view_field.ArrayViewField]
    gives for viewing an array on every read.
    """

    __slots__ = ()

    def __get__(self, instance: BaseModel | None, owner: type | None = None) -> Any:
        value = super().__get__(instance, owner)
        if isinstance(value, ReadOnlyMapping):
            return ReadOnlyMapping({
                name: property_values.view() for name, property_values in value.items()
            })

        # A catalog built with model_construct skips the validation,
        # so its properties are whatever the caller supplied.
        return value


# A property could not take the name of a field,
# so the labels and the properties are read as views of the frozen arrays stored
# through descriptors set once pydantic has built the class.
expose_fields_as_views(Catalog, labels_field)


setattr(Catalog, properties_field, _PropertiesViewField(properties_field))
