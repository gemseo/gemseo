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
"""The input of a catalog and its formatting into the arrays it holds."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from copy import deepcopy
from typing import Any
from typing import Final
from typing import NoReturn

from numpy import asarray
from numpy import full
from numpy import ndarray
from numpy import str_
from numpy.char import strip as strip_array
from pandas import DataFrame
from pandas import Index
from pandas import MultiIndex
from pandas import isna
from pydantic import BaseModel
from pydantic import Field

from gemseo.util._numpy import freeze_array
from gemseo.util.read_only_mapping import ReadOnlyMapping
from gemseo.util.string import pretty_repr
from gemseo.util.string import pretty_str

logger = logging.getLogger(__name__)


labels_field: Final[str] = "labels"
"""The name of the field storing the labels of a catalog."""


properties_field: Final[str] = "properties"
"""The name of the field storing the properties of a catalog."""


CatalogPropertiesType = DataFrame | Mapping[str, Any]
"""The type of the input properties of a catalog."""


CatalogLabelsType = Sequence[Any] | ndarray | Index
"""The type of the input labels of a catalog."""


class CatalogInput(BaseModel, arbitrary_types_allowed=True):
    """The input of a catalog, checked against its types before it is formatted.

    The fields of a [Catalog][gemseo.space.catalog.catalog.Catalog] are typed as what
    it holds, so its input is checked against these input types first, which
    gives the errors the locations a field annotated with them would give.
    """

    properties: CatalogPropertiesType = Field(description="The input properties.")

    labels: CatalogLabelsType = Field(default=(), description="The input labels.")


class CatalogFormatter:
    """The formatting of the input of a catalog into the arrays it holds.

    The data are read as they are:
    cleaning them is the job of
    [clean_catalog_data][gemseo.space.catalog._input.clean_catalog_data].

    The steps are applied in this order,
    which is part of the contract
    because each step changes what the next one sees:

    1. **Convert**: read the labels and the properties from the input.
       A `str`, `bytes`, `bytearray` or zero-dimensional array given as
       `labels` is read as a single label, not as a sequence of its
       characters or bytes, the same way such a value is read as a single
       scalar property.
       A `DataFrame` whose row or column axis is a `MultiIndex` is rejected,
       since a catalog is a flat table.
       A `DataFrame` whose column names collide once converted to strings,
       e.g. the integer `1` and the string `"1"`,
       is rejected here,
       before the collision could silently drop one of the properties,
       the same way two properties literally named alike are rejected here too,
       rather than later when a duplicate would otherwise only be caught
       as a two-dimensional property.
    2. **Check the kinds**: a property given as an unordered collection,
       e.g. a `set`, is rejected,
       because the order of the rows it would define is arbitrary,
       and so is a property given as a nested `DataFrame`,
       which is iterable over its column names rather than over values.
    3. **Normalize the strings**, in the properties and in the labels,
       without changing their characters:
       a zero-dimensional string array is unwrapped into the string it holds,
       a `bytearray` is converted into `bytes`,
       whether it is a scalar or an element of a sequence,
       and a sequence other than a NumPy array of a non-object dtype,
       e.g. a tuple, a `Series` or an array of Python objects,
       is converted into a list of its elements,
       which reveals the true shape of a `DataFrame` column
       of equal-length sequences;
       then a byte-string label, given as labels
       or as the index of a `DataFrame`, is decoded from UTF-8,
       the encoding a catalog writes its labels with to an HDF file;
       a byte-string label that is not valid UTF-8 is rejected.
    4. **Check the dimensionality**: every property, and the labels,
       must be one-dimensional,
       which rules out, in particular,
       a property, or labels, built from a nested sequence.
       A `DataFrame` with duplicate column labels would fail the same check,
       since such a label reads as a sub-frame rather than a column,
       but it is already rejected in the convert step, as two column labels
       that collide once converted to strings.
       This runs after the normalization,
       because normalizing a `DataFrame` column of equal-length sequences
       turns it from an opaque 1-D object array
       into a plain nested list that reveals its true shape.
       The labels must also be distinct once converted to strings,
       whitespace included,
       since a label names a single alternative.
    5. **Check the rectangularity**: every sequence property, and the labels,
       must hold the same number of elements,
       which defines the number of rows;
       the input is rejected if there is no row,
       i.e. if a property is empty:
       when every sequence property is empty
       and there are no labels, or empty labels,
       the error states that a catalog must have at least one row,
       while otherwise, another property or the labels naming at least one row,
       the error states that the properties do not have the same length,
       since the empty property contradicts them.
    6. **Broadcast the scalars**: a scalar property is repeated over the rows.
    7. **Check the non-emptiness**: there must be at least one property.
    8. **Freeze**: every array is copied and made read-only,
       and a property of Python objects is deep-copied,
       so that the catalog shares nothing mutable with the caller.
       The data type of a property is inferred by NumPy from its values,
       unless the property mixes strings and missing values,
       which is kept as Python objects.

    The errors of the steps 1 to 4 are reported together, in one pass,
    since none of these steps needs the others to succeed;
    a property or a column reported by a step is left out of the next ones,
    so that it is reported once.
    The next steps rely on a well-formed input,
    so each of them reports its own errors.
    """

    @classmethod
    def format(
        cls, properties: CatalogPropertiesType, raw_labels: CatalogLabelsType
    ) -> tuple[ndarray, ReadOnlyMapping[str, ndarray]]:
        """Format the input of a catalog and freeze it.

        Args:
            properties: The input properties.
            raw_labels: The input labels.

        Returns:
            The frozen labels and the frozen properties.

        Raises:
            ValueError: If the input is not a flat table,
                or if two of its column labels are equal once converted to
                strings,
                or if a property is an unordered collection or a table,
                or if a byte-string label is not valid UTF-8,
                or if a property, or the labels, is not one-dimensional,
                or if two labels are equal once converted to strings,
                naming every such error at once,
                or if there is no row,
                or if the properties and the labels do not have the same length,
                or if there is no property.
        """
        labels, properties, error_messages = cls.__convert(properties, raw_labels)
        error_messages.extend(cls.__find_property_kind_errors(properties))
        # A property of a kind no value can be read from is already reported,
        # so leave it out of the next steps rather than report it again.
        properties = {
            name: property_values
            for name, property_values in properties.items()
            if not isinstance(property_values, (DataFrame, AbstractSet))
        }
        labels, decoding_error_messages = _decode_labels(_normalize_strings(labels))
        error_messages.extend(decoding_error_messages)
        properties = {
            name: _normalize_strings(property_values)
            for name, property_values in properties.items()
        }
        # Undecoded labels are already reported,
        # and NumPy cannot read byte strings that are not ASCII.
        error_messages.extend(
            cls.__find_dimension_errors(
                None if decoding_error_messages else labels, properties
            )
        )
        # Undecoded or multi-dimensional labels are already reported.
        if (
            labels is not None
            and not decoding_error_messages
            and asarray(labels).ndim == 1
        ):
            error_messages.extend(_find_duplicate_label_errors(labels))

        # These steps do not depend on one another's success,
        # so their errors are reported together, in one pass,
        # whereas the next steps rely on a well-formed input.
        raise_errors("The input of the catalog is invalid:", error_messages)
        n_rows = cls.__compute_row_count(labels, properties)
        properties = {
            name: full(n_rows, property_values)
            if _is_scalar(property_values)
            else property_values
            for name, property_values in properties.items()
        }
        if not properties:
            msg = "A catalog must have at least one property."
            raise ValueError(msg)

        if labels is None:
            labels = [str(index) for index in range(n_rows)]

        # Freeze the validated data so that an accidental in-place mutation
        # cannot make the catalog and the bounds derived from it disagree.
        # The frozen arrays are stored as they are: the descriptors set on the
        # two fields of Catalog at the end of its module hand out a fresh view
        # of each of them on every read, so a caller never gets a hand on the
        # array the catalog stores (see BaseIntervalVariable.__convert_bound and
        # DiscreteVariable for the same precaution).
        return _freeze(labels, str_), ReadOnlyMapping({
            name: _freeze(property_values)
            for name, property_values in properties.items()
        })

    @staticmethod
    def __convert(
        properties: CatalogPropertiesType, raw_labels: CatalogLabelsType
    ) -> tuple[list[Any] | None, dict[str, Any], list[str]]:
        """Read the labels and the properties from the input.

        A `str`, `bytes`, `bytearray` or zero-dimensional array given as the
        labels is read as a single label, using the same rule
        [_is_scalar][gemseo.space.catalog._input._is_scalar] uses for a property, so
        that neither is silently exploded into one row per character, per
        byte or, for a zero-dimensional array, read through a bare `len`
        that raises on it.

        Args:
            properties: The input properties.
            raw_labels: The input labels.

        Returns:
            The labels, `None` if not supplied,
            the properties, without those whose names collide,
            and the error messages,
            naming whether the input is not a flat table,
            whether it has no row,
            and the column labels that are equal once converted to strings,
            which includes two column labels that are already equal.
        """
        if _is_scalar(raw_labels):
            labels = [raw_labels]
        elif len(raw_labels):
            labels = list(raw_labels)
        else:
            labels = None
        if not isinstance(properties, DataFrame):
            # The CatalogPropertiesType annotation of CatalogInput.properties
            # already guarantees that the keys are strings.
            return labels, dict(properties), []

        error_messages = []
        nested_axes = tuple(
            axis
            for axis, index in (
                ("row", properties.index),
                ("column", properties.columns),
            )
            if isinstance(index, MultiIndex)
        )
        if nested_axes:
            plural = len(nested_axes) > 1
            msg = (
                "A catalog is a flat table, but its "
                f"{pretty_str(nested_axes, sort=False)} "
                f"{'axes are' if plural else 'axis is'} a MultiIndex."
            )
            error_messages.append(msg)

        if not len(properties.index):
            error_messages.append("A catalog must have at least one row.")

        if labels is None:
            # A byte-string label is left as is, to be decoded from UTF-8
            # by _decode_labels like a byte-string label given as labels,
            # rather than turned into its representation, e.g. "b'alu'".
            labels = [
                label if isinstance(label, (bytes, bytearray)) else str(label)
                for label in properties.index
            ]

        str_names = tuple(str(name) for name in properties.columns)
        seen_names = set()
        duplicate_names = []
        for str_name in str_names:
            if str_name not in seen_names:
                seen_names.add(str_name)
            elif str_name not in duplicate_names:
                duplicate_names.append(str_name)

        if duplicate_names:
            plural = len(duplicate_names) > 1
            msg = (
                "The properties of the catalog do not have distinct names "
                f"once converted to strings: "
                f"{pretty_repr(duplicate_names, sort=False)} "
                f"{'each name' if plural else 'names'} more than one property."
            )
            error_messages.append(msg)

        # A name shared by several columns reads as a sub-frame,
        # which would also be reported as a two-dimensional property,
        # so these columns are left out once reported.
        return (
            labels,
            {
                str_name: properties[name].to_numpy()
                for str_name, name in zip(str_names, properties.columns, strict=False)
                if str_name not in duplicate_names
            },
            error_messages,
        )

    @staticmethod
    def __find_property_kind_errors(properties: Mapping[str, Any]) -> list[str]:
        """Find the properties of a kind a property cannot be read from.

        A `set` is iterable and sized,
        so it would otherwise be read as a sequence of values
        whose order, and hence the order of the rows of the catalog,
        would be arbitrary.
        A view over the keys or the items of a mapping is a `Set` too,
        and is rejected with it:
        it does follow the insertion order of its mapping,
        but a `Set` defined elsewhere need not,
        and the message tells the caller to pass an ordered sequence.

        A `DataFrame` is iterable over its **column names**,
        so a nested table would otherwise be read,
        silently and without any warning,
        as a property holding those names.

        Args:
            properties: The properties.

        Returns:
            The error messages, one per property
            that is an unordered collection or a table.
        """
        error_messages = []
        for name, property_values in properties.items():
            if isinstance(property_values, DataFrame):
                error_messages.append(
                    f"The property {name!r} of the catalog is a DataFrame; "
                    "a catalog is a flat table, "
                    "so a property holds values, not another table."
                )
            elif isinstance(property_values, AbstractSet):
                error_messages.append(
                    f"The property {name!r} of the catalog is an unordered collection "
                    f"({type(property_values).__name__}); "
                    "use an ordered sequence instead."
                )

        return error_messages

    @staticmethod
    def __find_dimension_errors(
        labels: list[Any] | None, properties: Mapping[str, Any]
    ) -> list[str]:
        """Find the properties, and the labels, that are not one-dimensional.

        A nested sequence yields a multi-dimensional array,
        which would otherwise go unnoticed
        and only fail later, in
        [to_dataframe][gemseo.space.catalog.catalog.Catalog.to_dataframe],
        for a property,
        or in [write_hdf][gemseo.space.catalog.catalog.Catalog.write_hdf]
        and [__hash__][gemseo.space.catalog.catalog.Catalog.__hash__],
        for the labels.
        A `DataFrame` with duplicate column labels would raise the same way,
        since it yields a sub-frame rather than a `Series`,
        but `__convert` already reports it and leaves it out,
        as two column labels that collide once converted to strings.

        Args:
            labels: The labels, `None` if not supplied.
            properties: The properties.

        Returns:
            The error messages, one per property that is not one-dimensional,
            followed by one if the labels are not one-dimensional.
        """
        error_messages = [
            f"The property {name!r} of the catalog "
            "has a dimension greater than 1; "
            "a property holds one value per row."
            for name, property_values in properties.items()
            if asarray(property_values).ndim > 1
        ]
        if labels is not None and asarray(labels).ndim > 1:
            error_messages.append(
                "The labels of the catalog have a dimension greater than 1."
            )

        return error_messages

    @staticmethod
    def __compute_row_count(
        labels: list[Any] | None, properties: Mapping[str, Any]
    ) -> int:
        """Compute the number of rows from the sized properties and the labels.

        Args:
            labels: The labels, `None` if not supplied.
            properties: The properties.

        Returns:
            The number of rows.

        Raises:
            ValueError: If the properties and the labels do not have the same length,
                or if there is no row.
        """
        name_to_length = _compute_property_lengths(properties)
        lengths = set(name_to_length.values())
        if labels is not None:
            lengths.add(len(labels))

        if len(lengths) > 1:
            _raise_length_mismatch(labels, name_to_length)

        # Without any sized property nor label, e.g. with scalar properties only,
        # nothing defines the number of rows,
        # but the catalog is still well formed and has a single row.
        n_rows = lengths.pop() if lengths else 1
        if not n_rows:
            msg = "A catalog must have at least one row."
            raise ValueError(msg)

        return n_rows


def raise_errors(header: str, error_messages: Sequence[str]) -> None:
    """Report every error found by a check step at once.

    A single error is reported as it stands,
    and several are reported as a bullet list under a header,
    so that a caller fixing an input sees every offender in one pass
    rather than one build attempt at a time.

    Args:
        header: The sentence introducing the list,
            used only when there is more than one error.
        error_messages: The errors, empty when the step found none.

    Raises:
        ValueError: If there is at least one error.
    """
    if not error_messages:
        return

    if len(error_messages) == 1:
        raise ValueError(error_messages[0])

    errors = "\n".join(f"- {error_message}" for error_message in error_messages)
    msg = f"{header}\n{errors}"
    raise ValueError(msg)


def _find_duplicate_label_errors(labels: Sequence[Any]) -> list[str]:
    """Find the labels naming several alternatives.

    The labels are compared once converted to strings,
    as the catalog stores them,
    so that `1` and `"1"` are the same label.

    Args:
        labels: The one-dimensional labels.

    Returns:
        The error message naming each duplicate label once,
        empty when the labels are distinct.
    """
    seen = set()
    duplicates = []
    for label in map(str, labels):
        if label not in seen:
            seen.add(label)
        elif label not in duplicates:
            duplicates.append(label)

    if not duplicates:
        return []

    plural = len(duplicates) > 1
    # The labels are quoted, so that their whitespace is visible.
    return [
        (
            f"The following label{'s name' if plural else ' names'} "
            "several alternatives of the catalog: "
            f"{pretty_str(map(repr, duplicates), sort=False)}."
        )
    ]


def _compute_property_lengths(properties: Mapping[str, Any]) -> dict[str, int]:
    """Return the number of elements of each sized property, by name.

    Args:
        properties: The properties.

    Returns:
        The number of elements of each property that is not a scalar, by name.
    """
    return {
        name: len(property_values)
        for name, property_values in properties.items()
        if not _is_scalar(property_values)
    }


def _raise_length_mismatch(
    labels: list[Any] | None, name_to_length: Mapping[str, int]
) -> NoReturn:
    """Reject properties and labels that do not describe the same number of rows.

    Args:
        labels: The labels, `None` if not supplied.
        name_to_length: The number of elements of each sized property, by name.

    Raises:
        ValueError: Always.
    """
    sizes = [f"{name} ({length})" for name, length in name_to_length.items()]
    if labels is not None:
        sizes.append(f"labels ({len(labels)})")

    msg = (
        "The properties of the catalog do not have the same length: "
        f"{pretty_str(sizes, sort=False)}."
    )
    raise ValueError(msg)


def _is_scalar(property_values: Any) -> bool:
    """Return whether a property is a scalar rather than a sequence of values.

    A string and a mapping are iterable but stand for a single value,
    so they are scalars;
    so is a zero-dimensional array,
    which claims to be iterable but raises when it is iterated over.
    Everything else that can be iterated over is a sequence of values,
    which covers a `Sequence`, an `ndarray`, a pandas `Index`, `Series`
    or extension array, and a generator.
    Testing for a concrete set of sequence types instead
    would read every other iterable as a scalar
    and broadcast it over the rows as a single value,
    e.g. crashing on a pandas extension array
    with a raw NumPy broadcast error.

    Args:
        property_values: The property.

    Returns:
        Whether the property is a scalar.
    """
    if isinstance(property_values, (str, bytes, bytearray, Mapping)):
        return True

    return getattr(property_values, "ndim", None) == 0 or not isinstance(
        property_values, Iterable
    )


def _normalize_string(value: Any) -> Any:
    """Normalize a value if it is a string.

    Args:
        value: The value.

    Returns:
        The value, normalized to `bytes` if it is a byte string.
    """
    if isinstance(value, ndarray) and value.ndim == 0 and value.dtype.kind in "US":
        # A zero-dimensional string array stands for a single string, and is
        # read as a scalar property, so unwrap it rather than leave the only
        # scalar kind of property that is not a string instance.
        return value.item()

    # NumPy reads a bytearray through the buffer protocol, unlike a str
    # or a bytes object, which it treats as a single value; left as a
    # bytearray, a scalar property would be exploded into one row per
    # byte when it is later broadcast over the rows or frozen, and an
    # element of a sequence property would make the property inhomogeneous,
    # so it is normalized to bytes here, upstream of both.
    return bytes(value) if isinstance(value, bytearray) else value


def _decode_labels(labels: list[Any] | None) -> tuple[list[Any] | None, list[str]]:
    """Decode the byte-string labels from UTF-8.

    NumPy converts a byte string to a string with the ASCII codec,
    which fails on any other byte with an unclear message,
    while a catalog writes its labels to an HDF file encoded to UTF-8;
    so a byte-string label is decoded from UTF-8 instead.

    Args:
        labels: The normalized labels, `None` if not supplied.

    Returns:
        The labels, with the byte strings decoded to strings,
        `None` if not supplied,
        and the error messages, one per byte-string label
        that is not valid UTF-8, which is left as is.
    """
    if labels is None:
        return None, []

    decoded_labels = []
    error_messages = []
    for label in labels:
        if isinstance(label, bytes):
            try:
                label = label.decode()  # noqa: PLW2901
            except UnicodeDecodeError:
                error_messages.append(
                    f"The label {label!r} of the catalog is a byte string "
                    "that cannot be decoded from UTF-8."
                )

        decoded_labels.append(label)

    return decoded_labels, error_messages


def _normalize_strings(property_values: Any) -> Any:
    """Normalize the strings of a property.

    A sequence of Python objects is converted to a list,
    which also reveals the true shape of a column of equal-length sequences.

    Args:
        property_values: The property, possibly a scalar or `None`.

    Returns:
        The property with its strings normalized,
        see [_normalize_string][gemseo.space.catalog._input._normalize_string].
    """
    if property_values is None:
        return None

    if _is_scalar(property_values):
        return _normalize_string(property_values)

    if isinstance(property_values, ndarray) and property_values.dtype.kind != "O":
        # Such an array holds no bytearray, no zero-dimensional array
        # and no nested sequence, so there is nothing to normalize;
        # leaving it alone also keeps its data type, which the list
        # below would lose, and skips one Python-level pass over its values.
        return property_values

    return [_normalize_string(value) for value in property_values]


def is_missing(value: Any) -> bool:
    """Return whether a value is missing.

    Args:
        value: The value.

    Returns:
        Whether the value is missing.
    """
    missing = isna(value)
    # isna returns an array for a sequence, which is not a value here.
    return bool(missing) if isinstance(missing, bool) or missing.ndim == 0 else False


def _is_blank(value: Any) -> bool:
    """Return whether a value carries no information.

    A blank value is a missing value
    or a string that is empty once stripped of its leading and trailing whitespace.
    A zero and a `False` are values, so they are not blank.

    Args:
        value: The value.

    Returns:
        Whether the value is blank.
    """
    if isinstance(value, (str, bytes)):
        return not value.strip()

    return is_missing(value)


def is_blank_property(property_values: Iterable[Any]) -> bool:
    """Return whether a property holds no value.

    A property holds no value when each of its elements is blank,
    i.e. is a missing value
    or a string that is empty once stripped of its leading and trailing whitespace;
    `0` and `False` are values, so their properties hold values.
    This is the notion
    [clean_catalog_data][gemseo.space.catalog._input.clean_catalog_data]
    drops a property by.

    Args:
        property_values: The property.

    Returns:
        Whether the property holds no value.
    """
    return all(_is_blank(value) for value in property_values)


def clean_catalog_data(
    labels: ndarray, properties: Mapping[str, ndarray]
) -> tuple[ndarray, ReadOnlyMapping[str, ndarray]]:
    """Clean the data of a catalog.

    The strings of the labels and of the properties are stripped
    of their leading and trailing whitespace,
    then the properties holding no value are dropped
    and named in a log warning,
    see [is_blank_property][gemseo.space.catalog._input.is_blank_property].

    The data of a catalog are already valid,
    and cleaning them keeps them so,
    so the cleaned data are frozen directly rather than formatted again.

    Args:
        labels: The labels of the catalog.
        properties: The properties of the catalog.

    Returns:
        The cleaned labels and the cleaned properties, frozen.

    Raises:
        ValueError: If every property holds no value,
            since cleaning would leave none,
            or if labels differ only by their leading and trailing whitespace,
            since they would name several alternatives once stripped,
            naming every such error at once.
    """
    cleaned_properties = {}
    dropped_names = []
    for name, property_values in properties.items():
        stripped_property = _strip_strings(property_values)
        if is_blank_property(stripped_property):
            dropped_names.append(name)
        else:
            cleaned_properties[name] = _freeze_stripped_strings(stripped_property)

    error_messages = []
    if dropped_names and not cleaned_properties:
        error_messages.append(
            "Every property of the catalog is blank, "
            "i.e. holds only missing values or empty strings, "
            "so cleaning the catalog would leave no property: "
            f"{pretty_str(dropped_names, sort=False)}."
        )

    stripped_labels = _strip_strings(labels)
    if merged_labels := _find_merged_labels(labels, stripped_labels):
        error_messages.append(
            "The following labels of the catalog differ only "
            "by their leading and trailing whitespace, "
            "so cleaning the catalog would make them duplicates: "
            f"{pretty_str(merged_labels, sort=False)}."
        )

    raise_errors("The catalog cannot be cleaned:", error_messages)
    if dropped_names:
        logger.warning(
            "The following properties of the catalog hold no value "
            "and were dropped: %s.",
            pretty_str(dropped_names, sort=False),
        )

    cleaned_labels = _freeze_stripped_strings(stripped_labels)
    return cleaned_labels, ReadOnlyMapping(cleaned_properties)


def _freeze_stripped_strings(stripped_values: ndarray) -> ndarray:
    """Freeze an array whose strings have been stripped.

    Stripping keeps the width of the data type of an array of strings,
    so such an array is rebuilt at the width of its longest stripped string,
    as the constructor of a catalog would build it.

    Args:
        stripped_values: The array whose strings have been stripped.

    Returns:
        The frozen array.
    """
    dtype = stripped_values.dtype
    if dtype.kind in "US":
        return _freeze(stripped_values.tolist(), dtype.type)

    return freeze_property_array(stripped_values)


def _find_merged_labels(labels: ndarray, stripped_labels: ndarray) -> list[str]:
    """Return the stripped labels that stripping made duplicates.

    Args:
        labels: The labels.
        stripped_labels: The labels stripped of their leading and trailing whitespace.

    Returns:
        The representation of each stripped label
        that two different labels are stripped to.
    """
    stripped_label_to_labels: dict[str, set[str]] = {}
    for label, stripped_label in zip(
        labels.tolist(), stripped_labels.tolist(), strict=True
    ):
        stripped_label_to_labels.setdefault(stripped_label, set()).add(label)

    # The labels are quoted, so that an empty label is visible.
    return [
        repr(stripped_label)
        for stripped_label, original_labels in stripped_label_to_labels.items()
        if len(original_labels) > 1
    ]


def _strip_strings(values: ndarray) -> ndarray:
    """Remove the leading and trailing whitespace of the strings of an array.

    Args:
        values: The array.

    Returns:
        A copy of the array with its strings stripped,
        or the array itself if it cannot hold a string.
    """
    kind = values.dtype.kind
    if kind in "US":
        return strip_array(values)

    if kind == "O":
        stripped_values = values.copy()
        for index, value in enumerate(values):
            if isinstance(value, (str, bytes)):
                stripped_values[index] = value.strip()

        return stripped_values

    return values


def _mixes_string_and_missing(property_values: Any) -> bool:
    """Return whether a property holds both a string value and a missing value.

    Left to infer the dtype of such a property, NumPy promotes a `str`,
    `bytes` or `bytearray` value and a missing one, e.g. a float `NaN`, to a
    common fixed-width string dtype, `<U` or `|S`, which silently turns the
    missing cell into the literal text `"nan"` and makes it indistinguishable
    from a real value once frozen.

    Args:
        property_values: The property, an array or a sequence of values.

    Returns:
        Whether the property holds at least one `str`, `bytes` or `bytearray`
        value and at least one missing value.
    """
    if isinstance(property_values, ndarray) and property_values.dtype.kind != "O":
        # A homogeneous array cannot hold both a string and a missing value,
        # so it needs none of the per-value calls below.
        return False

    has_string = False
    has_missing = False
    for value in property_values:
        if isinstance(value, (str, bytes, bytearray)):
            has_string = True
        elif is_missing(value):
            has_missing = True

        if has_string and has_missing:
            return True

    return False


def _freeze(property_values: Any, dtype: type | None = None) -> ndarray:
    """Convert a property to a read-only array.

    Args:
        property_values: The property.
        dtype: The NumPy type of the array. If `None`, let NumPy infer it,
            unless the property mixes a string and a missing value, in which
            case the array is kept as `object` so that the missing value is
            not silently turned into the literal string `"nan"`, see
            [_mixes_string_and_missing][gemseo.space.catalog._input._mixes_string_and_missing].

    Returns:
        The frozen array.
    """
    if dtype is None and _mixes_string_and_missing(property_values):
        dtype = object

    return freeze_property_array(asarray(property_values, dtype=dtype))


def freeze_property_array(property_values: ndarray) -> ndarray:
    """Return a read-only copy of an array.

    Args:
        property_values: The array.

    Returns:
        The frozen array,
        sharing neither its buffer nor, for an array of Python objects,
        its values with the array given.
    """
    if property_values.dtype.kind != "O":
        return freeze_array(property_values)

    # freeze_array cannot represent an array of Python objects, which does not
    # survive a round trip through a buffer of bytes. Deep-copy it instead, so
    # that the catalog neither shares the objects of the caller nor lets them
    # be changed through it: copying the array alone would only copy the
    # pointers to them. The array returned owns its data, so a determined
    # caller can re-enable its writeable flag, unlike a frozen one.
    frozen_property = deepcopy(property_values)
    frozen_property.setflags(write=False)
    return frozen_property
