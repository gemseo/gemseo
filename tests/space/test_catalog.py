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
"""Tests for the catalog of a catalog variable."""

from __future__ import annotations

import copy
import logging
import pickle
from pathlib import Path

import h5py
import pytest
from numpy import array
from numpy import dtype
from numpy import float64
from numpy import nan
from numpy import ndarray
from numpy.testing import assert_array_equal
from pandas import Categorical
from pandas import DataFrame
from pandas import Index
from pandas import MultiIndex
from pandas import Series
from pandas import array as pandas_array
from pandas import isna
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from gemseo.space import Catalog
from gemseo.util.read_only_mapping import ReadOnlyMapping
from gemseo.util.testing.helper import assert_exception


@pytest.fixture
def table() -> DataFrame:
    """A catalog of materials."""
    return DataFrame(
        {"mass": [2.7, 7.8, 4.5], "supplier": ["ACME", "Foundry", "Mill"]},
        index=["aluminium", "steel", "titanium"],
    )


def test_fields(table) -> None:
    """Check the fields of a catalog built from a table."""
    catalog = Catalog(properties=table)

    assert len(catalog) == 3
    assert tuple(catalog.properties) == ("mass", "supplier")
    assert_array_equal(catalog.labels, array(["aluminium", "steel", "titanium"]))
    assert_array_equal(catalog.properties["mass"], array([2.7, 7.8, 4.5]))


def test_mapping_and_table_agree(table) -> None:
    """Check that a mapping and a table describing the same catalog are equal."""
    catalog = Catalog(
        properties={"mass": [2.7, 7.8, 4.5], "supplier": ["ACME", "Foundry", "Mill"]},
        labels=["aluminium", "steel", "titanium"],
    )
    assert catalog == Catalog(properties=table)


def test_labels_override_table_index(table) -> None:
    """Check that explicit labels take precedence over a table index."""
    catalog = Catalog(properties=table, labels=["alu", "steel", "ti"])
    assert_array_equal(catalog.labels, array(["alu", "steel", "ti"]))


def test_default_labels() -> None:
    """Check that the positions are used when no label is supplied."""
    catalog = Catalog(properties={"mass": [2.7, 7.8]})
    assert_array_equal(catalog.labels, array(["0", "1"]))


def test_strip() -> None:
    """Check that the stored strings are stripped, in the properties and the labels."""
    catalog = Catalog(
        properties={"supplier": [" ACME ", "Foundry\t"]}, labels=[" alu", "steel "]
    )

    assert_array_equal(catalog.properties["supplier"], array(["ACME", "Foundry"]))
    assert_array_equal(catalog.labels, array(["alu", "steel"]))


def test_series_property_is_stripped() -> None:
    """Check that a property given as a Series is stripped like a list.

    A Series is neither a `Sequence`, an `ndarray` nor an `Index`,
    but it is read as a sequence of values, each of them stripped.
    """
    catalog = Catalog(properties={"s": Series([" ACME ", "B "]), "x": [1, 2]})

    assert_array_equal(catalog.properties["s"], array(["ACME", "B"]))


@pytest.mark.parametrize(
    "property_values",
    [
        [],
        [nan, nan],
        [None, None],
        ["", ""],
        ["  ", "\t"],
        ["", nan],
        [b"", b""],
        [b"  ", b" "],
        nan,
        None,
        "   ",
    ],
)
def test_blank_property_is_dropped(property_values, caplog) -> None:
    """Check that a property holding no value is dropped and named in a warning."""
    with caplog.at_level(logging.WARNING):
        catalog = Catalog(properties={"mass": [2.7, 7.8], "blank": property_values})

    assert tuple(catalog.properties) == ("mass",)
    assert (
        "The following properties of the catalog hold no value and were dropped: blank."
        in caplog.text
    )


def test_blank_properties_are_named_together(caplog) -> None:
    """Check that a single warning names every dropped property."""
    with caplog.at_level(logging.WARNING):
        Catalog(properties={"mass": [2.7], "note": [""], "void": [nan]})

    assert len(caplog.records) == 1
    assert (
        "hold no value and were dropped: note and void"
        in caplog.records[0].getMessage()
    )


@pytest.mark.parametrize(
    "property_values", [[0, 0], [False, False], [2.7, nan], ["ACME", ""]]
)
def test_property_is_kept(property_values) -> None:
    """Check that a property holding at least one value is kept.

    A zero and a `False` are values, so the emptiness of a property is
    its blankness and never its falsiness.
    """
    catalog = Catalog(properties={"mass": [2.7, 7.8], "kept": property_values})
    assert tuple(catalog.properties) == ("mass", "kept")


def test_ragged_mapping(snapshot) -> None:
    """Check that properties of different lengths are rejected."""
    with assert_exception(ValidationError, snapshot):
        Catalog(properties={"mass": [2.7, 7.8], "cost": [1.0, 2.0, 3.0]})


def test_ragged_mapping_with_series_property(snapshot) -> None:
    """Check that a length-mismatched Series property is rejected like a list.

    A Series is read as a sequence, not as a scalar,
    so a 1-element Series is not broadcast over the rows
    but trips the rectangularity check.
    """
    with assert_exception(ValidationError, snapshot):
        Catalog(properties={"mass": Series([5.0]), "cost": [1.0, 2.0, 3.0]})


def test_labels_of_wrong_length(snapshot) -> None:
    """Check that labels not matching the properties are rejected."""
    with assert_exception(ValidationError, snapshot):
        Catalog(properties={"mass": [2.7, 7.8]}, labels=["alu"])


def test_scalar_is_broadcast() -> None:
    """Check that a scalar property is repeated over the rows."""
    catalog = Catalog(properties={"mass": [2.7, 7.8, 4.5], "stock": 0})
    assert_array_equal(catalog.properties["stock"], array([0, 0, 0]))


def test_scalars_only() -> None:
    """Check that a mapping of scalars only yields a single-row catalog.

    Nothing carries a length, so the catalog is degenerate but well formed.
    """
    catalog = Catalog(properties={"mass": 2.7, "supplier": "ACME"})

    assert len(catalog) == 1
    assert_array_equal(catalog.properties["mass"], array([2.7]))


def test_scalars_only_with_labels() -> None:
    """Check that the labels fix the row count of a mapping of scalars only."""
    catalog = Catalog(properties={"mass": 2.7}, labels=["alu", "steel"])

    assert len(catalog) == 2
    assert_array_equal(catalog.properties["mass"], array([2.7, 2.7]))


def test_scalar_over_one_row() -> None:
    """Check that a scalar and a one-element sequence give the same catalog."""
    assert Catalog(properties={"mass": 2.7}) == Catalog(properties={"mass": [2.7]})


def test_catalog_from_single_series_property() -> None:
    """Check that a catalog built from a single Series property does not crash.

    A Series is read as a plain sequence, whose length defines the rows.
    """
    catalog = Catalog(properties={"mass": Series([2.7, 7.8, 4.5])})

    assert len(catalog) == 3
    assert_array_equal(catalog.properties["mass"], array([2.7, 7.8, 4.5]))


def test_series_property_with_non_default_index() -> None:
    """Check that a Series property is read positionally, ignoring its index.

    The pandas index of the Series must not leak into either the values or
    the default labels of the catalog.
    """
    catalog = Catalog(properties={"mass": Series([1.0, 2.0, 3.0], index=[10, 20, 30])})

    assert_array_equal(catalog.properties["mass"], array([1.0, 2.0, 3.0]))
    assert_array_equal(catalog.labels, array(["0", "1", "2"]))


@pytest.mark.parametrize(
    "property_values", [pandas_array([1, 2], dtype="Int64"), Categorical(["a", "b"])]
)
def test_extension_array_property(property_values) -> None:
    """Check that a pandas extension array property is read as a sequence.

    An extension array is neither a `Sequence`, an `ndarray`, an `Index` nor
    a `Series`, but it is read as a sequence, whose length defines the rows.
    """
    catalog = Catalog(properties={"x": property_values})

    assert len(catalog) == 2
    assert_array_equal(catalog.properties["x"], array(list(property_values)))


def test_string_property_with_a_missing_value_keeps_it_missing() -> None:
    """Check that a NaN in a text property is not turned into the text "nan".

    NumPy infers a common dtype for a property given as a plain sequence, and
    promotes a `str` and a `float` to a fixed-width string dtype, `<U`,
    which would turn the missing cell into the literal string `"nan"`;
    the property is kept as an object property holding a missing value.
    """
    catalog = Catalog(
        properties={"supplier": ["ACME", nan, "Mill"], "mass": [1.0, 2.0, 3.0]}
    )
    literal_nan = Catalog(
        properties={"supplier": ["ACME", "nan", "Mill"], "mass": [1.0, 2.0, 3.0]}
    )
    property_values = catalog.properties["supplier"]

    assert property_values.dtype.kind == "O"
    assert isna(property_values[1])
    assert (catalog == literal_nan) is False


def test_byte_string_property_with_a_missing_value_keeps_it_missing() -> None:
    """Check that a NaN in a byte-string property is not turned into "b'nan'".

    NumPy promotes `bytes` and `float` the same way it promotes `str` and
    `float`, to a fixed-width byte-string dtype, `|S`, so a byte-string
    property mixing a value and a missing one gets the same treatment as a
    text property.
    """
    catalog = Catalog(properties={"x": [b"ACME", nan]})
    property_values = catalog.properties["x"]

    assert property_values.dtype.kind == "O"
    assert isna(property_values[1])


def test_numeric_property_with_a_missing_value_keeps_a_float_dtype() -> None:
    """Check that a NaN in a purely numeric property does not force an object dtype.

    Only a property mixing a string and a missing value must be kept as
    `object`; a numeric property must keep its float dtype, which
    `properties_are_equal` relies on for its `equal_nan` comparison.
    """
    catalog = Catalog(properties={"mass": [1.0, nan]})

    assert catalog.properties["mass"].dtype.kind == "f"
    assert catalog == catalog


def test_generator_property() -> None:
    """Check that a property given as a generator is read as a sequence.

    A generator is iterable but is no `Sequence`;
    it is read as the sequence of the values it yields,
    not as a single-row object property holding the generator itself.
    """
    catalog = Catalog(properties={"mass": (float(index) for index in range(3))})

    assert len(catalog) == 3
    assert_array_equal(catalog.properties["mass"], array([0.0, 1.0, 2.0]))


def test_mapping_property_is_a_scalar() -> None:
    """Check that a property given as a mapping is broadcast as a single value.

    A mapping is iterable, but over its keys, so it stands for one value.
    """
    catalog = Catalog(properties={"mass": [2.7, 7.8], "spec": {"grade": 1}})

    assert_array_equal(catalog.properties["spec"], array([{"grade": 1}, {"grade": 1}]))


@pytest.mark.parametrize("property_values", [{"ACME"}, frozenset({"ACME"})])
def test_unordered_property_is_rejected(property_values, snapshot) -> None:
    """Check that a property given as an unordered collection is rejected.

    A set is iterable and sized, so it would otherwise be read as a sequence
    whose order, and hence the order of the rows, would be arbitrary.
    A single element keeps the rejection message, which renders the input,
    free of the iteration order that a larger set would make unstable.
    """
    with assert_exception(ValidationError, snapshot):
        Catalog(properties={"supplier": property_values})


def test_nested_table_property_is_rejected(snapshot) -> None:
    """Check that a property given as a table is rejected.

    A `DataFrame` is iterable over its column **names**,
    so a nested table is rejected
    rather than read as a property holding those names.
    """
    with assert_exception(ValidationError, snapshot):
        Catalog(properties={"spec": DataFrame({"mass": [2.7, 7.8]})})


def test_every_property_of_an_invalid_kind_is_reported(snapshot) -> None:
    """Check that every property of a kind a property cannot be read from is named.

    A single error names all the offenders,
    so that a caller learns about them all at once.
    """
    with assert_exception(ValidationError, snapshot):
        Catalog(
            properties={
                "spec": DataFrame({"mass": [2.7, 7.8]}),
                "supplier": {"ACME"},
                "grade": frozenset({"A"}),
            }
        )


def test_zero_dimensional_array_property_is_a_scalar() -> None:
    """Check that a zero-dimensional array property is broadcast as one value.

    Such an array claims to be iterable but raises when it is iterated over;
    it is read as a scalar, like the `np.float64` it stands with.
    """
    catalog = Catalog(properties={"mass": [2.7, 7.8], "stock": array(0)})

    assert len(catalog) == 2
    assert_array_equal(catalog.properties["stock"], array([0, 0]))


def test_zero_dimensional_string_array_property_is_stripped() -> None:
    """Check that a zero-dimensional string array property is stripped.

    It is the one kind of scalar property that is not a `str` instance,
    so stripping has to unwrap it first.
    """
    catalog = Catalog(properties={"mass": [2.7, 7.8], "supplier": array("  ACME  ")})

    assert_array_equal(catalog.properties["supplier"], array(["ACME", "ACME"]))


def test_byte_strings_in_a_sequence_property_are_stripped() -> None:
    """Check that the byte strings of a sequence property are stripped.

    A `bytearray` element is stripped into a `bytes` value,
    since NumPy would read a `bytearray` through the buffer protocol
    as a nested sequence of bytes rather than as a single value.
    """
    catalog = Catalog(properties={"name": [b"  steel  ", bytearray(b" alu ")]})

    assert_array_equal(catalog.properties["name"], array([b"steel", b"alu"]))


def test_non_string_mapping_key(snapshot) -> None:
    """Check that a mapping whose key is not a string is rejected.

    A `DataFrame` has its column names stringified, but a mapping must supply
    them as strings already, since its keys are the property names as is.
    """
    with assert_exception(ValidationError, snapshot):
        Catalog(properties={1: [2.7, 7.8]})


def test_table_column_names_are_stringified() -> None:
    """Check that the column names of a table are converted to strings."""
    catalog = Catalog(properties=DataFrame({1: [2.7, 7.8]}))
    assert tuple(catalog.properties) == ("1",)


@pytest.mark.parametrize("other_name", [1, True])
def test_table_column_names_colliding_once_stringified(other_name, snapshot) -> None:
    """Check that column names colliding once stringified are rejected.

    Otherwise, the dict built from the property names would keep only one of
    the colliding properties, silently dropping the other.
    """
    with assert_exception(ValidationError, snapshot):
        Catalog(
            properties=DataFrame({other_name: [10, 20], str(other_name): ["a", "b"]})
        )


def test_every_duplicate_column_name_is_reported(snapshot) -> None:
    """Check that every column name colliding once stringified is named."""
    with assert_exception(ValidationError, snapshot):
        Catalog(
            properties=DataFrame(
                data=[[10, 20, 30, 40]],
                columns=Index([1, "1", 2, "2"], dtype=object),
            )
        )


def test_a_column_name_colliding_several_times_is_reported_once(snapshot) -> None:
    """Check that a column name colliding more than once is named only once."""
    with assert_exception(ValidationError, snapshot):
        Catalog(
            properties=DataFrame(
                data=[[10, 20, 30]], columns=Index([1, "1", "1"], dtype=object)
            )
        )


@pytest.mark.parametrize("labels", [array(["alu", "steel"]), Index(["alu", "steel"])])
def test_labels_from_an_array_or_an_index(labels) -> None:
    """Check that the labels can be given as an array or a pandas index."""
    catalog = Catalog(properties={"mass": [2.7, 7.8]}, labels=labels)
    assert_array_equal(catalog.labels, array(["alu", "steel"]))


def test_str_labels_is_a_single_label() -> None:
    """Check that a `str` labels input is read as a single label.

    A `str` is a `Sequence` of its characters,
    but it is read as one label, the way a scalar `str` property is,
    not as one label per character, e.g. a 9-row catalog
    for `labels="aluminium"`.
    """
    catalog = Catalog(properties={"mass": 2.7}, labels="aluminium")

    assert len(catalog) == 1
    assert_array_equal(catalog.labels, array(["aluminium"]))


@pytest.mark.parametrize("value", [b"ab", bytearray(b"ab")])
def test_bytes_labels_is_a_single_label(value) -> None:
    """Check that a `bytes` or `bytearray` labels input is read as a single label.

    `bytes` and `bytearray` are `Sequence` instances,
    but such a labels input is read as one label,
    the way a scalar `bytes` or `bytearray` property is,
    not as one label per byte, e.g. `["97", "98"]` for `labels=b"ab"`.
    """
    catalog = Catalog(properties={"mass": 2.7}, labels=value)

    assert len(catalog) == 1
    assert_array_equal(catalog.labels, array(["ab"]))


@pytest.mark.parametrize(
    "labels",
    [
        [b"\xc3\xa9", b" b\xc2\xa0"],
        array([b"\xc3\xa9", b"b"]),
        [bytearray(b"\xc3\xa9"), "b"],
    ],
)
def test_byte_string_labels_are_decoded_from_utf8(labels) -> None:
    """Check that byte-string labels are decoded from UTF-8, then stripped.

    NumPy converts a byte string to a string with the ASCII codec,
    which would reject a non-ASCII byte-string label;
    UTF-8 is used instead.
    """
    catalog = Catalog(properties={"mass": [2.7, 7.8]}, labels=labels)

    assert_array_equal(catalog.labels, array(["\u00e9", "b"]))


def test_byte_string_labels_not_utf8(snapshot) -> None:
    """Check that byte-string labels that are not valid UTF-8 are rejected."""
    with assert_exception(ValidationError, snapshot):
        Catalog(properties={"mass": [2.7, 7.8]}, labels=[b"\xff", b"\xfe"])


@pytest.mark.parametrize(
    "index", [[b"\xc3\xa9", b" b\xc2\xa0"], [bytearray(b"\xc3\xa9"), "b"]]
)
def test_byte_string_index_is_decoded_from_utf8(index) -> None:
    """Check that a byte-string index is decoded from UTF-8, then stripped.

    A byte-string index is decoded, not converted with `str`,
    which would give labels like `"b'alu'"` that no label could resolve.
    """
    catalog = Catalog(properties=DataFrame({"mass": [2.7, 7.8]}, index=index))

    assert_array_equal(catalog.labels, array(["\u00e9", "b"]))
    assert catalog.get_position("\u00e9") == 0


def test_byte_string_index_not_utf8(snapshot) -> None:
    """Check that a byte-string index that is not valid UTF-8 is rejected."""
    with assert_exception(ValidationError, snapshot):
        Catalog(properties=DataFrame({"mass": [2.7, 7.8]}, index=[b"\xff", b"a"]))


def test_zero_dimensional_array_labels_is_a_single_label() -> None:
    """Check that a zero-dimensional array labels input is read as a single label.

    A zero-dimensional array claims to be sized but raises a raw
    `TypeError` when `len` is called on it;
    it is one value, so it is read as a single label,
    the way a zero-dimensional array property is.
    """
    catalog = Catalog(properties={"mass": [1.0]}, labels=array("x"))

    assert len(catalog) == 1
    assert_array_equal(catalog.labels, array(["x"]))


def test_empty_string_labels_is_one_label() -> None:
    """Check that an empty-string labels input is one label, not an absence of labels.

    The default labels, `()`, is empty and means no label was supplied,
    but an empty `str` is still a single scalar value, like any other
    `str`, so it is read as one label naming a single row rather than as
    the absence of labels.
    """
    catalog = Catalog(properties={"mass": 2.7}, labels="")

    assert len(catalog) == 1
    assert_array_equal(catalog.labels, array([""]))


@pytest.mark.parametrize(
    ("value", "stripped"),
    [(b"  steel  ", b"steel"), (bytearray(b"  steel  "), b"steel")],
)
def test_byte_string_scalar_is_stripped_and_not_exploded(value, stripped) -> None:
    """Check that a scalar byte string is stripped and broadcast as one value.

    `bytes` and `bytearray` are `Sequence` instances,
    but a scalar byte-string property is stripped and broadcast
    as a single value over the rows, the way a scalar `str` property is,
    not exploded into one row per byte,
    e.g. into a 5-row property of byte codes for `b"steel"`.
    """
    catalog = Catalog(properties={"name": value, "mass": 7.8})

    assert len(catalog) == 1
    assert_array_equal(catalog.properties["name"], array([stripped]))


def test_no_row(snapshot) -> None:
    """Check that a table without any row is rejected."""
    with assert_exception(ValidationError, snapshot):
        Catalog(properties=DataFrame({"mass": []}))


@pytest.mark.parametrize("properties", [{"mass": []}, {"mass": [], "cost": []}])
def test_no_row_in_a_mapping(properties, snapshot) -> None:
    """Check that a mapping whose every property is empty is rejected as rowless.

    An empty property is blank, but the catalog is rejected for having no row,
    not for having no property once the blank ones are dropped:
    the caller did supply properties, without any row.
    """
    with assert_exception(ValidationError, snapshot):
        Catalog(properties=properties)


def test_no_row_in_a_mapping_with_labels(snapshot) -> None:
    """Check that empty properties contradicted by labels report the disagreement.

    The labels name rows that the empty properties deny, so the failure is a
    disagreement in lengths rather than the absence of a row.
    """
    with assert_exception(ValidationError, snapshot):
        Catalog(properties={"mass": []}, labels=["alu"])


def test_no_property(snapshot) -> None:
    """Check that a catalog without any property is rejected."""
    with assert_exception(ValidationError, snapshot):
        Catalog(properties={})


def test_every_property_blank(snapshot, caplog) -> None:
    """Check that a catalog whose every property is blank is rejected.

    The properties are dropped first, and named in a warning,
    so the failure reports the absence of a property.
    """
    with caplog.at_level(logging.WARNING), assert_exception(ValidationError, snapshot):
        Catalog(properties={"note": ["", ""], "void": [nan, nan]})

    assert "were dropped: note and void" in caplog.text


@pytest.mark.parametrize("axis", ["index", "columns", "both"])
def test_multi_index(axis, snapshot) -> None:
    """Check that a table with a nested axis is rejected."""
    index = MultiIndex.from_tuples([("a", 1), ("b", 2)])
    if axis == "index":
        table = DataFrame({"mass": [2.7, 7.8]}, index=index)
    elif axis == "columns":
        table = DataFrame([[2.7], [7.8]], columns=index[:1])
    else:
        table = DataFrame([[2.7], [7.8]], index=index, columns=index[:1])
    with assert_exception(ValidationError, snapshot):
        Catalog(properties=table)


def test_duplicate_column_labels(snapshot) -> None:
    """Check that a table with duplicate column labels is rejected.

    Two labels that are already equal collide once converted to strings,
    like two labels that differ only in type,
    so both are rejected by the same check.
    """
    table = DataFrame([[1, 2], [3, 4]], columns=["mass", "mass"])
    with assert_exception(ValidationError, snapshot):
        Catalog(properties=table)


def test_nested_sequence_property(snapshot) -> None:
    """Check that a property built from a nested sequence is rejected."""
    with assert_exception(ValidationError, snapshot):
        Catalog(properties={"mass": [[1, 2], [3, 4]]})


def test_every_property_of_a_dimension_greater_than_one_is_reported(snapshot) -> None:
    """Check that every property of a dimension greater than 1 is named."""
    with assert_exception(ValidationError, snapshot):
        Catalog(properties={"mass": [[1, 2], [3, 4]], "size": [[5, 6], [7, 8]]})


@pytest.mark.parametrize(
    "catalog_input",
    [
        {
            "properties": {"supplier": {"ACME"}, "mass": [[1, 2], [3, 4]]},
            "labels": [["alu"], ["steel"]],
        },
        {
            "properties": DataFrame(
                columns=MultiIndex.from_tuples([("a", "x"), ("a", "x")])
            )
        },
        {
            "properties": {"spec": DataFrame({"mass": [2.7]}), "cost": [10.0]},
            "labels": [b"\xff"],
        },
    ],
    ids=(
        "kind_and_dimensions",
        "multiindex_duplicate_names_and_no_row",
        "nested_table_and_undecodable_label",
    ),
)
def test_every_error_of_the_input_is_reported_at_once(catalog_input, snapshot) -> None:
    """Check that the errors of independent formatting steps are reported together.

    A property reported by a step, e.g. a nested table or colliding columns,
    is not reported again as a property of a dimension greater than 1.
    """
    with assert_exception(ValidationError, snapshot):
        Catalog(**catalog_input)


def test_nested_sequence_column_in_dataframe(snapshot) -> None:
    """Check that a `DataFrame` property of equal-length sequences is rejected.

    Such a property is read back from the `DataFrame` as a 1-D object array,
    which only reveals its true, two-dimensional shape once stripped.
    """
    with assert_exception(ValidationError, snapshot):
        Catalog(properties=DataFrame({"mass": [[1, 2], [3, 4]]}))


def test_nested_sequence_labels(snapshot) -> None:
    """Check that labels built from a nested sequence are rejected.

    A 2-D `labels` would otherwise be accepted, then make `write_hdf`
    store a 2-D labels dataset that `read_hdf` cannot read back,
    and make `__hash__` raise on the array it finds inside `tuple(...)`.
    """
    with assert_exception(ValidationError, snapshot):
        Catalog(properties={"mass": [2.7, 7.8]}, labels=[["x", "y"], ["z", "w"]])


def test_frozen_arrays(table) -> None:
    """Check that every stored array is read-only."""
    catalog = Catalog(properties=table)

    assert not catalog.labels.flags.writeable
    for property_values in catalog.properties.values():
        assert not property_values.flags.writeable


def test_fields_hold_the_stored_types(table) -> None:
    """Check that the fields are typed as, and hold, what a catalog stores.

    The constructor accepts the input types, e.g. a `DataFrame`,
    and formats them into a read-only mapping of frozen arrays
    and a frozen array of labels, which the fields are annotated with.
    """
    catalog = Catalog(properties=table)

    assert (
        Catalog.model_fields["properties"].annotation == ReadOnlyMapping[str, ndarray]
    )
    assert Catalog.model_fields["labels"].annotation is ndarray
    assert isinstance(catalog.properties, ReadOnlyMapping)
    assert_array_equal(catalog.properties["mass"], array([2.7, 7.8, 4.5]))
    assert isinstance(catalog.labels, ndarray)
    assert_array_equal(catalog.labels, array(["aluminium", "steel", "titanium"]))


def test_input_is_isolated(table) -> None:
    """Check that the caller does not keep a hand on the catalog."""
    catalog = Catalog(properties=table)
    table.loc["steel", "mass"] = 999.0

    assert_array_equal(catalog.properties["mass"], array([2.7, 7.8, 4.5]))


def test_to_dataframe_is_isolated(table) -> None:
    """Check that the table handed back cannot change the catalog."""
    catalog = Catalog(properties=table)

    dataframe = catalog.to_dataframe()
    assert_array_equal(dataframe["mass"].to_numpy(), array([2.7, 7.8, 4.5]))
    assert_array_equal(dataframe.index, array(["aluminium", "steel", "titanium"]))

    dataframe.loc["steel", "mass"] = 999.0
    assert_array_equal(catalog.properties["mass"], array([2.7, 7.8, 4.5]))
    # Each call rebuilds the table.
    assert catalog.to_dataframe() is not dataframe


def test_duplicate_labels() -> None:
    """Check that duplicate labels are accepted, since the value is the position."""
    catalog = Catalog(properties={"mass": [2.7, 2.7]}, labels=["alu", "alu"])
    assert len(catalog) == 2


@pytest.mark.parametrize(
    ("other", "expected"),
    [
        (Catalog(properties={"mass": [2.7, 7.8]}, labels=["alu", "steel"]), True),
        (Catalog(properties={"mass": [2.7, 7.8]}, labels=["alu", "iron"]), False),
        (Catalog(properties={"mass": [2.7, 9.9]}, labels=["alu", "steel"]), False),
        (Catalog(properties={"cost": [2.7, 7.8]}, labels=["alu", "steel"]), False),
        (Catalog(properties={"mass": [2.7]}, labels=["alu"]), False),
        ("not a catalog", False),
    ],
)
def test_eq(other, expected) -> None:
    """Check the equality of two catalogs.

    The comparison must return a boolean,
    since `BaseVariable.__eq__` uses it as is.
    """
    catalog = Catalog(properties={"mass": [2.7, 7.8]}, labels=["alu", "steel"])
    assert (catalog == other) is expected


def test_eq_property_order() -> None:
    """Check that the order of the properties takes part in the equality."""
    properties = {"mass": [2.7], "cost": [1.0]}
    assert Catalog(properties=properties) != Catalog(
        properties=dict(reversed(properties.items()))
    )


def test_eq_numeric_vs_string_property() -> None:
    """Check that a numeric property never equals a same-named string property.

    `equal_nan` is requested from `array_equal` only when both properties
    support it, so comparing a float property against a same-shaped string
    property returns `False` instead of raising `TypeError` from `isnan`,
    whichever side is compared first.
    """
    numeric = Catalog(properties={"p": [1.0, 2.0]})
    string = Catalog(properties={"p": ["x", "y"]})

    assert (numeric == string) is False
    assert (string == numeric) is False


def test_eq_with_missing_numeric_value() -> None:
    """Check that a catalog with a missing numeric value equals itself.

    `NaN != NaN`, so comparing the properties without NaN handling would make
    such a catalog never equal itself.
    """
    catalog = Catalog(properties=DataFrame({"mass": [2.7, None, 4.5]}))
    assert catalog == catalog


def test_eq_with_missing_value_in_an_object_property() -> None:
    """Check that a catalog with a missing value in an object property equals itself.

    `array_equal` only takes `equal_nan` for a pair of float or complex
    properties, so a property mixing numbers and missing values, which NumPy reads
    as an object property, is compared element by element,
    a missing value equalling a missing value.
    """
    catalog = Catalog(properties={"mass": [1, nan, None]})

    assert catalog.properties["mass"].dtype.kind == "O"
    assert catalog == catalog


def test_eq_with_missing_value_in_an_int64_extension_array_property() -> None:
    """Check that Int64 properties compare unequal, without raising, on a missing value.

    A raw `==` between `pandas.NA` and a value that is not missing is itself
    `pandas.NA`, and `bool` raises `TypeError` on it rather than reading it
    as `False`; `Catalog.__eq__` returns `False`
    for two catalogs differing where one holds `pandas.NA`.
    """
    with_na = Catalog(properties={"x": pandas_array([1, None], dtype="Int64")})
    without_na = Catalog(properties={"x": pandas_array([1, 2], dtype="Int64")})

    assert (with_na == without_na) is False
    assert (without_na == with_na) is False


def test_eq_with_missing_value_in_a_string_extension_array_property() -> None:
    """Check the same regression as the Int64 case for a pandas "string" property."""
    with_na = Catalog(properties={"x": pandas_array(["a", None], dtype="string")})
    without_na = Catalog(properties={"x": pandas_array(["a", "b"], dtype="string")})

    assert (with_na == without_na) is False
    assert (without_na == with_na) is False


def test_eq_empty_string_and_missing_value() -> None:
    """Check that an empty string is not equal to a missing value.

    Both are blank, which is what drops a property, but only a missing value
    equals another missing value.
    """
    empty = Catalog(properties={"mass": ["", 1]})
    missing = Catalog(properties={"mass": [None, 1]})

    assert (empty == missing) is False


@pytest.mark.parametrize(
    ("property_values", "expected"),
    [
        (Series([1, "1"], dtype=object), ["1", "1"]),
        (["", 1], ["", "1"]),
        (["a", 1.5], ["a", "1.5"]),
    ],
)
def test_mixed_property_is_converted_to_strings(property_values, expected) -> None:
    """Check that a property mixing strings and other values holds strings."""
    catalog_property = Catalog(properties={"code": property_values}).properties["code"]

    assert catalog_property.dtype.kind == "U"
    assert_array_equal(catalog_property, expected)


@pytest.mark.parametrize(
    ("other_value", "expected"),
    [
        ({"a": array([1, 2])}, True),
        ({"a": array([1, 3])}, False),
        ({"a": array([1, 2, 3])}, False),
        ({"b": array([1, 2])}, False),
        ({"a": [array([1, 2])]}, False),
        ({"a": (1, 2)}, False),
    ],
)
def test_eq_object_property_of_dicts_holding_arrays(other_value, expected) -> None:
    """Check the equality of object properties of dicts holding arrays."""
    catalog = Catalog(properties={"x": [{"a": array([1, 2])}, {"a": array([3])}]})
    other = Catalog(properties={"x": [other_value, {"a": array([3])}]})

    assert (catalog == other) is expected
    assert (pickle.loads(pickle.dumps(catalog)) == catalog) is True


@pytest.mark.parametrize(
    ("other_value", "expected"),
    [
        ({"a": [array([1, 2]), 3]}, True),
        ({"a": [array([1, 2])]}, False),
    ],
)
def test_eq_object_property_of_dicts_holding_sequences(other_value, expected) -> None:
    """Check the equality of object properties of dicts holding sequences of arrays."""
    catalog = Catalog(properties={"x": [{"a": [array([1, 2]), 3]}, {"a": 3}]})
    other = Catalog(properties={"x": [other_value, {"a": 3}]})

    assert (catalog == other) is expected


@pytest.mark.parametrize(
    "value",
    [
        {"a": array([nan])},
        {"a": array([nan, 1.0])},
        {"a": nan, "b": array([1, 2])},
        {"b": array([1, 2]), "a": nan},
        {"a": Series([nan, 1.0])},
    ],
)
def test_eq_object_property_with_a_nested_missing_value(value) -> None:
    """Check that a catalog nesting a missing value equals its pickle round trip.

    A missing value nested in an array, a `Series` or a mapping
    equals a missing value, whatever the nesting and the order of the keys.
    """
    catalog = Catalog(properties={"x": [value, {"k": 1}]})
    other = pickle.loads(pickle.dumps(catalog))

    assert (catalog == other) is True
    assert hash(catalog) == hash(other)


@pytest.mark.parametrize(
    ("other_value", "expected"),
    [
        (Series([1.0, 2.0]), True),
        (Series([1.0, 3.0]), False),
        (Series([1.0, 2.0], index=[1, 2]), False),
    ],
)
def test_eq_object_property_of_dicts_holding_series(other_value, expected) -> None:
    """Check the equality of object properties of dicts holding a `Series`."""
    catalog = Catalog(properties={"x": [{"a": Series([1.0, 2.0])}, {"k": 1}]})
    other = Catalog(properties={"x": [{"a": other_value}, {"k": 1}]})

    assert (catalog == other) is expected


@pytest.mark.parametrize(
    ("properties", "other_properties"),
    [
        ({"mass": array([1, 2], dtype="int8")}, {"mass": [1, 2]}),
        ({"mass": [0.0]}, {"mass": [-0.0]}),
        ({"mass": [1, None]}, {"mass": [1, None]}),
        ({"supplier": ["ACME"]}, {"supplier": [" ACME "]}),
    ],
)
def test_hash_agrees_with_eq(properties, other_properties) -> None:
    """Check that two equal catalogs share a hash.

    A frozen model is hashable, but the hash pydantic derives from the field
    values raises on the mapping of arrays the properties field holds. Hashing
    the values themselves would break the contract the other way: equal
    properties may differ in dtype, and equal values in bytes.
    """
    catalog = Catalog(properties=properties)
    other_catalog = Catalog(properties=other_properties)

    assert catalog == other_catalog
    assert hash(catalog) == hash(other_catalog)
    assert len({catalog, other_catalog}) == 1


def test_hash_of_a_catalog_used_as_a_key(table) -> None:
    """Check that a catalog can be used as a mapping key."""
    catalog = Catalog(properties=table)
    assert {catalog: "materials"}[Catalog(properties=table)] == "materials"


@pytest.mark.parametrize(
    ("n_labels", "expected"),
    [
        (1, "[0]"),
        (6, "[0, 1, 2, 3, 4, 5]"),
        (7, "[0, 1, 2, ..., 4, 5, 6] (7 labels)"),
    ],
)
def test_format_labels(n_labels, expected) -> None:
    """Check the rendering of the labels, elided beyond six of them."""
    catalog = Catalog(properties={"mass": list(range(n_labels))})
    assert catalog.format_labels() == expected


@pytest.mark.parametrize(
    ("label", "expected"), [("aluminium", 0), ("titanium", 2), (" steel\t", 1)]
)
def test_get_position(table, label, expected) -> None:
    """Check that a label is stripped then mapped to the position it names."""
    assert Catalog(properties=table).get_position(label) == expected


def test_get_position_with_default_labels() -> None:
    """Check that a catalog without labels is labelled by the positions."""
    assert Catalog(properties={"mass": [2.7, 7.8]}).get_position("1") == 1


@pytest.mark.parametrize("label", ["gold", "Steel"])
def test_get_position_unknown_label(table, label, snapshot) -> None:
    """Check that a label naming no alternative is rejected."""
    with assert_exception(ValueError, snapshot):
        Catalog(properties=table).get_position(label)


def test_get_position_duplicate_label(snapshot) -> None:
    """Check that a label naming several alternatives is rejected."""
    catalog = Catalog(
        properties={"mass": [2.7, 7.8, 4.5, 7.9]},
        labels=["aluminium", "steel", "titanium", "steel"],
    )
    with assert_exception(ValueError, snapshot):
        catalog.get_position("steel")


@pytest.mark.parametrize(
    ("max_length", "expected"),
    [
        (-1, "[..., 6] (7 labels)"),
        (0, "[..., 6] (7 labels)"),
        (1, "[..., 6] (7 labels)"),
        (2, "[0, ..., 6] (7 labels)"),
        (3, "[0, ..., 5, 6] (7 labels)"),
        (4, "[0, 1, ..., 5, 6] (7 labels)"),
        (7, "[0, 1, 2, 3, 4, 5, 6]"),
        (100, "[0, 1, 2, 3, 4, 5, 6]"),
    ],
)
def test_format_labels_max_length(max_length, expected) -> None:
    """Check that the head and tail sizes are derived from `max_length`.

    The elided head and tail must never overlap nor repeat a label.
    A `max_length` below two leaves no room for a head,
    which is then not rendered as an empty leading element,
    and a `max_length` of zero or less is read as one,
    which keeps only the last label,
    although `labels[-0:]` is every label.
    """
    catalog = Catalog(properties={"mass": list(range(7))})
    assert catalog.format_labels(max_length) == expected


@pytest.mark.parametrize("copier", [pickle.loads, copy.copy, copy.deepcopy])
def test_copy(table, copier) -> None:
    """Check that a copy is equal to the original and keeps its arrays frozen."""
    catalog = Catalog(properties=table)
    copied = copier(pickle.dumps(catalog) if copier is pickle.loads else catalog)

    assert copied == catalog
    assert not copied.labels.flags.writeable
    for property_values in copied.properties.values():
        assert not property_values.flags.writeable


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_copy_returns_the_same_instance(table, copier) -> None:
    """Check that copying a catalog hands back the catalog itself.

    A catalog is immutable and its arrays are read-only, so a copy can be
    shared with the original; this also keeps the arrays frozen, since NumPy
    does not preserve the writeable flag across a copy.
    """
    catalog = Catalog(properties=table)
    assert copier(catalog) is catalog


def test_unpickled_arrays_are_views(table, snapshot) -> None:
    """Check that an unpickled catalog hands out views, not the frozen arrays.

    NumPy pickles a view as an independent, data-owning array, and pydantic
    restores a model without re-validating it, so the arrays are refrozen and
    re-viewed on unpickling; a frozen array that owns its data would accept
    `setflags(write=True)` and let the catalog be mutated in place.
    """
    catalog = pickle.loads(pickle.dumps(Catalog(properties=table)))

    with assert_exception(ValueError, snapshot):
        catalog.labels.setflags(write=True)

    with assert_exception(ValueError, snapshot):
        catalog.properties["mass"].setflags(write=True)


def test_repr(table) -> None:
    """Check that a catalog is rendered by its shape, not by its values.

    The pydantic rendering spells out every value of every property, which
    takes 13 kB for a thousand rows and drowns a traceback.
    """
    assert repr(Catalog(properties=table)) == (
        "Catalog(3 alternatives [aluminium, steel, titanium], "
        "properties: mass and supplier)"
    )
    assert repr(Catalog(properties={"mass": 2.7})) == (
        "Catalog(1 alternative [0], properties: mass)"
    )


def test_str(table) -> None:
    """Check that str(), print() and f-strings use the same short rendering.

    The pydantic default for str() also spells out every value of every
    property, same as its default repr(); str() must not fall back to it.
    """
    catalog = Catalog(properties=table)
    assert str(catalog) == repr(catalog)
    assert f"{catalog}" == repr(catalog)


def test_json_is_unsupported(table, snapshot) -> None:
    """Check that a catalog is not serializable to JSON.

    It holds NumPy arrays, which pydantic cannot serialize; HDF is the only
    format a catalog is written to, and a design space holding one is refused
    by `to_csv`. This pins the limitation, so that lifting it is deliberate.
    """
    with assert_exception(PydanticSerializationError, snapshot):
        Catalog(properties=table).model_dump_json()


def test_json_schema() -> None:
    """Check that the JSON schemas of the properties and labels are their input ones.

    Pydantic cannot generate a JSON schema for the read-only mapping and the array
    that a catalog holds, so the fields describe the types accepted at construction.
    """
    properties = Catalog.model_json_schema()["properties"]

    assert properties["properties"]["type"] == "object"
    assert properties["properties"]["additionalProperties"] is True
    assert properties["labels"]["anyOf"] == [
        {"type": "array", "items": {}},
        {"type": "string"},
    ]


def test_json_schema_descriptions() -> None:
    """Check that the descriptions of the fields are not indented.

    An indented paragraph would be rendered by Markdown as a code block.
    """
    properties = Catalog.model_json_schema()["properties"]

    for name in ("properties", "labels"):
        assert "\n " not in properties[name]["description"]


def test_model_validate_a_non_mapping(snapshot) -> None:
    """Check that validating an input that is neither a mapping nor a catalog fails.

    The model validator leaves an input that is not a mapping to pydantic,
    which rejects it.
    """
    with assert_exception(ValidationError, snapshot):
        Catalog.model_validate(1)


def test_model_construct_properties() -> None:
    """Check that the properties of a catalog built with `model_construct` are kept.

    `model_construct` skips the validation,
    so the properties are not frozen into a read-only mapping
    and are returned as they were supplied.
    """
    properties = {"mass": array([2.7])}

    assert Catalog.model_construct(properties=properties).properties is properties


def test_hdf_round_trip_with_non_ascii(tmp_wd) -> None:
    """Check that a round trip through HDF preserves non-ASCII strings.

    A label, a property name and a cell each carry a non-ASCII character,
    which `write_hdf` encodes as UTF-8, not as ASCII.
    """
    catalog = Catalog(
        properties={"matériau": ["acier", "béton"]}, labels=["pièce", "poutre"]
    )
    file_path = Path("catalog.h5")
    with h5py.File(file_path, "w") as h5file:
        catalog.write_hdf(h5file.create_group("catalog"))

    with h5py.File(file_path) as h5file:
        read_catalog = Catalog.read_hdf(h5file["catalog"])

    assert tuple(read_catalog.properties) == ("matériau",)
    assert_array_equal(read_catalog.labels, array(["pièce", "poutre"]))
    assert_array_equal(read_catalog.properties["matériau"], array(["acier", "béton"]))
    assert read_catalog == catalog


def test_hdf_round_trip_with_byte_string_property(tmp_wd) -> None:
    """Check that a round trip through HDF preserves a byte-string property.

    The round trip keeps an `"S"` property an `"S"` one,
    rather than decoding it into a `"U"` one.
    """
    catalog = Catalog(properties={"a": array([b"x", b"y"])})
    file_path = Path("catalog.h5")
    with h5py.File(file_path, "w") as h5file:
        catalog.write_hdf(h5file.create_group("catalog"))

    with h5py.File(file_path) as h5file:
        read_catalog = Catalog.read_hdf(h5file["catalog"])

    assert read_catalog.properties["a"].dtype.kind == "S"
    assert_array_equal(read_catalog.properties["a"], array([b"x", b"y"]))
    assert read_catalog == catalog


def test_hdf_round_trip_preserves_the_property_order(tmp_wd) -> None:
    """Check that a round trip through HDF preserves the order of the properties.

    HDF does not preserve the insertion order of the members of a group,
    while the order of the properties takes part in the equality of two
    catalogs, so the ordered names are written as a dataset of their own.
    """
    catalog = Catalog(
        properties={"zinc": [1, 2], "alu": ["x", "y"], "mass": [3.0, 4.0]},
        labels=["p", "q"],
    )
    file_path = Path("catalog.h5")
    with h5py.File(file_path, "w") as h5file:
        catalog.write_hdf(h5file.create_group("catalog"))

    with h5py.File(file_path) as h5file:
        read_catalog = Catalog.read_hdf(h5file["catalog"])

    assert tuple(read_catalog.properties) == ("zinc", "alu", "mass")
    assert read_catalog == catalog


def test_write_hdf_checks_before_writing_anything(tmp_wd, snapshot) -> None:
    """Check that a refused catalog leaves the HDF group untouched.

    `write_hdf` checks the catalog before writing anything,
    so that a failure leaves no half-written catalog group behind.
    """
    catalog = Catalog(properties={"date": array([0, 1], dtype="datetime64[D]")})
    with h5py.File("catalog.h5", "w") as h5file:
        group = h5file.create_group("catalog")

        with assert_exception(ValueError, snapshot):
            catalog.write_hdf(group)

        assert not len(group)


@pytest.mark.parametrize("dtype", ["datetime64[D]", "timedelta64[D]"])
def test_check_hdf_writable_rejects_a_dtype_hdf_cannot_store(dtype, snapshot) -> None:
    """Check that a property of a dtype an HDF file cannot store is rejected."""
    catalog = Catalog(properties={"date": array([0, 1], dtype=dtype)})
    with assert_exception(ValueError, snapshot):
        catalog.check_hdf_writable()


def test_check_hdf_writable_rejects_an_object_property(snapshot) -> None:
    """Check that a property holding Python objects is rejected.

    Such a property carries no dtype an HDF file could store, and is named as
    holding objects rather than by its opaque `object` dtype.
    """
    catalog = Catalog(properties={"spec": array([{"a": 1}, {"b": 2}], dtype=object)})
    with assert_exception(ValueError, snapshot):
        catalog.check_hdf_writable()


def test_check_hdf_writable_reports_every_offender_at_once(snapshot) -> None:
    """Check that every property an HDF file cannot store is named, and the labels.

    A single error names every offending property and the labels,
    so that a caller learns about them all from one export attempt.
    """
    catalog = Catalog(
        properties={
            "spec": array([{"a": 1}, {"b": 2}], dtype=object),
            "date": array([0, 1], dtype="datetime64[D]"),
            "mass": [2.7, 7.8],
        },
        labels=["alu", "\ud800"],
    )
    with assert_exception(ValueError, snapshot):
        catalog.check_hdf_writable()


@pytest.mark.parametrize("name", ["", "a/b", ".", "..", "a\x00b"])
def test_check_hdf_writable_rejects_an_invalid_property_name(name, snapshot) -> None:
    """Check that a name that is not a valid HDF5 link name is rejected.

    Such a name would be an invalid or ambiguous HDF path,
    so it is rejected before anything is written.
    """
    catalog = Catalog(properties={name: [2.7, 7.8]})
    with assert_exception(ValueError, snapshot):
        catalog.check_hdf_writable()


def test_check_hdf_writable_rejects_a_label_utf8_cannot_encode(snapshot) -> None:
    """Check that a label UTF-8 cannot encode is rejected.

    Such a label, e.g. a lone surrogate produced by `surrogateescape`
    decoding, cannot be written to HDF,
    so it is rejected before anything is written.
    """
    catalog = Catalog(properties={"m": [1.0]}, labels=["\ud800"])
    with assert_exception(ValueError, snapshot):
        catalog.check_hdf_writable()


def test_check_hdf_writable_rejects_a_property_value_utf8_cannot_encode(
    snapshot,
) -> None:
    """Check that a `"U"` property value UTF-8 cannot encode is rejected.

    Such a value, e.g. a lone surrogate produced by `surrogateescape`
    decoding, cannot be written to HDF,
    so it is rejected before anything is written.
    """
    catalog = Catalog(properties={"m": ["\ud800"]})
    with assert_exception(ValueError, snapshot):
        catalog.check_hdf_writable()


def test_check_hdf_writable_rejects_a_property_name_utf8_cannot_encode(
    snapshot,
) -> None:
    """Check that a property name UTF-8 cannot encode is rejected.

    Such a name, e.g. a lone surrogate produced by `surrogateescape`
    decoding, cannot be written to HDF,
    so it is rejected before anything is written.
    """
    catalog = Catalog(properties={"\ud800": [1.0]})
    with assert_exception(ValueError, snapshot):
        catalog.check_hdf_writable()


def test_write_hdf_checks_a_label_utf8_cannot_encode_before_writing(
    tmp_wd, snapshot
) -> None:
    """Check that a catalog with an unencodable label leaves the group empty.

    `write_hdf` checks the labels before writing anything,
    so that a failure leaves no half-written catalog group behind.
    """
    catalog = Catalog(properties={"m": [1.0]}, labels=["\ud800"])
    with h5py.File("catalog.h5", "w") as h5file:
        group = h5file.create_group("catalog")

        with assert_exception(ValueError, snapshot):
            catalog.write_hdf(group)

        assert not len(group)


def test_model_copy_without_update_returns_self(table) -> None:
    """Check that model_copy without an update returns the same instance."""
    catalog = Catalog(properties=table)
    assert catalog.model_copy() is catalog


def test_model_copy_deep_returns_self(table) -> None:
    """Check that a deep model_copy without an update returns the same instance.

    The properties and the labels are copied by the validation, so a deep copy
    has nothing left to copy.
    """
    catalog = Catalog(properties=table)
    assert catalog.model_copy(deep=True) is catalog


def test_model_copy_with_new_properties(table) -> None:
    """Check that model_copy can replace the properties and keeps the labels."""
    catalog = Catalog(properties=table)

    copied = catalog.model_copy(update={"properties": {"cost": [10.0, 5.0, 50.0]}})

    assert tuple(copied.properties) == ("cost",)
    assert_array_equal(copied.properties["cost"], array([10.0, 5.0, 50.0]))
    assert_array_equal(copied.labels, array(["aluminium", "steel", "titanium"]))
    assert not copied.properties["cost"].flags.writeable
    assert tuple(catalog.properties) == ("mass", "supplier")


@pytest.mark.parametrize(
    ("labels", "update", "expected"),
    [
        (
            ["alu", "steel"],
            {"properties": DataFrame({"rho": [3.0, 4.0]}, index=["ti", "cu"])},
            ["ti", "cu"],
        ),
        ((), {"properties": {"rho": [1.0, 2.0, 3.0]}}, ["0", "1", "2"]),
        (
            ["alu", "steel"],
            {"properties": {"rho": [3.0, 4.0]}, "labels": ["ti", "cu"]},
            ["ti", "cu"],
        ),
        (["alu", "steel"], {"labels": ["ti", "cu"]}, ["ti", "cu"]),
        (["alu", "steel"], {"properties": {"rho": [3.0, 4.0]}}, ["alu", "steel"]),
    ],
)
def test_model_copy_labels(labels, update, expected) -> None:
    """Check the labels of a catalog copied with an update.

    New labels are used when given;
    otherwise, a `DataFrame` is labelled by its index,
    a catalog labelled by the positions is labelled by the new positions,
    and any other catalog keeps its labels.
    """
    catalog = Catalog(properties={"rho": [1.0, 2.0]}, labels=labels)
    assert_array_equal(catalog.model_copy(update=update).labels, array(expected))


def test_model_copy_does_not_mutate_the_original(table) -> None:
    """Check that model_copy with an update leaves the original untouched.

    `__copy__` and `__deepcopy__` return the catalog itself,
    so `model_copy` builds a new catalog through validation
    rather than writing the update into the original;
    the arrays of the new catalog are frozen too.
    """
    catalog = Catalog(properties=table)
    original_property_names = tuple(catalog.properties)
    original_mass = catalog.properties["mass"].copy()

    copied = catalog.model_copy(update={"labels": ["alu", "acier", "ti"]})

    assert copied is not catalog
    assert tuple(copied.properties) == original_property_names
    assert_array_equal(catalog.labels, array(["aluminium", "steel", "titanium"]))
    assert_array_equal(copied.labels, array(["alu", "acier", "ti"]))
    assert_array_equal(catalog.properties["mass"], original_mass)
    assert not catalog.labels.flags.writeable
    assert not copied.labels.flags.writeable


def test_properties_field_cannot_be_mutated(table, snapshot) -> None:
    """Check that the public properties field cannot be used to add a property.

    `properties` hands out a read-only mapping,
    so assigning into it raises
    rather than adding a property that the formatting never saw.
    """
    catalog = Catalog(properties=table)

    with assert_exception(TypeError, snapshot):
        catalog.properties["cost"] = array([1.0, 2.0, 3.0])

    assert tuple(catalog.properties) == ("mass", "supplier")
    assert "cost" not in catalog.to_dataframe().columns


def test_properties_field_values_cannot_be_made_writeable(table, snapshot) -> None:
    """Check that a property handed out by the public properties field stays read-only.

    `properties` hands out a view of the frozen array,
    which, unlike a data-owning array, refuses to be made writeable.
    """
    catalog = Catalog(properties=table)
    property_values = catalog.properties["mass"]

    with assert_exception(ValueError, snapshot):
        property_values.setflags(write=True)


def test_labels_field_cannot_be_made_writeable(table, snapshot) -> None:
    """Check that the public labels field stays read-only.

    `labels` hands out a view of the frozen array,
    which, unlike a data-owning array, refuses to be made writeable.
    """
    catalog = Catalog(properties=table)

    with assert_exception(ValueError, snapshot):
        catalog.labels.setflags(write=True)


def test_arrays_are_handed_out_as_fresh_views(table) -> None:
    """Check that each read of the labels and the properties builds a new view.

    A single view shared by every caller would let one of them reassign the
    shape, the strides or the data type of the array the catalog stores,
    which NumPy allows on a read-only array.
    """
    catalog = Catalog(properties=table)

    assert catalog.labels is not catalog.labels
    assert catalog.properties["mass"] is not catalog.properties["mass"]


@pytest.mark.parametrize("reshape", [True, False])
def test_reshaping_an_array_handed_out_does_not_reach_the_catalog(
    table, reshape
) -> None:
    """Check that reassigning the shape or the dtype of an array is harmless.

    NumPy lets these be reassigned on a read-only array, since they belong to
    the array object itself; the catalog hands out a fresh view,
    so such a reassignment changes neither the number of alternatives
    nor the values every other holder reads.
    """
    catalog = Catalog(properties=table)
    labels = catalog.labels
    property_values = catalog.properties["mass"]

    if reshape:
        labels.shape = (3, 1)
        property_values.shape = (3, 1)
    else:
        property_values.dtype = "int64"

    assert len(catalog) == 3
    assert catalog.labels.shape == (3,)
    assert catalog.properties["mass"].shape == (3,)
    assert catalog.properties["mass"].dtype == float64


def test_frozen_arrays_cannot_be_thawed_through_their_base(table, snapshot) -> None:
    """Check that no array behind the ones handed out can be made writeable.

    Freezing an array in place is not enough: NumPy re-enables the writeable
    flag of an array owning its data, so the frozen array reachable as the
    `base` of a view would let a caller mutate the catalog.
    """
    catalog = Catalog(properties=table)

    with assert_exception(ValueError, snapshot):
        catalog.labels.base.setflags(write=True)

    with assert_exception(ValueError, snapshot):
        catalog.properties["mass"].base.setflags(write=True)


def test_object_property_is_deep_copied() -> None:
    """Check that the values of an object property are copied, not shared.

    Copying such a property copies the pointers only,
    so the catalog deep-copies it,
    and holds none of the objects of the caller.
    """
    supplier = {"name": "alu inc."}
    catalog = Catalog(properties={"supplier": [supplier, {"name": "steel inc."}]})

    catalog.properties["supplier"][0]["name"] = "other"

    assert supplier == {"name": "alu inc."}


def test_extra_field_is_rejected(snapshot) -> None:
    """Check that a misspelled field name is rejected rather than ignored."""
    with assert_exception(ValidationError, snapshot):
        Catalog(properties={"mass": [2.7, 7.8]}, labls=["alu", "steel"])


def test_extra_field_is_rejected_by_an_update(table, snapshot) -> None:
    """Check that a misspelled field name is rejected by `model_copy` too."""
    catalog = Catalog(properties=table)

    with assert_exception(ValidationError, snapshot):
        catalog.model_copy(update={"labls": ["alu", "acier", "ti"]})


@pytest.mark.parametrize("dtype", ["datetime64[D]", "timedelta64[D]"])
def test_property_with_a_missing_date_equals_itself(dtype) -> None:
    """Check that a NaT in a date property equals a NaT.

    `array_equal` treats two missing values as equal only when asked to,
    which is done for a date and a duration property too,
    so a catalog holding a NaT equals itself.
    """
    catalog = Catalog(properties={"delivery": array([1, "NaT"], dtype=dtype)})

    assert catalog == catalog  # noqa: PLR0124
    assert catalog == Catalog(properties={"delivery": array([1, "NaT"], dtype=dtype)})


def test_empty_property_beside_a_scalar_property_is_dropped(caplog) -> None:
    """Check that a scalar property defines the row of an otherwise empty table.

    The empty property is dropped as blank,
    the way a blank scalar property is,
    e.g. in `{"mass": "", "supplier": "alu inc."}`,
    rather than making the input rowless.
    """
    catalog = Catalog(properties={"mass": [], "supplier": "alu inc."})

    assert tuple(catalog.properties) == ("supplier",)
    assert len(catalog) == 1
    assert "were dropped: mass" in caplog.text


def test_numeric_property_keeps_its_dtype() -> None:
    """Check that a numeric array property is not rebuilt from Python values.

    Only a property holding strings is stripped,
    so a numeric property keeps its data type.
    """
    catalog = Catalog(properties={"mass": array([1, 2], dtype="int8")})

    assert catalog.properties["mass"].dtype == dtype("int8")
