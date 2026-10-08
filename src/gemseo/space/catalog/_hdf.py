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
"""The HDF storage of the properties of a catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from numpy.char import encode

if TYPE_CHECKING:
    from h5py import Dataset
    from numpy import ndarray


labels_group: Final[str] = "labels"
"""The name of the HDF dataset storing the labels of a catalog."""


property_names_group: Final[str] = "property_names"
"""The name of the HDF dataset storing the ordered property names of a catalog."""


properties_group: Final[str] = "properties"
"""The name of the HDF group storing the properties of a catalog."""


was_unicode_attribute: Final[str] = "was_unicode"
"""The name of the HDF attribute of a property dataset flagging it as encoded.

Set on a property that was a unicode string property encoded to bytes
before it was written, and left unset on a byte-string property,
which is written as is,
so that only the former is decoded back to a unicode property when read.
"""


_hdf_writable_dtype_kinds: Final[frozenset[str]] = frozenset({
    "b",  # Boolean.
    "i",  # Signed integer.
    "u",  # Unsigned integer.
    "f",  # Floating-point.
    "c",  # Complex floating-point.
    "S",  # Byte string.
    "U",  # Unicode string, encoded to bytes before it is written.
})
"""The kinds of dtype of a property that an HDF file can store."""


def find_hdf_writing_error(name: str, property_values: ndarray) -> str:
    """Return why a property cannot be written to an HDF file.

    Only the first reason is returned for a given property,
    since a property that fails one check need not be submitted to the next,
    e.g. a property whose name is not a valid HDF name.

    Args:
        name: The name of the property.
        property_values: The property.

    Returns:
        The reason why the property cannot be written to an HDF file,
        empty if it can.
    """
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        return (
            f"The property name {name!r} of the catalog "
            "is not a valid HDF name and cannot be written to an HDF file."
        )

    try:
        encode(name, "utf-8")
    except UnicodeEncodeError:
        return (
            f"The property name {name!r} of the catalog cannot be encoded "
            "to UTF-8 and cannot be written to an HDF file."
        )

    kind = property_values.dtype.kind
    if kind == "O":
        return (
            f"The property {name!r} of the catalog holds Python objects "
            "and cannot be written to an HDF file."
        )

    if kind not in _hdf_writable_dtype_kinds:
        return (
            f"The property {name!r} of the catalog has the dtype "
            f"{property_values.dtype} and cannot be written to an HDF file."
        )

    if (
        kind == "U"
        and (value := find_unencodable_utf8_value(property_values)) is not None
    ):
        return (
            f"The value {value!r} of the property {name!r} of the "
            "catalog cannot be encoded to UTF-8 "
            "and cannot be written to an HDF file."
        )

    return ""


def find_unencodable_utf8_value(values: ndarray) -> str | None:
    """Find a value of an array of strings that UTF-8 cannot encode.

    Args:
        values: An array of strings.

    Returns:
        The first value of the array that UTF-8 cannot encode,
        `None` if UTF-8 can encode them all.
    """
    try:
        encode(values, "utf-8")
    except UnicodeEncodeError:
        pass
    else:
        return None

    for value in values:
        try:
            encode(value, "utf-8")
        except UnicodeEncodeError:  # noqa: PERF203
            return str(value)

    return None  # pragma: no cover


def read_hdf_property(dataset: Dataset) -> ndarray | list[str]:
    """Read a property from an HDF dataset, decoding it if it was encoded.

    Args:
        dataset: The HDF dataset storing the property.

    Returns:
        The property, with its bytes decoded to strings
        if it was a unicode property encoded to bytes before it was written.
    """
    property_values = dataset[()]
    if dataset.attrs.get(was_unicode_attribute, False):
        return [value.decode() for value in property_values]

    return property_values
