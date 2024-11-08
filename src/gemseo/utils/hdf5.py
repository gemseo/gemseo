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
"""Helper functions for hdf5 data."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any

import h5py
from h5py import Group
from numpy import array
from numpy import bytes_
from numpy import issubdtype
from numpy import ndarray
from numpy import number
from numpy import object_

if TYPE_CHECKING:
    from numbers import Number

    from gemseo.typing import RealArray


def store_h5data(
    group: Any,
    data_array: RealArray[Number] | str | list[str | Number],
    dataset_name: str,
    dtype: str | None = None,
) -> None:
    """Store an array in a hdf5 file group.

    Args:
        group: The group pointer.
        data_array: The data to be stored.
        dataset_name: The name of the dataset to store the array.
        dtype: Numpy dtype or string. If ``None``, dtype('f') will be used.
    """
    if data_array is None or (isinstance(data_array, Iterable) and not len(data_array)):
        return
    if isinstance(data_array, ndarray):
        data_array = data_array.real
    if isinstance(data_array, str):
        data_array = array([data_array], dtype="bytes")
    if isinstance(data_array, list):
        all_str = reduce(
            lambda x, y: x or y,
            (isinstance(data, str) for data in data_array),
        )
        if all_str:
            data_array = array([data_array], dtype="bytes")
            dtype = data_array.dtype
    group.create_dataset(dataset_name, data=data_array, dtype=dtype)


def store_attr_h5data(obj: Any, group: Group) -> None:
    """Store an object in the HDF5 dataset.

    The object shall be a mapping or have a method to_dict().

    Args:
        obj: The object to store
        group: The hdf5 group.
    """
    from gemseo.algos.design_space import DesignSpace

    data = obj if isinstance(obj, Mapping) else obj.to_dict()
    for name, value in data.items():
        dtype = None
        if isinstance(value, str):
            value = value.encode("ascii", "ignore")
        elif isinstance(value, bytes):
            value = value.decode()
        elif isinstance(value, Mapping) and not isinstance(value, DesignSpace):
            grname = f"/{name}"
            if grname in group:
                del group[grname]
            new_group = group.require_group(grname)
            store_attr_h5data(value, new_group)
            continue
        elif hasattr(value, "__iter__") and not (
            isinstance(value, ndarray) and issubdtype(value.dtype, number)
        ):
            value = [
                att.encode("ascii", "ignore") if isinstance(att, str) else att
                for att in value
            ]
            dtype = h5py.special_dtype(vlen=str)

        store_h5data(group, value, name, dtype)


def convert_h5_group_to_dict(
    h5_handle: h5py.File | h5py.Group,
    group_name: str,
) -> dict[str, str | list[str]]:
    """Convert the values of a hdf5 dataset.

    Values that are of the kind string or bytes are converted
    to string or list of strings.

    Args:
        h5_handle: A hdf5 file or group.
        group_name: The name of the group to be converted.

    Returns:
        The converted dataset.
    """
    converted = {}

    group = get_hdf5_group(h5_handle, group_name)

    for key, value in group.items():
        value = value[()]

        # h5py does not handle bytes natively, it maps it to a numpy generic type
        if isinstance(value, ndarray) and value.dtype.type in {
            object_,
            bytes_,
        }:
            value = value[0] if value.size == 1 else value.tolist()

        if isinstance(value, bytes):
            value = value.decode()

        if isinstance(value, list):
            value = [
                sub_value.decode() if isinstance(sub_value, bytes) else sub_value
                for sub_value in value
            ]

        converted[key] = value

    return converted


def get_hdf5_group(
    h5py_data: h5py.File | h5py.Group,
    name: str = "",
) -> h5py.Group:
    """Return a group from a h5py data handle.

    This function shall be used to show a better error message to the end user.

    Args:
        h5py_data: The hdf5 data handle.
        name: The name of the group, if empty returns the root.

    Returns:
        The contents of the group.

    Raises:
        KeyError: if the group does not exist.
    """
    if name:
        try:
            return h5py_data[name]
        except KeyError as err:
            msg = f"In HDF5 file {h5py_data.file}: no such group {err.args[0]}."
            raise KeyError(msg) from None
    return h5py_data
