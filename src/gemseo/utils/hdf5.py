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

import h5py


def get_hdf5_group(
    h5py_data: h5py.File | h5py.Group,
    name: str,
) -> h5py.Group:
    """Return a group from a h5py data handle.

    This function shall be used to show a better error message to the end user.

    Args:
        h5py_data: The hdf5 data handle.
        name: The name of the group.

    Returns:
        The contents of the group.

    Raises:
        KeyError: if the group does not exist.
    """
    try:
        return h5py_data[name]
    except KeyError as err:
        raise KeyError(f"In HDF5 file {h5py_data.file}: no such group {err.args[0]}.")
