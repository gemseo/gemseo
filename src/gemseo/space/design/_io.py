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
"""HDF and CSV serialization for a design space."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Final

import h5py
from numpy import array
from numpy import bytes_
from numpy import float64
from numpy import genfromtxt
from pandas import DataFrame

from gemseo.space._variable import DataType
from gemseo.space.design._constants import _DESIGN_SPACE_GROUP
from gemseo.space.design._constants import _LB_GROUP
from gemseo.space.design._constants import _NAMES_GROUP
from gemseo.space.design._constants import _SIZE_GROUP
from gemseo.space.design._constants import _TABLE_NAMES
from gemseo.space.design._constants import _UB_GROUP
from gemseo.space.design._constants import _VALUE_GROUP
from gemseo.space.design._constants import _VAR_TYPE_GROUP
from gemseo.util.hdf5 import get_hdf5_group

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from numpy import ndarray

    from gemseo.space.design import DesignSpace

_MINIMAL_FIELDS: Final[list[str]] = ["name", "lower_bound", "upper_bound"]
"""The minimal fields required in a design space CSV file."""


def _to_real(data: ndarray) -> ndarray:
    """Cast a possibly-complex array to a real `float64` array.

    Args:
        data: The array to cast.

    Returns:
        The real `float64` array.
    """
    return array(array(data, copy=False).real, dtype=float64)


def _read_opt_attr_array(var_group: h5py.Group, dataset_name: str) -> ndarray | None:
    """Read a dataset from an HDF group if it exists.

    Args:
        var_group: The HDF group of a variable.
        dataset_name: The name of the dataset to read.

    Returns:
        The dataset as an array, or `None` if it does not exist.
    """
    data = var_group.get(dataset_name)
    if data is not None:
        data = array(data)
    return data


def to_hdf(
    design_space: DesignSpace,
    file_path: str | Path,
    append: bool = False,
    hdf_node_path: str = "",
) -> None:
    """Export a design space to an HDF file.

    Args:
        design_space: The design space.
        file_path: The path to the file.
        append: If `True`, append to the file.
        hdf_node_path: The path of the HDF node in which the design space
            should be exported. If empty, the root node is used.
    """
    cls = type(design_space)
    int_dtype = cls.VARIABLE_TYPES_TO_DTYPES[cls.DesignVariableType.INTEGER]
    mode = "a" if append else "w"

    with h5py.File(file_path, mode) as h5file:
        if hdf_node_path:
            h5file = h5file.require_group(hdf_node_path)
        design_vars_grp = h5file.require_group(_DESIGN_SPACE_GROUP)
        name_array = array(design_space.variable_names, dtype=bytes_)
        names_dataset = design_vars_grp.require_dataset(
            _NAMES_GROUP, name_array.shape, name_array.dtype
        )
        names_dataset[...] = name_array

        for name, variable in design_space._variables.items():
            var_grp = design_vars_grp.require_group(name)
            size_ds = var_grp.require_dataset(_SIZE_GROUP, (), dtype=int_dtype)
            size_ds[...] = variable.size

            lb = array(variable.lower_bound, copy=False)
            lb_ds = var_grp.require_dataset(_LB_GROUP, lb.shape, lb.dtype)
            lb_ds[...] = lb

            ub = array(variable.upper_bound, copy=False)
            ub_ds = var_grp.require_dataset(_UB_GROUP, ub.shape, ub.dtype)
            ub_ds[...] = ub

            data_array = array([variable.type] * variable.size, dtype="bytes")
            type_ds = var_grp.require_dataset(
                _VAR_TYPE_GROUP, data_array.shape, data_array.dtype
            )
            type_ds[...] = data_array

            value = design_space._current_value.get(name)
            if value is not None:
                real_val = _to_real(value)
                val_ds = var_grp.require_dataset(
                    _VALUE_GROUP, real_val.shape, real_val.dtype
                )
                val_ds[...] = real_val


def from_hdf(
    cls: type[DesignSpace], file_path: str | Path, hdf_node_path: str = ""
) -> DesignSpace:
    """Create a design space from an HDF file.

    Args:
        cls: The DesignSpace class (or subclass) to instantiate.
        file_path: The path to the HDF file.
        hdf_node_path: The path of the HDF node from which the design space
            should be imported. If empty, the root node is used.

    Returns:
        The design space.
    """
    design_space = cls()
    with h5py.File(file_path) as h5file:
        h5file = get_hdf5_group(h5file, hdf_node_path)
        design_vars_grp = get_hdf5_group(h5file, _DESIGN_SPACE_GROUP)
        variable_names = get_hdf5_group(design_vars_grp, _NAMES_GROUP)
        for name in variable_names:
            name = name.decode()
            var_group = get_hdf5_group(design_vars_grp, name)
            l_b = _read_opt_attr_array(var_group, _LB_GROUP)
            u_b = _read_opt_attr_array(var_group, _UB_GROUP)
            var_type = _read_opt_attr_array(var_group, _VAR_TYPE_GROUP)[0]
            value = _read_opt_attr_array(var_group, _VALUE_GROUP)
            size = get_hdf5_group(var_group, _SIZE_GROUP)[()]
            design_space.add_variable(name, size, var_type, l_b, u_b, value)
    design_space.check()
    return design_space


def _to_dataframe(design_space: DesignSpace) -> DataFrame:
    """Export a design space to a pandas `DataFrame`.

    Args:
        design_space: The design space.

    Returns:
        The design space as a `DataFrame` with one row per scalar component.
    """
    variable_names: list[str] = []
    variable_values: list = []
    lower_bounds: list = []
    upper_bounds: list = []
    variable_types: list[str] = []
    for name, variable in design_space._variables.items():
        curr = design_space._current_value.get(name)
        for i in range(variable.size):
            variable_names.append(name)
            variable_types.append(variable.type)
            lower_bounds.append(variable.lower_bound[i])
            upper_bounds.append(variable.upper_bound[i])
            if curr is None:
                value = None
            else:
                value = curr[i]
                if variable.type == DataType.FLOAT:
                    value = value.real
            variable_values.append(value)
    data = {
        "name": variable_names,
        "value": variable_values,
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds,
        "type": variable_types,
    }
    return DataFrame(data)


def to_csv(
    design_space: DesignSpace,
    output_file: str | Path,
    fields: Sequence[str] = (),
    delimiter: str = " ",
) -> None:
    """Export a design space to a CSV file.

    Args:
        design_space: The design space.
        output_file: The path to the CSV file.
        fields: The fields to be exported. If empty, export all fields.
        delimiter: The string used to separate values.
    """
    dataframe = _to_dataframe(design_space)
    dataframe.to_csv(
        Path(output_file),
        sep=delimiter or " ",
        index=False,
        columns=fields or _TABLE_NAMES,
        na_rep="None",
    )


def from_csv(
    cls: type[DesignSpace],
    file_path: str | Path,
    header: Iterable[str] = (),
    delimiter: str = "",
) -> DesignSpace:
    """Create a design space from a CSV file.

    Args:
        cls: The DesignSpace class (or subclass) to instantiate.
        file_path: The path to the CSV file.
        header: The names of the fields saved in the file.
            If empty, read them from the file.
        delimiter: The delimiter. If empty, any whitespace acts as delimiter.

    Returns:
        The design space.

    Raises:
        ValueError: If the file does not contain the minimal variables in
            its header.
    """
    design_space = cls()
    float_data = genfromtxt(file_path, delimiter=delimiter or None, dtype="float")
    str_data = genfromtxt(file_path, delimiter=delimiter or None, dtype="str")
    if header:
        start_read = 0
    else:
        header = str_data[0, :].tolist()
        start_read = 1
    if not set(_MINIMAL_FIELDS).issubset(set(header)):
        msg = (
            f"Malformed DesignSpace input file {file_path} does not contain "
            f"minimal variables in header:{_MINIMAL_FIELDS}; got instead: {header}."
        )
        raise ValueError(msg)
    col_map = {field: i for i, field in enumerate(header)}
    name_field = _MINIMAL_FIELDS[0]
    var_names = str_data[start_read:, col_map[name_field]].tolist()
    unique_names: list[str] = []
    prev_name: str | None = None
    for name in var_names:
        if name not in unique_names:
            unique_names.append(name)
            prev_name = name
        elif prev_name != name:
            msg = (
                f"Malformed DesignSpace input file {file_path} contains some "
                f"variables ({name}) in a non-consecutive order."
            )
            raise ValueError(msg)

    k = start_read
    lower_bounds_field = _MINIMAL_FIELDS[1]
    upper_bounds_field = _MINIMAL_FIELDS[2]
    value_field = _TABLE_NAMES[2]
    var_type_field = _TABLE_NAMES[-1]
    for name in unique_names:
        size = var_names.count(name)
        l_b = float_data[k : k + size, col_map[lower_bounds_field]]
        u_b = float_data[k : k + size, col_map[upper_bounds_field]]
        if value_field in col_map:
            value = float_data[k : k + size, col_map[value_field]]
            if "None" in str_data[k : k + size, col_map[value_field]]:
                value = None
        else:
            value = None
        if var_type_field in col_map:
            var_type = str_data[k, col_map[var_type_field]]
        else:
            var_type = cls.DesignVariableType.FLOAT
        design_space.add_variable(name, size, var_type, l_b, u_b, value)
        k += size
    design_space.check()
    return design_space


def to_file(
    design_space: DesignSpace,
    file_path: str | Path,
    delimiter: str = " ",
    append: bool = False,
    fields: Sequence[str] = (),
) -> None:
    """Save a design space to either HDF or CSV depending on the extension.

    Args:
        design_space: The design space.
        file_path: The path to the file. An `.hdf`/`.h5` extension selects HDF,
            otherwise CSV is used.
        delimiter: The string used to separate values for CSV files.
        append: If `True`, append to the HDF file.
        fields: The fields to be exported for CSV files. If empty, export all.
    """
    file_path = Path(file_path)
    if file_path.suffix.startswith((".hdf", ".h5")):
        to_hdf(design_space, file_path, append=append)
    else:
        to_csv(design_space, file_path, delimiter=delimiter, fields=fields)


def from_file(
    cls: type[DesignSpace],
    file_path: str | Path,
    hdf_node_path: str = "",
    header: Iterable[str] = (),
    delimiter: str = "",
) -> DesignSpace:
    """Load a design space from either an HDF or a CSV file.

    Args:
        cls: The `DesignSpace` class (or subclass) to instantiate.
        file_path: The path to the file.
        hdf_node_path: The path of the HDF node to import from (HDF files only).
            If empty, the root node is used.
        header: The names of the CSV fields. If empty, read them from the file.
        delimiter: The CSV delimiter. If empty, any whitespace acts as delimiter.

    Returns:
        The design space defined in the file.
    """
    if h5py.is_hdf5(file_path):
        return from_hdf(cls, file_path, hdf_node_path)
    return from_csv(cls, file_path, header=header, delimiter=delimiter)
