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

from gemseo.space.design._constants import _DESIGN_SPACE_GROUP
from gemseo.space.design._constants import _LB_GROUP
from gemseo.space.design._constants import _NAMES_GROUP
from gemseo.space.design._constants import _SIZE_GROUP
from gemseo.space.design._constants import _TABLE_NAMES
from gemseo.space.design._constants import _UB_GROUP
from gemseo.space.design._constants import _VALUE_GROUP
from gemseo.space.design._constants import _VAR_TYPE_GROUP
from gemseo.util._numpy import INT64_DTYPE
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


def _get_dataset(group: h5py.Group, name: str) -> ndarray | None:
    """Retrieve the dataset stored in an HDF group.

    Args:
        group: The HDF group.
        name: The name of the dataset.

    Returns:
        The dataset as an array, or `None` if it does not exist.
    """
    dataset = group.get(name)
    if dataset is not None:
        dataset = array(dataset)
    return dataset


def _check_structure_is_unchanged(
    design_space: DesignSpace, space_group: h5py.Group, file_path: str | Path
) -> None:
    """Check that a design space is consistent with that of an HDF file.

    Only the structure that the stored input values rely on is compared:
    the variable names, in order, and the size and the type of each variable.

    Args:
        design_space: The design space.
        space_group: The HDF group of the reference design space.
        file_path: The HDF file path.

    Raises:
        ValueError: If the design spaces are not consistent.
    """
    error_messages = []
    stored_names = [name.decode() for name in get_hdf5_group(space_group, _NAMES_GROUP)]
    if design_space.variable_names != stored_names:
        # The variables are compared one by one only when they match,
        # otherwise a variable may be missing from the HDF file.
        error_messages.append(
            f"The names of the design variables are {stored_names}; "
            f"got {design_space.variable_names}."
        )
    else:
        for name, variable in design_space._variables.items():
            variable_group = get_hdf5_group(space_group, name)
            stored_size = get_hdf5_group(variable_group, _SIZE_GROUP)[()]
            if stored_size != variable.size:
                error_messages.append(
                    f"The size of the design variable {name!r} is {stored_size}; "
                    f"got {variable.size}."
                )

            stored_type = get_hdf5_group(variable_group, _VAR_TYPE_GROUP)[0].decode()
            if stored_type != variable.type:
                error_messages.append(
                    f"The type of the design variable {name!r} is {stored_type!r}; "
                    f"got {variable.type.value!r}."
                )

    if error_messages:
        errors = "\n".join(f"- {error_message}" for error_message in error_messages)
        msg = (
            f"The design space stored in the node {space_group.parent.name!r} "
            f"of the HDF file {file_path} has a different structure:\n{errors}"
        )
        raise ValueError(msg)


def _write_dataset(group: h5py.Group, name: str, data: ndarray) -> None:
    """Write a dataset of an HDF group, creating it if needed.

    The data are written in place when the dataset can hold them,
    because HDF5 does not reclaim the space freed by a deletion.

    Args:
        group: The HDF group of the dataset.
        name: The name of the dataset.
        data: The data to write.
    """
    dataset = group.get(name)
    if dataset is not None and (
        dataset.shape != data.shape or dataset.dtype != data.dtype
    ):
        del group[name]
        dataset = None

    if dataset is None:
        group.create_dataset(name, data=data)
    else:
        dataset[...] = data


def to_hdf(
    design_space: DesignSpace,
    file_path: str | Path,
    append: bool = False,
    hdf_node_path: str = "",
) -> None:
    """Export the design space to an HDF file node.

    Args:
        design_space: The design space.
        file_path: The path to the file.
        append: If `False`, the file is truncated
            and the design space is exported to the node.
            If `True` and the node does not contain a design space,
            the design space is exported
            and the rest of the file is left untouched.
            If `True` and the node already contains a design space,
            both design spaces must have the same structure
            (variables, sizes and types);
            the bounds are overwritten,
            and the current value is overwritten,
            or removed when the exported design space has none.
        hdf_node_path: The path of the HDF file node.
            If empty, the root node is used.

    Raises:
        ValueError: If the file already stores a design space with a different
            structure at this node.
    """
    mode = "a" if append else "w"

    with h5py.File(file_path, mode) as h5file:
        if hdf_node_path:
            h5file = h5file.require_group(hdf_node_path)

        space_group = h5file.get(_DESIGN_SPACE_GROUP)
        if space_group is None:
            space_group = h5file.create_group(_DESIGN_SPACE_GROUP)
            space_group.create_dataset(
                _NAMES_GROUP, data=array(design_space.variable_names, dtype=bytes_)
            )
        else:
            _check_structure_is_unchanged(design_space, space_group, file_path)

        for name, variable in design_space._variables.items():
            variable_group = space_group.require_group(name)
            _write_dataset(
                variable_group, _SIZE_GROUP, array(variable.size, dtype=INT64_DTYPE)
            )
            _write_dataset(
                variable_group, _LB_GROUP, array(variable.lower_bound, copy=False)
            )
            _write_dataset(
                variable_group, _UB_GROUP, array(variable.upper_bound, copy=False)
            )
            _write_dataset(
                variable_group,
                _VAR_TYPE_GROUP,
                array([variable.type] * variable.size, dtype="bytes"),
            )

            value = design_space._current_value.get(name)
            if value is None:
                if _VALUE_GROUP in variable_group:
                    del variable_group[_VALUE_GROUP]
            else:
                _write_dataset(variable_group, _VALUE_GROUP, _to_real(value))


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
        space_group = get_hdf5_group(h5file, _DESIGN_SPACE_GROUP)
        variable_names = get_hdf5_group(space_group, _NAMES_GROUP)
        for name in variable_names:
            name = name.decode()
            variable_group = get_hdf5_group(space_group, name)
            l_b = _get_dataset(variable_group, _LB_GROUP)
            u_b = _get_dataset(variable_group, _UB_GROUP)
            var_type = _get_dataset(variable_group, _VAR_TYPE_GROUP)[0]
            value = _get_dataset(variable_group, _VALUE_GROUP)
            size = get_hdf5_group(variable_group, _SIZE_GROUP)[()]
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
            # Strip the imaginary part of a complex-step perturbation.
            value = None if curr is None else curr[i].real
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
        append: If `False`, the file is truncated
            and the design space is exported.
            If `True` and the file does not contain a design space,
            the design space is exported
            and the rest of the file is left untouched.
            If `True` and the file already contains a design space,
            both design spaces must have the same structure
            (variables, sizes and types);
            the bounds are overwritten,
            and the current value is overwritten,
            or removed when the exported design space has none.
            This argument is ignored for CSV files.
        fields: The fields to be exported for CSV files. If empty, export all.

    Raises:
        ValueError: If the HDF file already stores a design space with a
            different structure.
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
