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
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Final

import h5py
from numpy import array
from numpy import bytes_
from numpy import float64
from numpy import genfromtxt
from pandas import DataFrame

from gemseo.space._design.constants import catalog_group
from gemseo.space._design.constants import choices_group
from gemseo.space._design.constants import choices_separator
from gemseo.space._design.constants import design_space_group
from gemseo.space._design.constants import lb_group
from gemseo.space._design.constants import names_group
from gemseo.space._design.constants import size_group
from gemseo.space._design.constants import table_names
from gemseo.space._design.constants import ub_group
from gemseo.space._design.constants import value_group
from gemseo.space._design.constants import var_type_group
from gemseo.space.catalog.catalog import Catalog
from gemseo.space.variable import CatalogVariable
from gemseo.space.variable import DataType
from gemseo.space.variable import DiscreteVariable
from gemseo.space.variable.factory import deterministic_variable_factory
from gemseo.util._numpy import int64_dtype
from gemseo.util.hdf5 import get_hdf5_group
from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any

    from numpy import ndarray

    from gemseo.space.base import BaseVariableSpace
    from gemseo.space.design import DesignSpace
    from gemseo.space.variable import BaseDeterministicVariable
    from gemseo.util.typing import NumberArray

minimal_fields: Final[tuple[str, ...]] = ("name", "lower_bound", "upper_bound")
"""The minimal fields required in a design space CSV file."""

_domain_payload_nouns: Final[Mapping[DataType, tuple[str, str]]] = MappingProxyType({
    DataType.DISCRETE: ("no choices", "choices"),
    DataType.CATALOG: ("no catalog", "a catalog"),
})
"""The noun phrases of the domain payload of a variable type, from its HDF group.

A type maps to the phrases saying that the payload is missing and present,
since the name of its HDF group does not read as one:
"has catalog" is not a sentence.
"""


def to_real(data: NumberArray) -> NumberArray:
    """Cast a possibly-complex array to a real `float64` array.

    Args:
        data: The array to cast.

    Returns:
        The real `float64` array.
    """
    return array(array(data, copy=False).real, dtype=float64)


def get_dataset(group: h5py.Group, name: str) -> ndarray | None:
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


def check_structure_is_unchanged(
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
    stored_names = [name.decode() for name in get_hdf5_group(space_group, names_group)]
    names = list(design_space.variables)
    if names != stored_names:
        # The variables are compared one by one only when they match,
        # otherwise a variable may be missing from the HDF file.
        error_messages.append(
            f"The names of the design variables are {stored_names}; got {names}."
        )
    else:
        for name, variable in design_space.variables.items():
            variable_group = get_hdf5_group(space_group, name)
            stored_size = get_hdf5_group(variable_group, size_group)[()]
            if stored_size != variable.size:
                error_messages.append(
                    f"The size of the design variable {name!r} is {stored_size}; "
                    f"got {variable.size}."
                )

            stored_type = get_hdf5_group(variable_group, var_type_group)[0].decode()
            # A file written by a past release stores the data type value of that
            # release, so normalize it before comparing;
            # a stored value that is not a data type at all is left as is,
            # and the message below reports it as it stands.
            stored_data_type = DataType._resolve_value(stored_type)

            if stored_data_type != variable.type:
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


def write_dataset(group: h5py.Group, name: str, data: ndarray) -> None:
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


def check_catalogs_are_hdf_writable(space: BaseVariableSpace) -> None:
    """Check that the catalogs of the catalog variables of a space can be written.

    A space without catalog variables, e.g. a random space, passes the check.

    Args:
        space: The space of variables.

    Raises:
        ValueError: If the catalog of a catalog variable cannot be written to
            HDF, see
            [Catalog.check_hdf_writable][gemseo.space.catalog.catalog.Catalog.check_hdf_writable].
    """
    for variable in space.variables.values():
        if isinstance(variable, CatalogVariable):
            variable.catalog.check_hdf_writable()


def check_catalogs_are_unchanged(
    space: BaseVariableSpace, space_group: h5py.Group, file_path: str | Path
) -> None:
    """Check that the catalogs of a space are those stored in an HDF file.

    The value of a catalog variable is a position in its catalog,
    so the values stored in the file only designate the right alternatives
    with the catalogs they were written with.
    Only the catalog variables of the space stored with a catalog
    in the HDF group are compared;
    a space without catalog variables, e.g. a random space, passes the check.

    Args:
        space: The space of variables.
        space_group: The HDF group of the reference design space.
        file_path: The HDF file path.

    Raises:
        ValueError: If the catalog of a catalog variable differs
            from the stored one.
    """
    changed_names = []
    for name, variable in space.variables.items():
        if not isinstance(variable, CatalogVariable):
            continue

        catalog_hdf_group = space_group.get(f"{name}/{catalog_group}")
        if catalog_hdf_group is None:
            continue

        if Catalog.read_hdf(catalog_hdf_group) != variable.catalog:
            changed_names.append(name)

    if changed_names:
        errors = "\n".join(
            f"- The catalog of the design variable {name!r} "
            "differs from the stored one."
            for name in changed_names
        )
        msg = (
            f"The design space stored in the node {space_group.parent.name!r} "
            f"of the HDF file {file_path} has different catalogs; "
            "the value of a catalog variable is a position in its catalog, "
            f"so the stored values are only valid with the stored catalog:\n{errors}"
        )
        raise ValueError(msg)


def check_stored_catalogs_are_unchanged(
    space: BaseVariableSpace, file_path: str | Path, hdf_node_path: str = ""
) -> None:
    """Check that the catalogs of a space are those stored in an HDF file.

    The file is opened read-only,
    so that this check can run before the file is opened for writing,
    and a file that does not exist
    or that stores no design space at this node passes the check.

    Args:
        space: The space of variables.
        file_path: The HDF file path.
        hdf_node_path: The path of the HDF file node.
            If empty, the root node is used.

    Raises:
        ValueError: If the catalog of a catalog variable differs
            from the stored one, see
            [check_catalogs_are_unchanged][gemseo.space._design.io.check_catalogs_are_unchanged].
    """
    if not Path(file_path).exists() or not space.variables.has_variables_of_type(
        DataType.CATALOG
    ):
        return

    with h5py.File(file_path, "r") as h5file:
        node = h5file.get(hdf_node_path) if hdf_node_path else h5file
        if node is None:
            return

        space_group = node.get(design_space_group)
        if space_group is not None:
            check_catalogs_are_unchanged(space, space_group, file_path)


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
            (variables, sizes and types)
            and the same catalogs;
            the bounds are overwritten,
            and the current value is overwritten,
            or removed when the exported design space has none.
        hdf_node_path: The path of the HDF file node.
            If empty, the root node is used.

    Raises:
        ValueError: If the file already stores a design space with a different
            structure or different catalogs at this node,
            or if the catalog of a catalog variable cannot be written to
            HDF, see
            [Catalog.check_hdf_writable][gemseo.space.catalog.catalog.Catalog.check_hdf_writable].
    """
    mode = "a" if append else "w"

    # Every catalog is checked before the file is opened, so a design space
    # holding a catalog that cannot be written (e.g. an object property, a
    # datetime property, or a bad property name) leaves the file on disk
    # untouched instead of truncating it (append=False) or leaving it with a
    # half-written, unreadable design space (append=True).
    check_catalogs_are_hdf_writable(design_space)

    with h5py.File(file_path, mode) as h5file:
        if hdf_node_path:
            h5file = h5file.require_group(hdf_node_path)

        space_group = h5file.get(design_space_group)
        if space_group is None:
            space_group = h5file.create_group(design_space_group)
            space_group.create_dataset(
                names_group, data=array(tuple(design_space.variables), dtype=bytes_)
            )
        else:
            # Both checks run before anything is written,
            # so a failing one leaves the file untouched.
            check_structure_is_unchanged(design_space, space_group, file_path)
            check_catalogs_are_unchanged(design_space, space_group, file_path)

        for name, variable in design_space.variables.items():
            variable_group = space_group.require_group(name)

            if isinstance(variable, DiscreteVariable):
                # A variable cannot lose its choices from one export to the next:
                # check_structure_is_unchanged() rejects a change of type.
                write_dataset(
                    variable_group,
                    choices_group,
                    array(variable.choices, copy=False),
                )

            if (
                isinstance(variable, CatalogVariable)
                and catalog_group not in variable_group
            ):
                # A stored catalog is equal to this one,
                # as check_catalogs_are_unchanged() passed,
                # so it is only written when missing.
                # The catalog was already checked writable above,
                # before the file was opened.
                variable.catalog.write_hdf(variable_group.create_group(catalog_group))

            write_dataset(
                variable_group, size_group, array(variable.size, dtype=int64_dtype)
            )
            write_dataset(
                variable_group, lb_group, array(variable.lower_bound, copy=False)
            )
            write_dataset(
                variable_group, ub_group, array(variable.upper_bound, copy=False)
            )
            write_dataset(
                variable_group,
                var_type_group,
                array([variable.type] * variable.size, dtype="bytes"),
            )

            value = design_space._current_value.get(name)
            if value is None:
                if value_group in variable_group:
                    del variable_group[value_group]
            else:
                write_dataset(variable_group, value_group, to_real(value))


def _check_domain_payload(
    file_path: str | Path,
    name: str,
    var_type: str,
    payloads: Mapping[DataType, Any],
) -> None:
    """Check that the domain payloads of a variable match its type.

    A variable of a type of `_domain_payload_nouns` requires the payload
    of this type, e.g. the choices of a discrete variable,
    and any other type requires none of them.

    Args:
        file_path: The path to the HDF file, for the error message.
        name: The name of the variable.
        var_type: The type of the variable.
        payloads: The payload read from the file for each type of
            `_domain_payload_nouns`, `None` when the file has none.

    Raises:
        ValueError: If the payloads do not match the type of the variable,
            naming every mismatch at once.
    """
    # A variable can disagree with several payloads at once, e.g. a float
    # variable stored with choices and a catalog, so gather the mismatches
    # and report them together rather than one import attempt at a time.
    error_messages = []
    for data_type, (missing_payload, payload_noun) in _domain_payload_nouns.items():
        payload = payloads[data_type]
        if var_type == data_type and payload is None:
            error_messages.append(
                f"has {missing_payload} for the variable {name!r} "
                f"of type {data_type.value!r}"
            )
        elif var_type != data_type and payload is not None:
            error_messages.append(
                f"has {payload_noun} for the variable {name!r} "
                f"of type {var_type!r} instead of {data_type.value!r}"
            )

    if not error_messages:
        return

    prefix = f"Malformed DesignSpace input file {file_path}"
    if len(error_messages) == 1:
        msg = f"{prefix} {error_messages[0]}."
    else:
        errors = "\n".join(f"- it {error_message}." for error_message in error_messages)
        msg = f"{prefix}:\n{errors}"

    raise ValueError(msg)


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
        space_group = get_hdf5_group(h5file, design_space_group)
        variable_names = get_hdf5_group(space_group, names_group)
        for name in variable_names:
            name = name.decode()
            variable_group = get_hdf5_group(space_group, name)
            l_b = get_dataset(variable_group, lb_group)
            u_b = get_dataset(variable_group, ub_group)
            var_type = get_dataset(variable_group, var_type_group)[0]
            value = get_dataset(variable_group, value_group)
            size = get_hdf5_group(variable_group, size_group)[()]
            choices = get_dataset(variable_group, choices_group)
            catalog_hdf_group = variable_group.get(catalog_group)
            decoded_var_type = (
                var_type.decode() if isinstance(var_type, bytes) else var_type
            )
            _check_domain_payload(
                file_path,
                name,
                decoded_var_type,
                {DataType.DISCRETE: choices, DataType.CATALOG: catalog_hdf_group},
            )
            if choices is not None:
                # The bounds of a discrete variable are derived from its choices,
                # so the persisted ones are informational.
                design_space._register_variable(
                    name,
                    deterministic_variable_factory.create(var_type, choices=choices),
                    value,
                )
            elif catalog_hdf_group is not None:
                # Likewise, the bounds of a catalog variable are derived
                # from its catalog.
                design_space._register_variable(
                    name,
                    deterministic_variable_factory.create(
                        var_type, catalog=Catalog.read_hdf(catalog_hdf_group)
                    ),
                    value,
                )
            else:
                design_space.add_variable(
                    name,
                    size=size,
                    type_=var_type,
                    lower_bound=l_b,
                    upper_bound=u_b,
                    value=value,
                )
    design_space.check()
    return design_space


def format_choices_cell(variable: BaseDeterministicVariable) -> str:
    """Return the CSV cell storing the choices of a variable.

    Args:
        variable: The variable.

    Returns:
        The choices separated by `choices_separator`,
        or `"None"` when the variable is not discrete.
    """
    if not isinstance(variable, DiscreteVariable):
        # An empty cell would make a whitespace-delimited file unreadable.
        return "None"

    return choices_separator.join(str(value) for value in variable.choices)


def to_dataframe(design_space: DesignSpace) -> DataFrame:
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
    choices: list[str] = []
    for name, variable in design_space.variables.items():
        curr = design_space._current_value.get(name)
        cell = format_choices_cell(variable)
        for i in range(variable.size):
            variable_names.append(name)
            variable_types.append(variable.type)
            lower_bounds.append(variable.lower_bound[i])
            upper_bounds.append(variable.upper_bound[i])
            choices.append(cell)
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
    if design_space.variables.has_variables_of_type(DataType.DISCRETE):
        # Do not add a column that every variable would leave empty.
        data[choices_group] = choices
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
    separator = delimiter or " "
    columns = list(fields) if fields else list(table_names)
    if design_space.variables.has_variables_of_type(DataType.CATALOG):
        names = tuple(
            name
            for name, variable in design_space.variables.items()
            if isinstance(variable, CatalogVariable)
        )
        plural = len(names) > 1
        variables = "catalog variables" if plural else "a catalog variable"
        catalogs = "catalogs" if plural else "catalog"
        verb = "do" if plural else "does"
        msg = (
            f"A design space holding {variables} cannot be exported "
            f"to CSV because the {catalogs} of {pretty_str(names, sort=False)} "
            f"{verb} not fit in a CSV cell; use to_hdf instead."
        )
        raise ValueError(msg)

    if design_space.variables.has_variables_of_type(DataType.DISCRETE):
        if separator == choices_separator:
            msg = (
                "A design space holding a discrete variable cannot be exported "
                f"with {choices_separator!r} as delimiter, "
                "which separates the choices within a cell."
            )
            raise ValueError(msg)

        # A file written by to_csv must always be readable back by from_csv,
        # whichever way fields was supplied.
        if choices_group not in columns:
            columns.append(choices_group)

        type_field = table_names[-1]
        if type_field not in columns:
            columns.append(type_field)
    elif choices_group in columns:
        # to_dataframe() below does not emit this column when there is no
        # discrete variable; drop it here instead of letting DataFrame.to_csv()
        # raise a bare pandas KeyError for an explicitly requested column.
        columns = [column for column in columns if column != choices_group]

    dataframe = to_dataframe(design_space)
    dataframe.to_csv(
        Path(output_file),
        sep=separator,
        index=False,
        columns=columns,
        na_rep="None",
    )


def read_choices_cell(
    str_data: ndarray, col_map: dict[str, int], row: int
) -> list[float] | None:
    """Read the choices of a variable from a CSV row.

    Args:
        str_data: The CSV data read as strings.
        col_map: The map from a field name to its column index.
        row: The index of the row of the variable.

    Returns:
        The choices,
        or `None` when the file has no such column
        or the variable is not discrete.
    """
    index = col_map.get(choices_group)
    if index is None:
        return None

    cell = str_data[row, index]
    if not cell or cell == "None":
        return None

    return [float(value) for value in cell.split(choices_separator)]


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
    if not set(minimal_fields).issubset(set(header)):
        msg = (
            f"Malformed DesignSpace input file {file_path} does not contain "
            f"minimal variables in header:{minimal_fields}; got instead: {header}."
        )
        raise ValueError(msg)
    col_map = {field: i for i, field in enumerate(header)}
    name_field = minimal_fields[0]
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
    lower_bounds_field = minimal_fields[1]
    upper_bounds_field = minimal_fields[2]
    value_field = table_names[2]
    var_type_field = table_names[-1]
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
            var_type = cls.DesignVariableType.REAL
        choices = read_choices_cell(str_data, col_map, k)
        if choices is not None and var_type != cls.DesignVariableType.DISCRETE:
            msg = (
                f"Malformed DesignSpace input file {file_path} has choices "
                f"for the variable {name!r} of type {str(var_type)!r} "
                f"instead of {cls.DesignVariableType.DISCRETE.value!r}."
            )
            raise ValueError(msg)

        if choices is None:
            if var_type == cls.DesignVariableType.DISCRETE:
                msg = (
                    f"Malformed DesignSpace input file {file_path} has no "
                    f"choices for the variable {name!r} of type "
                    f"{cls.DesignVariableType.DISCRETE.value!r}."
                )
                raise ValueError(msg)

            design_space.add_variable(name, size, var_type, l_b, u_b, value)
        else:
            # The bounds of a discrete variable are derived from its choices,
            # so the persisted ones are informational.
            design_space._register_variable(
                name,
                deterministic_variable_factory.create(var_type, choices=choices),
                value,
            )
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
