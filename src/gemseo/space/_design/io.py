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

from operator import itemgetter
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Final

import h5py
from numpy import array
from numpy import bytes_
from numpy import float64
from numpy import full
from numpy import genfromtxt
from numpy import isin
from numpy import nan
from numpy import where
from numpy.char import find
from pandas import DataFrame
from pydantic import ValidationError

from gemseo.space._design.constants import catalog_group
from gemseo.space._design.constants import categories_group
from gemseo.space._design.constants import choices_group
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
from gemseo.space.variable import DiscreteVariable
from gemseo.space.variable import VariableType
from gemseo.space.variable.factory import deterministic_variable_factory
from gemseo.util._numpy import int64_dtype
from gemseo.util.hdf5 import get_hdf5_group
from gemseo.util.string import convert_strings_to_iterable
from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any

    from numpy import ndarray

    from gemseo.space.base import BaseVariableSpace
    from gemseo.space.design import DesignSpace
    from gemseo.util.typing import NumberArray
    from gemseo.util.typing import RealArray

minimal_fields: Final[tuple[str, ...]] = ("name", "lower_bound", "upper_bound")
"""The minimal fields required in a design space CSV file."""

_csv_variable_types: Final[tuple[VariableType, ...]] = (
    VariableType.REAL,
    VariableType.INTEGER,
)
"""The types of the variables that a CSV file can store."""

_domain_payload_nouns: Final[Mapping[VariableType, tuple[str, str]]] = (
    MappingProxyType({
        VariableType.DISCRETE: ("no choices", "choices"),
        VariableType.CATALOG: ("no catalog", "a catalog"),
        VariableType.CATEGORICAL: ("no categories", "categories"),
    })
)
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


def get_strings(group: h5py.Group, name: str) -> list[str] | None:
    """Retrieve the strings stored in a dataset of an HDF group.

    Args:
        group: The HDF group.
        name: The name of the dataset.

    Returns:
        The strings, or `None` if the dataset does not exist.
    """
    dataset = group.get(name)
    if dataset is None:
        return None

    return [
        string.decode() if isinstance(string, bytes) else str(string)
        for string in dataset[()].ravel()
    ]


def check_structure_is_unchanged(
    design_space: DesignSpace, space_group: h5py.Group, file_path: str | Path
) -> None:
    """Check that a design space is consistent with that of an HDF file.

    Only the structure that the stored input values rely on is compared:
    the variable names, in order, the size and the type of each variable,
    and the categories, in order, of each categorical variable.

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
            # A file written by a past release stores the variable type value of that
            # release, so normalize it before comparing;
            # a stored value that is not a variable type at all is left as is,
            # and the message below reports it as it stands.
            stored_variable_type = VariableType._resolve_value(stored_type)

            if stored_variable_type != variable.type:
                error_messages.append(
                    f"The type of the design variable {name!r} is {stored_type!r}; "
                    f"got {variable.type.value!r}."
                )
            elif variable.type == VariableType.CATEGORICAL:
                # The stored values and the database rows are positions
                # in the list of categories, so its order matters.
                stored_categories = get_strings(variable_group, categories_group)
                if stored_categories != list(variable.categories):
                    error_messages.append(
                        f"The categories of the design variable {name!r} "
                        f"are {stored_categories}; got {list(variable.categories)}."
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
        VariableType.CATALOG
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

            is_categorical = variable.type == VariableType.CATEGORICAL
            if is_categorical:
                # The labels may be non-ASCII, hence variable-length UTF-8 strings.
                write_dataset(
                    variable_group,
                    categories_group,
                    array(variable.categories, dtype=h5py.string_dtype()),
                )

            write_dataset(
                variable_group, size_group, array(variable.size, dtype=int64_dtype)
            )
            if not is_categorical:
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
            elif is_categorical:
                # The file stores the label, not its position among the categories.
                write_dataset(
                    variable_group,
                    value_group,
                    array(
                        [str(label) for label in variable.decode(value)],
                        dtype=h5py.string_dtype(),
                    ),
                )
            else:
                write_dataset(variable_group, value_group, to_real(value))


def _check_domain_payload(
    file_path: str | Path,
    name: str,
    var_type: str,
    payloads: Mapping[VariableType, Any],
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
    for variable_type, (missing_payload, payload_noun) in _domain_payload_nouns.items():
        payload = payloads[variable_type]
        if var_type == variable_type and payload is None:
            error_messages.append(
                f"has {missing_payload} for the variable {name!r} "
                f"of type {variable_type.value!r}"
            )
        elif var_type != variable_type and payload is not None:
            error_messages.append(
                f"has {payload_noun} for the variable {name!r} "
                f"of type {var_type!r} instead of {variable_type.value!r}"
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
            var_type = get_dataset(variable_group, var_type_group)[0]
            decoded_var_type = (
                var_type.decode() if isinstance(var_type, bytes) else var_type
            )
            categories = get_strings(variable_group, categories_group)
            choices = get_dataset(variable_group, choices_group)
            catalog_hdf_group = variable_group.get(catalog_group)
            _check_domain_payload(
                file_path,
                name,
                decoded_var_type,
                {
                    VariableType.CATALOG: catalog_hdf_group,
                    VariableType.CATEGORICAL: categories,
                    VariableType.DISCRETE: choices,
                },
            )
            if categories is not None:
                labels = get_strings(variable_group, value_group)
                design_space.add_categorical_variable(
                    name, categories, None if labels is None else labels[0]
                )
                continue

            value = get_dataset(variable_group, value_group)
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
                l_b = get_dataset(variable_group, lb_group)
                u_b = get_dataset(variable_group, ub_group)
                size = get_hdf5_group(variable_group, size_group)[()]
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


def _inflect(count: int, singular: str, plural: str) -> str:
    """Return the form of a phrase agreeing with a number of items.

    Args:
        count: The number of items.
        singular: The phrase for a single item.
        plural: The phrase for several items.

    Returns:
        The singular phrase if there is a single item, otherwise the plural one.
    """
    return singular if count == 1 else plural


def _get_unsupported_variables_sentence(
    variable_types: Mapping[str, VariableType],
    subject: str,
    advice: str,
    plural_advice: str = "",
) -> str:
    """Return the sentence naming the variables that a CSV file cannot store.

    Args:
        variable_types: The type of each variable, in declaration order.
        subject: The subject of the sentence, e.g. `"The design space"`.
        advice: The end of the sentence telling what to do.
        plural_advice: The end of the sentence telling what to do
            when there are several such variables.
            If empty, use `advice`.

    Returns:
        The sentence naming every variable that a CSV file cannot store,
        if any, otherwise an empty string.
    """
    unsupported_variables = tuple(
        f"{name!r} ({variable_type})"
        for name, variable_type in variable_types.items()
        if variable_type not in _csv_variable_types
    )
    if not unsupported_variables:
        return ""

    n_unsupported_variables = len(unsupported_variables)
    variables = _inflect(
        n_unsupported_variables, "a variable that is", "variables that are"
    )
    return (
        f"{subject} has {variables} neither {' nor '.join(_csv_variable_types)}: "
        f"{pretty_str(unsupported_variables, sort=False)}; "
        f"{_inflect(n_unsupported_variables, advice, plural_advice or advice)}"
    )


def _get_fields_sentences(fields: Sequence[str]) -> tuple[str, ...]:
    """Return the sentences naming the problems of the fields to export to CSV.

    Args:
        fields: The fields to export.

    Returns:
        The sentences naming the unknown fields, the missing `"name"` field
        and the repeated fields,
        each one being an empty string when there is no such problem.
    """
    if not fields:
        return ()

    unknown_fields = tuple(
        dict.fromkeys(repr(field) for field in fields if field not in table_names)
    )
    unknown_sentence = ""
    if unknown_fields:
        unknown_sentence = (
            "A design space CSV file has no "
            f"{_inflect(len(unknown_fields), 'field', 'fields')} "
            f"{pretty_str(unknown_fields, sort=False)}; its fields are "
            f"{pretty_str(tuple(repr(field) for field in table_names), sort=False)}."
        )

    # As in the past releases, a file without bounds is written,
    # although from_csv cannot read it back;
    # a file without names would not even name its variables.
    name_field = minimal_fields[0]
    missing_sentence = ""
    if name_field not in fields:
        missing_sentence = f"A design space CSV file requires the field {name_field!r}."

    repeated_fields = tuple(
        dict.fromkeys(repr(field) for field in fields if fields.count(field) > 1)
    )
    repeated_sentence = ""
    if repeated_fields:
        n_repeated_fields = len(repeated_fields)
        repeated_sentence = (
            f"The {_inflect(n_repeated_fields, 'field', 'fields')} "
            f"{pretty_str(repeated_fields, sort=False)} "
            f"{_inflect(n_repeated_fields, 'is', 'are')} given more than once."
        )

    return unknown_sentence, missing_sentence, repeated_sentence


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
        fields: The fields to be exported, without repetition,
            including `"name"`;
            `"type"` is added when a variable is not real.
            If empty, export all fields.
        delimiter: The string used to separate values.

    Raises:
        ValueError: If the design space has a variable
            that is neither real nor integer,
            as only an HDF file can store it,
            if a field is unknown or repeated,
            or if `"name"` is not a field.
    """
    # A string is a sequence of strings, which would be read as one field per letter.
    fields = tuple(convert_strings_to_iterable(fields)) if fields else ()
    sentences = (
        _get_unsupported_variables_sentence(
            {name: variable.type for name, variable in design_space.variables.items()},
            "The design space",
            "save the design space to an HDF file instead.",
        ),
        *_get_fields_sentences(fields),
    )
    msg = " ".join(sentence for sentence in sentences if sentence)
    if msg:
        raise ValueError(msg)

    variable_names: list[str] = []
    variable_values: list = []
    lower_bounds: list = []
    upper_bounds: list = []
    variable_types: list[str] = []
    for name, variable in design_space.variables.items():
        curr = design_space._current_value.get(name)
        for i in range(variable.size):
            variable_names.append(name)
            variable_types.append(variable.type)
            lower_bounds.append(variable.lower_bound[i])
            upper_bounds.append(variable.upper_bound[i])
            # Strip the imaginary part of a complex-step perturbation.
            value = None if curr is None else curr[i].real
            variable_values.append(value)

    columns = list(fields) if fields else list(table_names)
    type_field = table_names[-1]
    if type_field not in columns and any(
        variable.type != VariableType.REAL
        for variable in design_space.variables.values()
    ):
        # from_csv reads a variable as real when the file has no type.
        columns.append(type_field)

    DataFrame({
        "name": variable_names,
        "value": variable_values,
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds,
        "type": variable_types,
    }).to_csv(
        Path(output_file),
        sep=delimiter or " ",
        index=False,
        columns=columns,
        na_rep="None",
    )


_no_value_cells: Final[tuple[str, str]] = ("", "None")
"""The CSV cells meaning that a variable has no value."""


def _convert_cells_to_floats(
    cells: ndarray, no_value_rows: ndarray | None = None
) -> tuple[RealArray, ndarray]:
    """Convert CSV cells to floats as `float` does, except for underscores.

    Args:
        cells: The cells.
        no_value_rows: Whether the cell of a row means no value.
            If `None`, every cell must be a number.

    Returns:
        The floats, NaN for a cell that is not a number,
        and whether a cell is invalid,
        i.e. neither a number nor, if `no_value_rows` is given, a cell meaning no value.
    """
    # float() reads "1_0" as 10, which hides a typing error in the file.
    is_invalid = find(cells, "_") >= 0
    is_masked = is_invalid if no_value_rows is None else is_invalid | no_value_rows
    # Masking the cells that are not numbers lets a single cast convert the others.
    cells = where(is_masked, "nan", cells)
    try:
        # A cast from strings to floats converts each string as float() does.
        floats = cells.astype(float64)
    except ValueError:
        # Another cell is not a number, so convert them one by one.
        floats = full(len(cells), nan)
        for row, cell in enumerate(cells.tolist()):
            try:
                floats[row] = float(cell)
            except ValueError:
                is_invalid[row] = True

    return floats, is_invalid


def _check_minimal_fields(file_path: str | Path, header: tuple[str, ...]) -> None:
    """Check that the header of a CSV file has the minimal fields.

    Args:
        file_path: The path to the CSV file.
        header: The names of the fields saved in the file.

    Raises:
        ValueError: If the header does not have the minimal fields.
    """
    missing_fields = tuple(
        repr(field) for field in minimal_fields if field not in header
    )
    if missing_fields:
        msg = (
            f"The file {file_path} lacks the "
            f"{_inflect(len(missing_fields), 'field', 'fields')} "
            f"{pretty_str(missing_fields, sort=False)}."
        )
        raise ValueError(msg)


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
        ValueError: If the header of the file lacks a minimal field,
            if the file has fewer columns than `header` has fields,
            if the rows of a variable are not consecutive,
            if a variable has an unknown type,
            if the rows of a variable have different types,
            if the rows of the file have different numbers of cells,
            if a variable is neither real nor integer,
            as only an HDF file can store it,
            if a bound or value cell is not a number,
            if only some value cells of a variable mean no value,
            or if a variable has invalid bounds or an invalid value.
    """
    # The header is read several times, which would exhaust an iterator;
    # its truth value is not evaluated, as a NumPy array has none.
    # A string is an iterable of strings, which would be read as one field per letter.
    header = (
        tuple(map(str, convert_strings_to_iterable(header)))
        if header is not None
        else ()
    )
    design_space = cls()
    # Without ndmin, a file with a single row or column is read as a 1D array.
    # A cell can be padded with spaces when the delimiter is not a whitespace.
    try:
        str_data = genfromtxt(
            file_path,
            delimiter=delimiter or None,
            dtype="str",
            ndmin=2,
            autostrip=True,
        )
    except ValueError as error:
        # NumPy names the rows whose number of cells differs from the first row,
        # one per line after a heading line.
        bad_rows = "; ".join(line.strip() for line in str(error).splitlines()[1:])
        msg = (
            f"The file {file_path} has rows with different numbers of cells: "
            f"{bad_rows}."
        )
        raise ValueError(msg) from error
    if not header:
        if str_data.size:
            header = tuple(str_data[0, :].tolist())
            str_data = str_data[1:]
    # The columns after the last field of the header are ignored.
    elif str_data.size and (n_fields := len(header)) > (n_columns := str_data.shape[1]):
        msg = (
            f"The file {file_path} has {n_columns} "
            f"{_inflect(n_columns, 'column', 'columns')} but the header has "
            f"{n_fields} {_inflect(n_fields, 'field', 'fields')}: "
            f"{pretty_str(tuple(repr(field) for field in header), sort=False)}."
        )
        raise ValueError(msg)

    _check_minimal_fields(file_path, header)
    if not str_data.size:
        # to_csv writes a design space without variable as a header-only file.
        return design_space

    col_map = {field: i for i, field in enumerate(header)}
    name_field = minimal_fields[0]
    var_names = str_data[:, col_map[name_field]].tolist()
    # The map from a variable name to its first row and its number of rows.
    name_to_rows: dict[str, tuple[int, int]] = {}
    prev_name: str | None = None
    for row, name in enumerate(var_names):
        if name == prev_name:
            first_row, size = name_to_rows[name]
            name_to_rows[name] = (first_row, size + 1)
        elif name in name_to_rows:
            msg = (
                f"Malformed DesignSpace input file {file_path} contains some "
                f"variables ({name}) in a non-consecutive order."
            )
            raise ValueError(msg)
        else:
            name_to_rows[name] = (row, 1)
            prev_name = name

    var_type_field = table_names[-1]
    type_column = col_map.get(var_type_field)
    raw_to_type: dict[str, VariableType | str] = {}
    if type_column is not None:
        type_cells = str_data[:, type_column].tolist()
        # A file written by a past release may store a legacy type value;
        # each distinct value is resolved once, so as to warn once.
        # An empty type means real, as a file without types does.
        raw_to_type = {
            raw_type: VariableType._resolve_value(raw_type)
            if raw_type
            else VariableType.REAL
            for raw_type in dict.fromkeys(type_cells)
        }

    variable_types: dict[str, VariableType] = {}
    unknown_variables: list[str] = []
    mixed_variables: list[str] = []
    for name, (k, size) in name_to_rows.items():
        if type_column is None:
            variable_types[name] = VariableType.REAL
            continue

        row_types = tuple(
            dict.fromkeys(
                raw_to_type[raw_type] for raw_type in type_cells[k : k + size]
            )
        )
        # A variable whose rows mix a valid and an unknown type is reported twice,
        # lest the user fix the mix toward the unknown type.
        unknown_types = tuple(
            row_type for row_type in row_types if not isinstance(row_type, VariableType)
        )
        if unknown_types:
            unknown_variables.append(
                f"{name!r} ({pretty_str(unknown_types, sort=False)})"
            )

        if len(row_types) > 1:
            mixed_variables.append(f"{name!r} ({pretty_str(row_types, sort=False)})")
        elif not unknown_types:
            variable_types[name] = row_types[0]

    # Every variable is checked before any is added,
    # so that all the invalid ones are reported at once.
    subject = f"The file {file_path}"
    unknown_sentence = ""
    if unknown_variables:
        variables = _inflect(
            len(unknown_variables),
            "a variable of unknown type",
            "variables of unknown types",
        )
        unknown_sentence = (
            f"{subject} has {variables}: "
            f"{pretty_str(unknown_variables, sort=False)}; "
            "a CSV file only stores "
            f"{pretty_str(_csv_variable_types, sort=False)} variables."
        )

    # Each column is converted once, then sliced for each variable.
    value_field = table_names[2]
    field_to_floats: dict[str, RealArray] = {}
    # The rows and the descriptions of the cells that are not numbers.
    invalid_cells: list[tuple[int, str]] = []
    no_value_rows = None
    for field in (*minimal_fields[1:], value_field):
        column = col_map.get(field)
        if column is None:
            continue

        cells = str_data[:, column]
        is_value_field = field == value_field
        if is_value_field:
            no_value_rows = isin(cells, _no_value_cells)

        floats, is_invalid = _convert_cells_to_floats(
            cells, no_value_rows if is_value_field else None
        )
        field_to_floats[field] = floats
        # The cells of a variable already refused for its type are not reported.
        invalid_cells.extend(
            (row, f"{str(cells[row])!r} ({field!r} of {name!r})")
            for row in is_invalid.nonzero()[0].tolist()
            if variable_types.get(name := var_names[row]) in _csv_variable_types
        )

    invalid_sentence = ""
    if invalid_cells:
        # A stable sort keeps the order of the fields within a row.
        invalid_cells.sort(key=itemgetter(0))
        cells_that_are = _inflect(
            len(invalid_cells),
            "a cell that is not a number",
            "cells that are not numbers",
        )
        invalid_sentence = (
            f"{subject} has {cells_that_are}: "
            f"{pretty_str(tuple(cell for _, cell in invalid_cells), sort=False)}."
        )

    partial_sentence = ""
    if no_value_rows is not None:
        # A value cell meaning no value among numbers is likely a typing error,
        # so the other numbers are not dropped silently.
        partial_variables = []
        for name, (k, size) in name_to_rows.items():
            variable_no_value_rows = no_value_rows[k : k + size]
            if (
                variable_types.get(name) in _csv_variable_types
                and variable_no_value_rows.any()
                and not variable_no_value_rows.all()
            ):
                row_numbers = tuple((variable_no_value_rows.nonzero()[0] + 1).tolist())
                partial_variables.append(
                    f"{name!r} ({_inflect(len(row_numbers), 'row', 'rows')} "
                    f"{pretty_str(row_numbers, sort=False)})"
                )

        if partial_variables:
            variables = _inflect(len(partial_variables), "a variable", "variables")
            partial_sentence = (
                f"{subject} has {variables} with a value in some rows only: "
                f"{pretty_str(partial_variables, sort=False)}; "
                "a variable has a value in every row or in none."
            )

    mixed_sentence = ""
    if mixed_variables:
        mixed_sentence = (
            f"{subject} has {_inflect(len(mixed_variables), 'a variable', 'variables')}"
            " whose rows have different types: "
            f"{pretty_str(mixed_variables, sort=False)}; "
            "the rows of a variable must have the same type."
        )

    sentences = (
        unknown_sentence,
        mixed_sentence,
        _get_unsupported_variables_sentence(
            variable_types,
            subject,
            "only an HDF file can store it.",
            "only an HDF file can store them.",
        ),
        partial_sentence,
        invalid_sentence,
    )
    msg = " ".join(sentence for sentence in sentences if sentence)
    if msg:
        raise ValueError(msg)

    lower_bounds = field_to_floats[minimal_fields[1]]
    upper_bounds = field_to_floats[minimal_fields[2]]
    values = field_to_floats.get(value_field)
    for name, (k, size) in name_to_rows.items():
        rows = slice(k, k + size)
        value = None
        # The value cells of a variable all mean no value or none does.
        if values is not None and not no_value_rows[k]:
            value = values[rows]

        try:
            design_space.add_variable(
                name,
                size,
                variable_types[name],
                lower_bounds[rows],
                upper_bounds[rows],
                value,
            )
        except ValueError as error:
            # The bounds or the value of the variable are invalid,
            # as its name is unique and its type is valid.
            if isinstance(error, ValidationError):
                reason = " ".join(
                    str(details.get("ctx", {}).get("error", details["msg"]))
                    for details in error.errors()
                )
            else:
                reason = str(error)

            msg = f"The file {file_path} has an invalid variable {name!r}. {reason}"
            raise ValueError(msg) from error

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
        file_path: The path to the file.
            An extension starting with `".hdf"` or `".h5"`, e.g. `".hdf5"`,
            selects HDF, otherwise CSV is used.
        delimiter: The string used to separate values for CSV files.
        append: If `False`, the file is truncated
            and the design space is exported.
            If `True` and the file does not contain a design space,
            the design space is exported
            and the rest of the file is left untouched.
            If `True` and the file already contains a design space,
            both design spaces must have the same structure
            (variables, sizes and types)
            and the same catalogs;
            the bounds are overwritten,
            and the current value is overwritten,
            or removed when the exported design space has none.
            This argument is ignored for CSV files.
        fields: The fields to be exported for CSV files, without repetition,
            including `"name"`;
            `"type"` is added when a variable is not real.
            If empty, export all.

    Raises:
        ValueError: If the HDF file already stores a design space with a
            different structure,
            or, if `append` is `True`, with different catalogs
            for the catalog variables,
            or, if the file is not an HDF file,
            if the design space has a variable that is neither real nor integer,
            if a field is unknown or repeated,
            or if `"name"` is not a field.
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
