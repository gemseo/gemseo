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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Benoit Pauwels - Stacked data management
#               (e.g. iteration index)
"""A database of function calls and design variables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Union

import h5py
from numpy import array
from numpy import bytes_
from numpy import float64
from numpy import ndarray

from gemseo.utils.hdf5 import get_hdf5_group

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike

    from gemseo.algos.database import Database
    from gemseo.algos.database import DatabaseValueType
    from gemseo.algos.hashable_ndarray import HashableNdarray

ReturnedHdfMissingOutputType = tuple[
    Mapping[str, Union[float, ndarray, list[int]]], Mapping[str, int]
]


class HDFDatabase:
    """Capabilities to export a database to an HDF file."""

    __pending_arrays: list[HashableNdarray]
    """A buffer of input values.

    To temporary save the last input values that have been stored before calling
    :meth:`.to_file`.

    It is used to append the exported HDF file.
    """

    def __init__(self) -> None:  # noqa:D107
        self.__pending_arrays = []

    @staticmethod
    def __to_real(data: ArrayLike) -> ndarray:
        """Convert complex to real NumPy array.

        Args:
            data: The original data.

        Returns:
            The real data.
        """
        return array(array(data, copy=False).real, dtype=float64)

    @staticmethod
    def __add_hdf_input_dataset(
        index_dataset: int,
        design_vars_group: h5py.Group,
        design_vars_values: HashableNdarray,
    ) -> None:
        """Add a new input to the HDF group of input values.

        Args:
            index_dataset: The index of the new HDF entry.
            design_vars_group: The HDF group of the design variable values.
            design_vars_values: The values of the design variables.

        Raises:
            ValueError: If the dataset name ``index_dataset`` already exists
                in the group of design variables.
        """
        str_index_dataset = str(index_dataset)
        if str_index_dataset in design_vars_group:
            raise ValueError(
                f"Dataset name '{str_index_dataset}' already exists "
                "in the group of design variables."
            )

        design_vars_group.create_dataset(
            str_index_dataset, data=design_vars_values.wrapped_array
        )

    def __add_hdf_output_dataset(
        self,
        index_dataset: int,
        keys_group: h5py.Group,
        values_group: h5py.Group,
        output_values: Mapping[str, float | ndarray | list],
        output_name_to_idx: Mapping[str, int] | None = None,
    ) -> None:
        """Add new outputs to the hdf group of output values.

        Args:
            index_dataset: The index of the new HDF entry.
            keys_group: The HDF group of the output names.
            values_group: The HDF group of the output values.
            output_values: The output values.
            output_name_to_idx: The indices of the output names in ``output_values``.
                If ``None``, these indices are automatically built using the
                order of the names in ``output_values``.
                These indices are used to build the dataset of output vectors.
        """
        self.__add_hdf_name_output(
            index_dataset, keys_group, list(output_values.keys())
        )

        if not output_name_to_idx:
            output_name_to_idx = dict(zip(output_values, range(len(output_values))))

        # We separate scalar data from vector data in the hdf file.
        # Scalar data are first stored into a list (``values``),
        # then added to the hdf file.
        # Vector data are directly added to the hdf file.
        values = []
        for name, value in output_values.items():
            idx_value = output_name_to_idx[name]
            if isinstance(value, (ndarray, list)):
                self.__add_hdf_vector_output(
                    index_dataset, idx_value, values_group, value
                )
            else:
                values.append(value)

        if values:
            self.__add_hdf_scalar_output(index_dataset, values_group, values)

    @staticmethod
    def __get_missing_hdf_output_dataset(
        index_dataset: int,
        keys_group: h5py.Group,
        output_values: DatabaseValueType,
    ) -> ReturnedHdfMissingOutputType:
        """Return the missing values in the HDF group of the output names.

        Compare the keys of ``output_values`` with the existing names
        in the group of the output names ``keys_group`` in order to know which
        outputs are missing.

        Args:
            index_dataset: The index of the new HDF entry.
            keys_group: The HDF group of the output names.
            output_values: The output values to be compared with.

        Returns:
            The missing values.
            The indices of the missing outputs.

        Raises:
            ValueError: If the index of the dataset does not correspond to
                an existing dataset.
        """
        name = str(index_dataset)
        if name not in keys_group:
            raise ValueError(f"The dataset named '{name}' does not exist.")

        existing_output_names = {out.decode() for out in keys_group[name]}
        all_output_names = set(output_values)
        missing_names = all_output_names - existing_output_names

        if not missing_names:
            return {}, {}

        missing_name_values = {name: output_values[name] for name in missing_names}
        all_output_idx_mapping = dict(zip(output_values, range(len(output_values))))
        missing_names_idx_mapping = {
            name: all_output_idx_mapping[name] for name in missing_names
        }

        return missing_name_values, missing_names_idx_mapping

    @staticmethod
    def __add_hdf_name_output(
        index_dataset: int, keys_group: h5py.Group, keys: list[str]
    ) -> None:
        """Add new output names to the HDF group of output names.

        Create a dataset in the group of output names
        if the dataset index is not found in the group.
        If the dataset already exists, the new names are appended
        to the existing dataset.

        Args:
            index_dataset: The index of the new HDF entry.
            keys_group: The HDF group of the output names.
            keys: The names that must be added.
        """
        name = str(index_dataset)
        keys = array(keys, dtype=bytes_)
        if name not in keys_group:
            keys_group.create_dataset(
                name, data=keys, maxshape=(None,), dtype=h5py.string_dtype()
            )
        else:
            offset = len(keys_group[name])
            keys_group[name].resize((offset + len(keys),))
            keys_group[name][offset:] = keys

    def __add_hdf_scalar_output(
        self, index_dataset: int, values_group: h5py.Group, values: list[float]
    ) -> None:
        """Add new scalar values to the HDF group of output values.

        Create a dataset in the group of output values
        if the dataset index is not found in the group.
        If the dataset already exists, the new values are appended to
        the existing dataset.

        Args:
            index_dataset: The index of the new HDF entry.
            values_group: The HDF group of the output values.
            values: The scalar values that must be added.
        """
        name = str(index_dataset)
        if name not in values_group:
            values_group.create_dataset(
                name, data=self.__to_real(values), maxshape=(None,), dtype=float64
            )
        else:
            offset = len(values_group[name])
            values_group[name].resize((offset + len(values),))
            values_group[name][offset:] = self.__to_real(values)

    def __add_hdf_vector_output(
        self,
        index_dataset: int,
        idx_sub_group: int,
        values_group: h5py.Group,
        value: ArrayLike,
    ) -> None:
        """Add a new vector of values to the HDF group of output values.

        Create a subgroup dedicated to vectors in the group of output
        values.
        Inside this subgroup, a new dataset is created for each vector.
        If the subgroup already exists, it is just appended.
        Otherwise, the sub-group is created.

        Args:
            index_dataset: The index of the HDF entry.
            idx_sub_group: The index of the dataset in the subgroup of vectors.
            values_group: The HDF group of the output values.
            value: The vector which is added to the group.

        Raises:
            ValueError: If the index of the dataset in the subgroup of vectors
                already exist.
        """
        sub_group_name = f"arr_{index_dataset}"

        if sub_group_name not in values_group:
            sub_group = values_group.require_group(sub_group_name)
        else:
            sub_group = values_group[sub_group_name]

        if str(idx_sub_group) in sub_group:
            raise ValueError(
                f"Dataset name '{idx_sub_group}' already exists "
                f"in the sub-group of array output '{sub_group_name}'."
            )

        sub_group.create_dataset(
            str(idx_sub_group), data=self.__to_real(value), dtype=float64
        )

    def __append_hdf_output(
        self,
        index_dataset: int,
        keys_group: h5py.Group,
        values_group: h5py.Group,
        output_values: DatabaseValueType,
    ) -> None:
        """Append the existing HDF datasets of the outputs with new values.

        Find the values among ``output_values`` that do not
        exist in the HDF datasets and append them to the datasets.

        Args:
            index_dataset: The index of the existing hdf5 entry.
            keys_group: The HDF group of the output names.
            values_group: The HDF group of the output values.
            output_values: The output values. Only the values that
                do not exist in the dataset will be appended.
        """
        added_values, mapping_to_idx = self.__get_missing_hdf_output_dataset(
            index_dataset, keys_group, output_values
        )
        if added_values:
            self.__add_hdf_output_dataset(
                index_dataset,
                keys_group,
                values_group,
                added_values,
                output_name_to_idx=mapping_to_idx,
            )

    def __create_hdf_input_output(
        self,
        index_dataset: int,
        design_vars_group: h5py.Group,
        keys_group: h5py.Group,
        values_group: h5py.Group,
        input_values: HashableNdarray,
        output_values: DatabaseValueType,
    ) -> None:
        """Create the new HDF datasets for the given inputs and outputs.

        Useful when exporting the database to an HDF file.

        Args:
            index_dataset: The index of the new HDF entry.
            design_vars_group: The HDF group of the design variable values.
            keys_group: The HDF group of the output names.
            values_group: The HDF group of the output values.
            input_values: The input values.
            output_values: The output values.
        """
        self.__add_hdf_input_dataset(index_dataset, design_vars_group, input_values)
        self.__add_hdf_output_dataset(
            index_dataset, keys_group, values_group, output_values
        )

    def to_file(
        self,
        database: Database,
        file_path: str | Path = "optimization_history.h5",
        append: bool = False,
    ) -> None:
        """Export the optimization database to an HDF file.

        Args:
            database: The database to export.
            file_path: The path of the HDF file.
            append: Whether to append the data to the file.
        """
        with h5py.File(file_path, "a" if append else "w") as h5file:
            design_vars_grp = h5file.require_group("x")
            keys_group = h5file.require_group("k")
            values_group = h5file.require_group("v")
            index_dataset = 0

            # The append mode loops over the last stored entries in order to
            # check whether some new outputs have been added.
            # However, if the hdf file has been re-written by a previous function
            # (such as OptimizationProblem.to_file),
            # there is no existing database inside the hdf file.
            # In such case, we have to check whether the design
            # variables group exists because otherwise the function tries to
            # append something empty.
            if append and len(design_vars_grp) != 0:
                input_values_to_idx = dict(
                    zip(database.keys(), range(len(database.keys())))
                )

                for input_values in self.__pending_arrays:
                    output_values = database[input_values]
                    index_dataset = input_values_to_idx[input_values]

                    if str(index_dataset) in design_vars_grp:
                        self.__append_hdf_output(
                            index_dataset, keys_group, values_group, output_values
                        )
                    else:
                        self.__create_hdf_input_output(
                            index_dataset,
                            design_vars_grp,
                            keys_group,
                            values_group,
                            input_values,
                            output_values,
                        )
            else:
                for input_values, output_values in database.items():
                    self.__create_hdf_input_output(
                        index_dataset,
                        design_vars_grp,
                        keys_group,
                        values_group,
                        input_values,
                        output_values,
                    )
                    index_dataset += 1

        self.__pending_arrays.clear()

    @staticmethod
    def update_from_file(
        database: Database,
        file_path: str | Path = "optimization_history.h5",
    ) -> None:
        """Update the current database from an HDF file.

        Args:
            database: The database to update.
            file_path: The path of the HDF file.
        """
        with h5py.File(file_path) as h5file:
            design_vars_grp = h5file["x"]
            keys_group = h5file["k"]
            values_group = h5file["v"]

            for raw_index in range(len(design_vars_grp)):
                str_index = str(raw_index)
                keys = [k.decode() for k in get_hdf5_group(keys_group, str_index)]

                array_name = f"arr_{str_index}"
                if array_name in values_group:
                    names_to_arrays = {
                        keys[int(k)]: array(v)
                        for k, v in values_group[array_name].items()
                    }
                else:
                    names_to_arrays = {}

                if str_index in values_group:
                    scalar_dict = dict(
                        zip(
                            (k for k in keys if k not in names_to_arrays),
                            get_hdf5_group(values_group, str_index),
                        )
                    )
                else:
                    scalar_dict = {}

                scalar_dict.update(names_to_arrays)
                database.store(array(design_vars_grp[str_index]), scalar_dict)

    def add_pending_array(self, data: HashableNdarray) -> None:
        """Record an array for later exporting to disk.

        Args:
            data: The data to be exported.
        """
        self.__pending_arrays.append(data)
