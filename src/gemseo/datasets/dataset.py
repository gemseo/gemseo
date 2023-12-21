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
"""A generic data structure with entries and variables.

The concept of dataset is a key element for machine learning, post-processing, data
analysis, ...

A :class:`.Dataset` is a pandas
`MultiIndex DataFrame <https://pandas.pydata.org/docs/user_guide/advanced.html>`_
storing series of data
representing the values of multidimensional features
belonging to different groups of features.

A :class:`.Dataset` can be set
either from a file (:meth:`~.Dataset.from_csv` and :meth:`~.Dataset.from_txt`)
or from a NumPy array (:meth:`~.Dataset.from_array`),
and can be enriched from a group of variables (:meth:`~.Dataset.add_group`)
or from a single variable (:meth:`~.Dataset.add_variable`).

An :class:`.AbstractFullCache` or an :class:`.OptimizationProblem`
can also be exported to a :class:`.Dataset`
using the methods :meth:`.AbstractFullCache.to_dataset`
and :meth:`.OptimizationProblem.to_dataset`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from numbers import Number
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Final
from typing import Literal
from typing import Union
from typing import overload

from docstring_inheritance import GoogleDocstringInheritanceMeta
from numpy import arange
from numpy import array
from numpy import atleast_1d
from numpy import int64 as np_int64
from numpy import isin
from numpy import ndarray
from numpy import newaxis
from numpy import setdiff1d
from pandas import DataFrame
from pandas import MultiIndex
from pandas import read_csv

from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from pathlib import Path

    from pandas._typing import Axes
    from pandas._typing import Dtype

LOGGER = logging.getLogger(__name__)

StrColumnType = Union[str, Iterable[str]]
IndexType = Union[str, int, Iterable[Union[str, int]]]
DataType = Union[ndarray, Iterable[Any], Any]
ComponentType = Union[int, Iterable[int]]


class Dataset(DataFrame, metaclass=GoogleDocstringInheritanceMeta):
    """A generic data structure with entries and variables.

    A variable is defined by a name and a number of components.
    For instance,
    the variable ``"x"`` can have ``2`` components:
    ``0`` and ``1``.
    Or the variable ``y`` can have ``4`` components:
    ``"a"``, ``"b"``, ``"c"`` and ``"d"``.

    A variable belongs to a group of variables (default: :attr:`.DEFAULT_GROUP`).
    Two variables can have the same name;
    only the tuple ``(group_name, variable_name)`` is unique
    and is therefore called a *variable identifier*.

    Based on a set of variable identifiers,
    :class:`.Dataset` is a collection of entries
    corresponding to a set of variable identifiers.
    An entry corresponds to an index of the :class:`.Dataset`.

    A :class:`.Dataset` is a special pandas DataFrame
    with the multi-index ``(group_name, variable_name, component)``.
    It must be built from the methods
    :meth:`.add_variable`, :meth:`.add_group`,
    :meth:`.from_array`,
    :meth:`.from_txt` and :meth:`.from_csv`.

    Miscellaneous information that is not specific to an entry of the dataset
    can be stored in the dictionary :attr:`.misc`,
    as ``dataset.misc["year"] = 2023``.

    Notes:
        The columns of a data structure
        (NumPy array, ``DataFrame``, :class:`.Dataset`, ...)
        are called *features*.
        The features of a :class:`.Dataset` include
        all the components of all the variables of all the groups.

    Warnings:
        A :class:`.Dataset` behaves like any multi-index DataFrame
        but its instantiation using the constructor ``dataset = Dataset(data, ...)``
        can lead to some inconsistencies
        (multi-index levels, index values, dtypes, ...).
        Hence, the construction from the dedicated methods is recommended,
        e.g. ``dataset = Dataset(); dataset.add_variable("x", data)``.

    See Also:
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    """

    name: str
    """The name of the dataset."""

    misc: dict[str, Any]
    """Miscellaneous information specific to the dataset, and not to an entry."""

    COLUMN_LEVEL_NAMES: Final[tuple[str, str, str]] = (
        "GROUP",
        "VARIABLE",
        "COMPONENT",
    )
    """The names of the column levels of the multi-index DataFrame."""

    __GROUP_LEVEL: Final[int] = 0
    """The group level in the multi-index column."""

    __VARIABLE_LEVEL: Final[int] = 1
    """The variable level in the multi-index column."""

    __COMPONENT_LEVEL: Final[int] = 2
    """The component level in the multi-index column."""

    PARAMETER_GROUP: Final[str] = "parameters"
    """The group name for the parameters."""

    GRADIENT_GROUP: Final[str] = "gradients"
    """The group name for the gradients."""

    DEFAULT_GROUP: ClassVar[str] = PARAMETER_GROUP
    """The default group name for the variables."""

    DEFAULT_VARIABLE_NAME: ClassVar[str] = "x"
    """The default name for the variable set with :meth:`add_group`."""

    # DataFrame._metadata lists the normal properties
    # which will be passed to manipulation results,
    # e.g. dataset.get_view(group_names=group_names).name
    # normal properties
    _metadata: ClassVar[list[str]] = ["name", "misc"]

    def __init__(
        self,
        data: ndarray | Iterable | dict | DataFrame | None = None,
        index: Axes | None = None,
        columns: Axes | None = None,
        dtype: Dtype | None = None,
        copy: bool | None = None,
        *,
        dataset_name: str = "",
    ) -> None:
        """
        Args:
            data: See :class:`.DataFrame`.
            index: See :class:`.DataFrame`.
            columns:See :class:`.DataFrame`.
            dtype: See :class:`.DataFrame`.
            copy: See :class:`.DataFrame`.
            dataset_name: The name of the dataset.
        """  # noqa: D205, D212, D415
        if data is None and index is None and columns is None:
            columns = MultiIndex(
                levels=[[], [], []],
                codes=[[], [], []],
                names=self.COLUMN_LEVEL_NAMES,
            )
        super().__init__(
            data=data, index=index, columns=columns, dtype=dtype, copy=copy
        )
        self._reindex()
        self.name = dataset_name or self.__class__.__name__
        self.misc = {}

    @property
    def _constructor(self) -> type[Dataset]:
        return Dataset

    @property
    def group_names(self) -> list[str]:
        """The names of the groups of variables in alphabetical order.

        Warnings:
            The names are sorted with the Python function ``sorted``.
        """
        return sorted(self.columns.levels[self.__GROUP_LEVEL].unique())

    @property
    def variable_names(self) -> list[str]:
        """The names of the variables in alphabetical order.

        Warnings:
            The names are sorted with the Python function ``sorted``.
        """
        return sorted(self.columns.levels[self.__VARIABLE_LEVEL].unique())

    @property
    def variable_identifiers(self) -> list[tuple[str, str]]:
        """The variable identifiers.

        A variable identifier is the tuple ``(group_name, variable_name)``.

        Notes:
            A variable name can belong to more than one group
            while a variable identifier is unique
            as a group name is unique.

        Warnings:
            The names are sorted with the Python function ``sorted``.
        """
        return sorted(self.columns.droplevel(self.__COMPONENT_LEVEL).unique())

    @property
    def variable_names_to_n_components(self) -> dict[str, int]:
        """The names of the variables bound to their number of components."""
        return {
            variable_name: len(
                self.get_view(variable_names=variable_name).columns.get_level_values(
                    self.__COMPONENT_LEVEL
                )
            )
            for variable_name in self.variable_names
        }

    @property
    def group_names_to_n_components(self) -> dict[str, int]:
        """The names of the groups bound to their number of components."""
        return {
            group_name: len(
                self.get_view(group_names=group_name).columns.get_level_values(
                    self.__COMPONENT_LEVEL
                )
            )
            for group_name in self.group_names
        }

    def get_group_names(self, variable_name: str) -> list[str]:
        """Return the names of the groups that contain a variable.

        Args:
            variable_name: The name of the variable.

        Returns:
            The names of the groups that contain the variable.

        Warnings:
            The names are sorted with the Python function ``sorted``.
        """
        # TODO: remove Try/Except when using exclusively Pandas>=2.0
        try:
            return sorted(
                self.get_view(variable_names=variable_name)
                .columns.get_level_values(self.__GROUP_LEVEL)
                .unique()
            )
        except KeyError:
            return []

    def get_variable_names(self, group_name: str) -> list[str]:
        """Return the names of the variables contained in a group.

        Args:
            group_name: The name of the group.

        Notes:
            Assure compatibility pandas 1 and 2
            by returning an empty list if KeyError is raised.

        Returns:
            The names of the variables contained in the group.

        Warnings:
            The names are sorted with the Python function ``sorted``.
        """
        # TODO: remove Try/Except when using exclusively Pandas>=2.0
        try:
            return sorted(
                self.get_view(group_names=group_name)
                .columns.get_level_values(self.__VARIABLE_LEVEL)
                .unique()
            )
        except KeyError:
            return []

    def get_variable_components(self, group_name: str, variable_name: str) -> list[int]:
        """Return the components of a given variable.

        Args:
            group_name: The name of the group.
            variable_name: The name of the variable.

        Notes:
            Assure compatibility pandas 1 and 2
            by returning an empty list if KeyError is raised.

        Returns:
             The components of the variables.
        """
        # TODO: remove Try/Except when using exclusively Pandas>=2.0
        try:
            return (
                self.get_view(group_name, variable_name)
                .columns.get_level_values(self.__COMPONENT_LEVEL)
                .tolist()
            )
        except KeyError:
            return []

    def get_normalized(
        self,
        excluded_variable_names: StrColumnType = (),
        excluded_group_names: StrColumnType = (),
        use_min_max: bool = True,
        center: bool = False,
        scale: bool = False,
    ) -> Dataset:
        r"""Return a normalized copy of the dataset.

        Args:
            excluded_variable_names: The names of the variables not to be normalized.
                If empty, normalize all the variables.
            excluded_group_names: The names of the groups not to be normalized.
                If empty, normalize all the groups.
            use_min_max: Whether to use the geometric normalization
                :math:`(x-\min(x))/(\max(x)-\min(x))`.
            center: Whether to center the variables so that they have a zero mean.
            scale: Whether to scale the variables so that they have a unit variance.

        Returns:
            A normalized dataset.
        """
        output_dataset = self.copy()
        variables_to_normalize = setdiff1d(self.variable_names, excluded_variable_names)
        groups_to_normalize = setdiff1d(self.group_names, excluded_group_names)
        data = output_dataset.get_view(
            group_names=groups_to_normalize, variable_names=variables_to_normalize
        ).to_numpy()

        if use_min_max:
            data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

        if center:
            data -= data.mean(axis=0)

        if scale:
            data /= data.std(axis=0)

        output_dataset.update_data(
            data,
            group_names=groups_to_normalize,
            variable_names=variables_to_normalize,
        )

        return output_dataset

    def update_data(
        self,
        data: DataType,
        group_names: StrColumnType = (),
        variable_names: StrColumnType = (),
        components: ComponentType = (),
        indices: IndexType = (),
    ) -> None:
        """Replace some existing indices and columns by a deepcopy of ``data``.

        Args:
            data: The new data to be inserted in the dataset.
            group_names: The name(s) of the group(s) corresponding to these data.
                If empty, consider all the groups.
            variable_names: The name(s) of the variables(s) corresponding to these data.
                If empty, consider all the variables of the considered groups.
            components: The component(s) corresponding to these data.
                If empty, consider all the components of the considered variables.
            indices: The index (indices) of the dataset into which these data is to be
                inserted. If empty, consider all the indices.

        Notes:
            Changing the type of data can turn a view into a copy.
        """
        # Using Multi-index may lead to inconsistency between the slices and the result.
        # Setting the sortorder to 0 seem to solve this inconsistency.
        # We set this parameter only for extracting data,
        # and set it to its previous value.
        original_sortorder = self.columns.sortorder
        self.columns.sortorder = 0
        self.loc[
            self.__to_slice_or_list(indices),
            (
                self.__to_slice_or_list(group_names),
                self.__to_slice_or_list(variable_names),
                self.__to_slice_or_list(components),
            ),
        ] = data
        self.columns.sortorder = original_sortorder

    def add_variable(
        self,
        variable_name: str,
        data: DataType,
        group_name: str = DEFAULT_GROUP,
        components: ComponentType = (),
    ) -> None:
        """Add the data related to a variable.

        If the variable does not exist, it is added.
        If the variable already exists, non-existing components can be added,
        when specified.
        It is impossible to add components that have already been added.

        Args:
            variable_name: The name of the variable.
            data: The data.
            group_name: The name of the group related to this variable.
            components: The component(s) considered.
               If empty, use ``[0, ..., n_features]``.

        Warnings:
            The shape of ``data`` must be consistent
            with the number of entries of the dataset.
            If the dataset is empty,
            the number of entries will be deducted from ``data``.

        Notes:
            The data can be passed as:

            - an array shaped as ``(n_entries, n_features)``,
            - an array shaped as ``(1, n_features)``
              that will be reshaped as ``(n_entries, n_features)``
              by replicating the original array ``n_entries`` times,
            - an array shaped as ``(n_entries,)``
              that will be reshaped as ``(n_entries, 1)``,
            - a scalar that will be converted into an array
              shaped as ``(n_entries, 1)`` if ``components`` is empty
              or ``(n_entries, n_features)``
              where ``n_features`` will be deducted from ``components``.

        Raises:
            ValueError: If the group already has the added components
                of the variable named ``variable_name``.
        """
        n_components = 1 if isinstance(components, int) else len(components)
        data = self.__force_to_2d_array(data, n_components)

        if components:
            components = atleast_1d(components).astype(np_int64)
        else:
            components = arange(data.shape[1], dtype=np_int64)

        self.__check_existence_variable_components(
            group_name, variable_name, components
        )
        self.__check_data_shape_consistency(
            data, len(self) or data.shape[0], len(components)
        )
        self.__transform_multi_index_column_to_single_index_column()
        apply_scalar_to_all_entries = len(data) == 1 and not self.empty

        columns = [(group_name, variable_name, component) for component in components]
        value = data[0] if apply_scalar_to_all_entries else data
        self[columns] = value

        self.__transform_single_index_column_to_multi_index_column()
        self._reindex()

    def __check_existence_variable_components(
        self, group_name, variable_name, components
    ):
        """Check if the variable components are already in the dataset."""
        if isin(
            components, self.get_variable_components(group_name, variable_name)
        ).any():
            raise ValueError(
                f"The group {group_name!r} "
                f"has already a variable {variable_name!r} defined."
            )

    @staticmethod
    def __force_to_2d_array(
        data: DataType, n_components: int = 0
    ) -> ndarray[(Any, Any), Any]:
        """Force data to be a 2D NumPy array.

        Args:
            data: A scalar, or an array shaped as ``(n,)`` or ``(n, n_components)``.
            n_components: Only used when ``data`` is a scalar.
                In that case,
                the returned array is constant and shaped as ``(1, n_components)``.

        Returns:
            A 2D NumPy array of shape ``(n, n_components)``.
        """
        if isinstance(data, Number) and n_components:
            return array([[data] * n_components])

        data = atleast_1d(data)
        if data.ndim == 1:
            return data[:, newaxis]

        return data

    def add_group(
        self,
        group_name: str,
        data: DataType,
        variable_names: StrColumnType = (),
        variable_names_to_n_components: dict[str, int] | None = None,
    ) -> None:
        """Add the data related to a new group.

        Args:
            group_name: The name of the group.
            data: The data.
            variable_names: The names of the variables.
                If empty, use :attr:`.DEFAULT_VARIABLE_NAME`.
            variable_names_to_n_components: The number of components of the variables.
                If ``variable_names`` is empty,
                this argument is not considered.
                If ``None``,
                assume that all the variables have a single component.

        Raises:
            ValueError: If the group already exists.
        """
        if group_name in self.group_names:
            raise ValueError(f"The group {group_name!r} is already defined.")

        data = self.__force_to_2d_array(data)
        n_rows, n_columns = data.shape
        if not variable_names:
            variables = [(self.DEFAULT_VARIABLE_NAME, i) for i in range(n_columns)]
        else:
            variables = []
            variable_names_to_n_components = variable_names_to_n_components or {}
            for variable in atleast_1d(variable_names):
                n_components = variable_names_to_n_components.get(variable, 1)
                variables.extend([(variable, i) for i in range(n_components)])

        self.__check_data_shape_consistency(data, n_rows, len(variables))
        for (variable_name, component), data_column in zip(variables, data.T):
            self.add_variable(
                variable_name,
                data_column[:, newaxis],
                group_name=group_name,
                components=component,
            )

    @staticmethod
    def __check_data_shape_consistency(
        data: ndarray[(Any, Any), float], n_rows: int, n_columns: int
    ) -> None:
        """Check the consistency of a data array with numbers of components and indices.

        Args:
            data: The data to be checked.
            n_rows: The expected number of rows.
            n_columns: The expected number of columns.

        Raises:
            ValueError: When the shape of the data is inconsistent.
        """
        if data.shape not in {(n_rows, n_columns), (1, n_columns)}:
            raise ValueError(
                "The data shape "
                f"must be ({n_rows}, {n_columns}) or (1, {n_columns}); "
                f"got {data.shape} instead."
            )

    def get_view(
        self,
        group_names: StrColumnType = (),
        variable_names: StrColumnType = (),
        components: ComponentType = (),
        indices: IndexType = (),
    ) -> Dataset:
        """Return a specific group of rows and columns of the dataset.

        Args:
            group_names: The name(s) of the group(s).
                If empty, consider all the groups.
            variable_names: The name(s) of the variables(s).
                If empty, consider all the variables of the considered groups.
            components: The component(s).
                If empty, consider all the components of the considered variables.
            indices: The index (indices) of the dataset
                into which these data is to be inserted.
                If empty, consider all the indices.

        Notes:
            The order asked by the user is preserved for the returned Dataset.
            See also :meth:`.loc`.

        Returns:
            The specific group of rows and columns of the dataset.
        """
        # Using Multi-index may lead to inconsistency between the slices and the result.
        # Setting the sortorder to 0 seem to solve this inconsistency.
        # We set this parameter only for extracting data,
        # and then set it to its previous value.
        original_sortorder = self.columns.sortorder
        self.columns.sortorder = 0
        data = self.loc[
            self.__to_slice_or_list(indices),
            (
                self.__to_slice_or_list(group_names),
                self.__to_slice_or_list(variable_names),
                self.__to_slice_or_list(components),
            ),
        ]
        self.columns.sortorder = original_sortorder
        return data

    @staticmethod
    def __to_slice_or_list(obj: Any) -> slice | list[Any]:
        """Convert an object to a ``slice`` or a ``list``.

        Args:
            obj: The object.

        Returns:
            The object as a ``slice`` or a ``list``.
        """
        if isinstance(obj, slice):
            return obj

        if not isinstance(obj, ndarray) and obj != 0 and not obj:
            return slice(None)

        return atleast_1d(obj).tolist()

    def rename_group(self, group_name: str, new_group_name: str) -> None:
        """Rename a group.

        Args:
            group_name: The group to rename.
            new_group_name: The new group name.
        """
        self.rename(columns={group_name: new_group_name}, level=0, inplace=True)

    def rename_variable(
        self,
        variable_name: str,
        new_variable_name: str,
        group_name: str = "",
    ) -> None:
        """Rename a variable.

        Args:
            variable_name: The name of the variable.
            new_variable_name: The new name of the variable.
            group_name: The group of the variable.
                If empty,
                change the name of all the variables matching ``variable_name``.
        """
        if group_name:
            components = self.get_variable_components(group_name, variable_name)
            self.__transform_multi_index_column_to_single_index_column()
            self.rename(
                columns={
                    (group_name, variable_name, component): (
                        group_name,
                        new_variable_name,
                        component,
                    )
                    for component in components
                },
                inplace=True,
            )
            self.__transform_single_index_column_to_multi_index_column()
        else:
            self.rename(
                columns={variable_name: new_variable_name},
                level=self.__VARIABLE_LEVEL,
                inplace=True,
            )

    def __transform_multi_index_column_to_single_index_column(self) -> None:
        """Transform the multi-index columns into tuple columns."""
        self.columns = self.columns.to_numpy()

    def __transform_single_index_column_to_multi_index_column(self) -> None:
        """Transform the tuple columns into multi-index columns."""
        self.columns = MultiIndex.from_tuples(
            self.columns, names=self.COLUMN_LEVEL_NAMES
        )

    @classmethod
    def from_array(
        cls,
        data: DataType,
        variable_names: StrColumnType = (),
        variable_names_to_n_components: dict[str, int] | None = None,
        variable_names_to_group_names: dict[str, str] | None = None,
    ) -> Dataset:
        """Create a dataset from a NumPy array.

        Args:
            data: The data to be stored, with the shape (n_entries, n_components).
            variable_names: The names of the variables.
                If empty, use default names.
            variable_names_to_n_components: The number of components of the variables.
                If ``None``,
                assume that all the variables have a single component.
            variable_names_to_group_names: The groups of the variables.
                If ``None``,
                use :attr:`.Dataset.DEFAULT_GROUP` for all the variables.

        Returns:
            A dataset built from the NumPy array.
        """
        if variable_names:
            variable_to_group = variable_names_to_group_names or {}
            variable_to_n_component = variable_names_to_n_components or {}
        else:
            _, n_total_components = data.shape
            variable_names = [
                "_".join([cls.DEFAULT_VARIABLE_NAME, str(component)])
                for component in arange(n_total_components)
            ]

            # Do not consider groups nor n_components.
            variable_to_group = {}
            variable_to_n_component = {}

        columns = []
        for variable in variable_names:
            group = variable_to_group.get(variable, cls.DEFAULT_GROUP)
            n_components = variable_to_n_component.get(variable, 1)
            columns.extend([
                (group, variable, component)
                for component in arange(n_components, dtype=np_int64)
            ])

        index = MultiIndex.from_tuples(columns, names=cls.COLUMN_LEVEL_NAMES)
        dataset = cls(data, columns=index)
        dataset._reindex()
        return dataset

    @classmethod
    def from_txt(
        cls,
        file_path: Path | str,
        variable_names: Iterable[str] = (),
        variable_names_to_n_components: dict[str, int] | None = None,
        variable_names_to_group_names: dict[str, str] | None = None,
        delimiter: str = ",",
        header: bool = True,
    ) -> Dataset:
        """Create a dataset from a text file.

        See Also:
            If the file contains multi-index information
            and not just an array,
            the :meth:`.from_csv` method is better suited.

        Args:
            file_path: The path to the file containing the data.
            variable_names: The names of the variables.
                If empty and ``header`` is ``True``,
                read the names from the first line of the file.
                If empty and ``header`` is ``False``,
                use default names
                based on the patterns the :attr:`.DEFAULT_NAMES`
                associated with the different groups.
            variable_names_to_n_components: The number of components of the variables.
                If ``None``,
                assume that all the variables have a single component.
            variable_names_to_group_names: The groups of the variables.
                If ``None``,
                use :attr:`.DEFAULT_GROUP` for all the variables.
            delimiter: The field delimiter.
            header: Whether to read the names of the variables
                on the first line of the file.

        Returns:
            A dataset built from the text file.
        """
        header = "infer" if header else None
        return cls.from_array(
            read_csv(file_path, delimiter=delimiter, header=header).to_numpy(),
            variable_names,
            variable_names_to_n_components,
            variable_names_to_group_names,
        )

    @classmethod
    def from_csv(cls, file_path: Path | str, delimiter: str = ",") -> Dataset:
        """Set the dataset from a CSV file.

        The first three rows contain the values of the multi-index
        ``(column_name, variable_name, component)``.

        See Also:
            If the file does not contain multi-index information
            and not just an array,
            the method :meth:`.from_txt` is better suited.

        Args:
            file_path: The path to the file containing the data.
            delimiter: The field delimiter.

        Returns:
            A dataset built from the CSV file.
        """
        dataframe = read_csv(file_path, delimiter=delimiter, header=[0, 1, 2])
        dataframe.columns = dataframe.columns.set_levels(
            dataframe.columns.levels[cls.__COMPONENT_LEVEL].astype(np_int64),
            level=cls.__COMPONENT_LEVEL,
        )

        dataset = cls(dataframe)
        dataset.columns = dataset.columns.set_names(cls.COLUMN_LEVEL_NAMES)
        dataset._reindex()
        return dataset

    @overload
    def get_columns(
        self, variable_names: Iterable[str] = (), as_tuple: Literal[False] = False
    ) -> list[str]: ...

    @overload
    def get_columns(
        self, variable_names: Iterable[str] = (), as_tuple: Literal[True] = True
    ) -> list[tuple[str, str, int]]: ...

    def get_columns(
        self, variable_names: Iterable[str] = (), as_tuple: bool = False
    ) -> list[str | tuple[str, str, int]]:
        """Return the columns based on variable names.

        Args:
            variable_names: The names of the variables.
                If empty, use all the variables.
            as_tuple: Whether the variable identifiers are returned as tuples.

        Returns:
            The columns,
            either as a variable identifier ``(group_name, variable_name, component)``
            or as a variable component name ``"variable_name[component]"``
            (or ``"variable_name"`` if the dimension of the variable is 1).
        """
        columns = self.get_view(variable_names=variable_names).columns.to_list()
        if as_tuple:
            return columns

        return [
            repr_variable(
                column[self.__VARIABLE_LEVEL],
                column[self.__COMPONENT_LEVEL],
                size=len(
                    self.get_variable_components(
                        column[self.__GROUP_LEVEL], column[self.__VARIABLE_LEVEL]
                    )
                ),
            )
            for column in columns
        ]

    def transform_data(
        self,
        transformation: Callable[[ndarray], ndarray],
        group_names: StrColumnType = (),
        variable_names: StrColumnType = (),
        components: ComponentType = (),
        indices: IndexType = (),
    ) -> None:
        """Transform some data of the dataset.

        Args:
            transformation: The function transforming the variable,
                e.g. ``"lambda x: 2*x"``.
            group_names: The name(s) of the group(s) corresponding to these data.
                If empty, consider all the groups.
            variable_names: The name(s) of the variables(s) corresponding to these data.
                If empty, consider all the variables of the considered groups.
            components: The component(s) corresponding to these data.
                If empty, consider all the components of the considered variables.
            indices: The index (indices) of the dataset corresponding to these data.
                If empty, consider all the indices.
        """
        self.update_data(
            transformation(
                self.get_view(
                    group_names, variable_names, components, indices
                ).to_numpy()
            ),
            group_names,
            variable_names,
            components,
            indices,
        )

    def _reindex(self) -> None:
        """Reindex the dataframe."""

    def to_dict_of_arrays(
        self, by_group: bool = True, by_entry: bool = False
    ) -> (
        dict[str, ndarray | dict[str, ndarray]]
        | list[dict[str, ndarray | dict[str, ndarray]]]
    ):
        """Convert the dataset into a dictionary of NumPy arrays.

        Args:
            by_group: Whether the data are returned as
                ``{group_name: {variable_name: variable_values}}``.
                Otherwise,
                the data are returned either as ``{variable_name: variable_values}``
                if only one group contains the variable ``variable_name``
                or as ``{f"{group_name}:{variable_name}": variable_values}``
                if at least two groups contain the variable ``variable_name``.
            by_entry: Whether the data are returned as
                ``[{group_name: {variable_name: variable_value_1}}, ...]``,
                ``[{variable_name: variable_value_1}, ...]``
                or ``[{f"{group_name}:{variable_name}": variable_value_1}, ...]``
                according to ``by_group``.
                Otherwise,
                the data are returned as
                ``{group_name: {variable_name: variable_value_1}}``,
                ``{variable_name: variable_value_1}``
                or ``{f"{group_name}:{variable_name}": variable_value_1}``.

        Returns:
            The dataset expressed as a dictionary of NumPy arrays.
        """
        indices = self.index if by_entry else [()]
        if by_group:
            list_of_dict_of_arrays = [
                {
                    group_name: {
                        variable_name: self.get_view(
                            group_name, variable_name, indices=index
                        ).to_numpy()
                        for variable_name in self.get_variable_names(group_name)
                    }
                    for group_name in self.group_names
                }
                for index in indices
            ]
        else:
            list_of_dict_of_arrays = []
            for index in indices:
                dict_of_arrays = {}
                for group_name, variable_name in self.variable_identifiers:
                    if len(self.get_group_names(variable_name)) == 1:
                        name = variable_name
                    else:
                        name = f"{group_name}:{variable_name}"

                    dict_of_arrays[name] = self.get_view(
                        group_names=group_name,
                        variable_names=variable_name,
                        indices=index,
                    ).to_numpy()
                list_of_dict_of_arrays.append(dict_of_arrays)

        if by_entry:
            return [
                {k: v.ravel() for k, v in d.items()} for d in list_of_dict_of_arrays
            ]

        return list_of_dict_of_arrays[0]

    @property
    def summary(self) -> str:
        """A summary of the dataset."""
        string = MultiLineString()
        string.add(self.name)
        string.indent()
        string.add("Class: {}", self.__class__.__name__)
        string.add("Number of entries: {}", len(self))
        string.add("Number of variable identifiers: {}", len(self.variable_identifiers))
        string.add("Variables names and sizes by group:")
        string.indent()
        for group_name in self.group_names:
            variable_names = self.get_variable_names(group_name)
            variable_names_and_sizes = []
            for variable_name in variable_names:
                variable_size = len(
                    self.get_variable_components(group_name, variable_name)
                )
                variable_names_and_sizes.append(f"{variable_name} ({variable_size})")
            if variable_names_and_sizes:
                string.add(
                    "{}: {}",
                    group_name,
                    pretty_str(variable_names_and_sizes, use_and=True),
                )
        total = sum(self.group_names_to_n_components.values())
        string.dedent()
        string.add("Number of dimensions (total = {}) by group:", total)
        string.indent()
        for group_name, group_size in sorted(self.group_names_to_n_components.items()):
            string.add("{}: {}", group_name, group_size)
        return str(string)
