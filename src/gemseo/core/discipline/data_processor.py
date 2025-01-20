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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The data conversion processors."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from numpy import array
from numpy import complex128

from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.typing import MutableStrKeyMapping
    from gemseo.typing import StrKeyMapping


class DataProcessor(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base class for pre- and post-processing data.

    This is useful to cast the data types, since the |g| kernel only handles numpy
    arrays, and interfaces to disciplinary tools, workflow engines or environment can
    require different types.
    """

    @abstractmethod
    def pre_process_data(self, data: StrKeyMapping) -> MutableStrKeyMapping:
        """Pre-process data.

        Args:
            data: The data to process.

        Returns:
            The processed data.
        """

    # TODO: post_process shall only take care of the output data, the input data
    # shall be restored from the original data passed to pre_process.
    @abstractmethod
    def post_process_data(self, data: StrKeyMapping) -> MutableStrKeyMapping:
        """Execute a post-processing of the output data.

        Args:
            data: The data to process.

        Returns:
            The processed data.
        """


class FloatDataProcessor(DataProcessor):
    """A data preprocessor that converts all scalar input data to floats.

    It converts all discipline output data to numpy arrays
    """

    def pre_process_data(self, data: StrKeyMapping) -> MutableStrKeyMapping:  # noqa: D102
        processed_data = dict(data)
        for key, val in data.items():
            if len(val) == 1:
                processed_data[key] = float(val[0])
            else:
                processed_data[key] = [float(val_i) for val_i in val]
        return processed_data

    def post_process_data(self, data: StrKeyMapping) -> MutableStrKeyMapping:  # noqa: D102
        processed_data = dict(data)
        for key, val in data.items():
            if not hasattr(val, "__len__"):
                processed_data[key] = array([val])
            else:
                processed_data[key] = array(val)
        return processed_data


class ComplexDataProcessor(DataProcessor):
    """Data preprocessor to convert complex arrays to float arrays back and forth."""

    def pre_process_data(self, data: StrKeyMapping) -> MutableStrKeyMapping:  # noqa: D102
        processed_data = dict(data)
        for key, val in data.items():
            processed_data[key] = array(val.real)
        return processed_data

    def post_process_data(self, data: StrKeyMapping) -> MutableStrKeyMapping:  # noqa: D102
        processed_data = dict(data)
        for key, val in data.items():
            processed_data[key] = array(val, dtype=complex128)
        return processed_data


class NameMapping(DataProcessor):
    """A data preprocessor to map process level names to local discipline names."""

    mapping: Mapping[str, str]
    """The mapping structure of the form ``{global_name: local_name}``."""

    reverse_mapping: Mapping[str, str]
    """The reverse mapping structure of the form ``{local_name: global_name}``."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        """
        Args:
            mapping: A mapping structure of the form ``{global_name: local_name}``
                where ``global_name`` must be consistent
                with the grammar of the discipline.
                The local name is the data provided
                to the :meth:`.Discipline._run` method.
                When missing,
                the global name is the local name.
        """  # noqa: D205, D212, D415
        super().__init__()
        # TODO: API: make those private.
        self.mapping = mapping
        self.reverse_mapping = {
            local_key: global_key for global_key, local_key in mapping.items()
        }

    def pre_process_data(self, data: StrKeyMapping) -> MutableStrKeyMapping:  # noqa: D102
        get_local_name = self.mapping.get
        return {
            get_local_name(global_key, global_key): value
            for global_key, value in data.items()
        }

    def post_process_data(self, data: StrKeyMapping) -> MutableStrKeyMapping:  # noqa: D102
        get_global_name = self.reverse_mapping.get
        return {
            get_global_name(local_key, local_key): value
            for local_key, value in data.items()
        }
