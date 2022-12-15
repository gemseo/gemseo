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

import logging
from abc import abstractmethod
from typing import Mapping

from numpy import array
from numpy import complex128

from gemseo.core.discipline_data import Data
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

LOGGER = logging.getLogger(__name__)


class DataProcessor(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base class for pre- and post-processing data.

    This is useful to cast the data types, since the |g| kernel only handles numpy
    arrays, and interfaces to disciplinary tools, workflow engines or environment can
    require different types.
    """

    @abstractmethod
    def pre_process_data(self, data: Data) -> Data:
        """Pre-process data.

        Args:
            data: The data to process.

        Returns:
            The processed data.
        """

    @abstractmethod
    def post_process_data(self, data: Data) -> Data:
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

    def pre_process_data(self, data: Data) -> Data:  # noqa: D102
        processed_data = data.copy()
        for key, val in data.items():
            if len(val) == 1:
                processed_data[key] = float(val[0])
            else:
                processed_data[key] = [float(val_i) for val_i in val]
        return processed_data

    def post_process_data(self, data: Data) -> Data:  # noqa: D102
        processed_data = data.copy()
        for key, val in data.items():
            if not hasattr(val, "__len__"):
                processed_data[key] = array([val])
            else:
                processed_data[key] = array(val)
        return processed_data


class ComplexDataProcessor(DataProcessor):
    """Data preprocessor to convert complex arrays to float arrays back and forth."""

    def pre_process_data(self, data: Data) -> Data:  # noqa: D102
        processed_data = data.copy()
        for key, val in data.items():
            processed_data[key] = array(val.real)
        return processed_data

    def post_process_data(self, data: Data) -> Data:  # noqa: D102
        processed_data = data.copy()
        for key, val in data.items():
            processed_data[key] = array(val, dtype=complex128)
        return processed_data


class NameMapping(DataProcessor):
    """A data preprocessor to map process level names to local discipline names."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        """
        Args:
            mapping: A mapping structure of the form ``{global_name: local_name}``
                where ``global_name`` must be consistent
                with the grammar of the discipline.
                The local name is the data provided
                to the :meth:`.MDODiscipline._run` method.
        """  # noqa: D205, D212, D415
        super().__init__()
        self.mapping = mapping
        self.reverse_mapping = {
            local_key: global_key for global_key, local_key in mapping.items()
        }

    def pre_process_data(self, data: Data) -> Data:  # noqa: D102
        mapping = self.mapping
        return {mapping[global_key]: value for global_key, value in data.items()}

    def post_process_data(self, data: Data) -> Data:  # noqa: D102
        reverse_mapping = self.reverse_mapping
        return {reverse_mapping[local_key]: value for local_key, value in data.items()}
